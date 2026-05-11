import json
import torch
import argparse
import random
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pope_json", required=True, help="POPE JSON file path")
    parser.add_argument("--out_file", default="./result.txt", help="Output file")
    args = parser.parse_args()

    model_path = "/root/autodl-tmp/models/llava-1.5-7b-hf"
    image_dir = "/root/autodl-tmp/DeGF/data/coco/val2014"

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    with open(args.pope_json, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    pred_list, label_list = [], []
    for item in tqdm(data):
        img = Image.open(os.path.join(image_dir, item['image'])).convert("RGB")
        question = item['text']
        label = 1 if item['label'] == 'yes' else 0

        # 使用 apply_chat_template 构造 prompt
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]}
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(prompt, images=img, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,       # 开启采样
                temperature=1.0,      # 温度=1
                top_p=1.0,            # 全词汇表
                top_k=0,              # 不限制 top_k
                num_beams=1,          # 贪婪解码
                repetition_penalty=1.0
            )
        answer = processor.decode(out[0], skip_special_tokens=True)
        if "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip().lower()
        else:
            answer = answer.strip().lower()
        pred = 1 if "yes" in answer else 0
        pred_list.append(pred)
        label_list.append(label)

    tp = sum(p==1 and l==1 for p,l in zip(pred_list, label_list))
    fp = sum(p==1 and l==0 for p,l in zip(pred_list, label_list))
    tn = sum(p==0 and l==0 for p,l in zip(pred_list, label_list))
    fn = sum(p==0 and l==1 for p,l in zip(pred_list, label_list))
    acc = (tp+tn)/len(pred_list)
    prec = tp/(tp+fp) if tp+fp>0 else 0
    rec = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0

    print(f"\n=== Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    with open(args.out_file, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")

if __name__ == "__main__":
    main()
