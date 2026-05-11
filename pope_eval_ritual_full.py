import json
import torch
import argparse
import torch.nn.functional as F
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm
import os
import random
from torchvision import transforms

def js_divergence(p, q):
    p = p.float()
    q = q.float()
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pope_json", required=True)
    parser.add_argument("--out_file", default="./result.txt")
    args = parser.parse_args()

    model_path = "/root/autodl-tmp/models/llava-1.5-7b-hf"
    image_dir = "/root/autodl-tmp/DeGF/data/coco/val2014"

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    aug_list = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5),
        transforms.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
        transforms.RandomResizedCrop(size=336),
    ]
    def random_transform(img):
        return random.choice(aug_list)(img)

    alpha_pos = 3.0
    alpha_neg = 1.0
    gamma = 0.1
    degf_beta = 0.1

    with open(args.pope_json, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    pred_list, label_list = [], []
    for item in tqdm(data):
        img_path = os.path.join(image_dir, item['image'])
        raw_img = Image.open(img_path).convert("RGB")
        question = item['text']
        label = 1 if item['label'] == 'yes' else 0

        trans_img = random_transform(raw_img)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]}
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs_orig = processor(prompt, images=raw_img, return_tensors="pt").to(model.device)
        inputs_trans = processor(prompt, images=trans_img, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out_orig = model.generate(
                **inputs_orig,
                max_new_tokens=1,
                output_logits=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            # 处理 logits 维度：可能是 (1, vocab_size) 或 (1,1,vocab_size)
            logits_orig = out_orig.logits[0]
            if logits_orig.dim() == 3:
                logits_orig = logits_orig[0, -1, :]
            else:
                logits_orig = logits_orig.squeeze()

            out_trans = model.generate(
                **inputs_trans,
                max_new_tokens=1,
                output_logits=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            logits_trans = out_trans.logits[0]
            if logits_trans.dim() == 3:
                logits_trans = logits_trans[0, -1, :]
            else:
                logits_trans = logits_trans.squeeze()

        prob_orig = F.softmax(logits_orig, dim=-1)
        prob_trans = F.softmax(logits_trans, dim=-1)
        jsd = js_divergence(prob_orig, prob_trans)

        if jsd < gamma:
            fused_logits = logits_orig + alpha_pos * logits_trans
        else:
            fused_logits = (1 + alpha_neg) * logits_orig - alpha_neg * logits_trans

        # Adaptive plausibility constraints (align with official DeGF cutoff)
        cutoff = torch.log(torch.tensor(degf_beta, device=logits_orig.device)) + logits_orig.max()
        fused_logits = fused_logits.masked_fill(logits_orig < cutoff, float("-inf"))

        next_token_id = torch.argmax(fused_logits).item()
        answer = processor.decode([next_token_id], skip_special_tokens=True).strip().lower()
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

    print(f"\n=== RITUAL Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    with open(args.out_file, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")

if __name__ == "__main__":
    main()
