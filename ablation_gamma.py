import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def js_divergence(p, q):
    p = p.float()
    q = q.float()
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).item()


def build_augmentations():
    return [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5),
        transforms.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
        transforms.RandomResizedCrop(size=336),
    ]


def eval_config(
    data,
    image_dir,
    processor,
    model,
    aug_list,
    gamma,
    alpha_pos,
    alpha_neg=1.0,
    degf_beta=0.1,
):
    pred_list, label_list = [], []
    for item in tqdm(data, desc=f"gamma={gamma}, alpha_pos={alpha_pos}"):
        img_path = os.path.join(image_dir, item["image"])
        raw_img = Image.open(img_path).convert("RGB")
        question = item["text"]
        label = 1 if item["label"] == "yes" else 0

        trans_img = random.choice(aug_list)(raw_img)

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
            out_trans = model.generate(
                **inputs_trans,
                max_new_tokens=1,
                output_logits=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        logits_orig = out_orig.logits[0]
        if logits_orig.dim() == 3:
            logits_orig = logits_orig[0, -1, :]
        else:
            logits_orig = logits_orig.squeeze()

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

        cutoff = torch.log(torch.tensor(degf_beta, device=logits_orig.device)) + logits_orig.max()
        fused_logits = fused_logits.masked_fill(logits_orig < cutoff, float("-inf"))

        next_token_id = torch.argmax(fused_logits).item()
        answer = processor.decode([next_token_id], skip_special_tokens=True).strip().lower()
        pred = 1 if "yes" in answer else 0
        pred_list.append(pred)
        label_list.append(label)

    acc = sum(p == l for p, l in zip(pred_list, label_list)) / len(pred_list)
    return acc


def main():
    set_seed(42)

    model_path = "/root/autodl-tmp/models/llava-1.5-7b-hf"
    pope_json = "/root/autodl-tmp/DeGF/data/pope/coco_pope_random.json"
    image_dir = "/root/autodl-tmp/DeGF/data/coco/val2014"

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    with open(pope_json, "r") as f:
        data = [json.loads(line) for line in f if line.strip()][:3000]

    aug_list = build_augmentations()

    gamma_values = [0.0, 0.1, 0.5]
    alpha_values = [1.0, 3.0, 5.0]

    results = {"gamma": {}, "alpha_pos": {}}

    for g in gamma_values:
        acc = eval_config(
            data=data,
            image_dir=image_dir,
            processor=processor,
            model=model,
            aug_list=aug_list,
            gamma=g,
            alpha_pos=3.0,
        )
        results["gamma"][str(g)] = acc
        print(f"gamma={g}, accuracy={acc:.4f}")

    for a in alpha_values:
        acc = eval_config(
            data=data,
            image_dir=image_dir,
            processor=processor,
            model=model,
            aug_list=aug_list,
            gamma=0.1,
            alpha_pos=a,
        )
        results["alpha_pos"][str(a)] = acc
        print(f"alpha_pos={a}, accuracy={acc:.4f}")

    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    gamma_x = [float(k) for k in results["gamma"].keys()]
    gamma_y = [results["gamma"][k] for k in results["gamma"].keys()]
    axes[0].plot(gamma_x, gamma_y, marker="o")
    axes[0].set_title("Gamma Ablation")
    axes[0].set_xlabel("gamma")
    axes[0].set_ylabel("accuracy")

    alpha_x = [float(k) for k in results["alpha_pos"].keys()]
    alpha_y = [results["alpha_pos"][k] for k in results["alpha_pos"].keys()]
    axes[1].plot(alpha_x, alpha_y, marker="o")
    axes[1].set_title("Alpha_pos Ablation")
    axes[1].set_xlabel("alpha_pos")
    axes[1].set_ylabel("accuracy")

    fig.tight_layout()
    fig.savefig("ablation_plots.png", dpi=200)
    print("Saved ablation_plots.png and ablation_results.json")


if __name__ == "__main__":
    main()
