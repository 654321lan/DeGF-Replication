from transformers import AutoModel, AutoProcessor
import os

os.makedirs("./models", exist_ok=True)

# ====================== 下载 LLaVA-1.5-7B ======================
print("正在下载 LLaVA-1.5-7B（国内稳定源）...")

llava_model_name = "liuhaotian/LLaVA-1.5-7b-hf"
llava_save_path = "./models/llava-1.5-7b-hf"

# 自动从国内镜像下载
model = AutoModel.from_pretrained(llava_model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(llava_model_name, trust_remote_code=True)

# 保存到本地
model.save_pretrained(llava_save_path)
processor.save_pretrained(llava_save_path)

# ====================== 下载 SD v1.5 ======================
print("正在下载 Stable-Diffusion-v1-5...")

from diffusers import StableDiffusionPipeline
sd_model_name = "runwayml/stable-diffusion-v1-5"
sd_save_path = "./models/stable-diffusion-v1-5"

sd = StableDiffusionPipeline.from_pretrained(sd_model_name)
sd.save_pretrained(sd_save_path)

print("✅ 所有模型下载完成！")
print("LLaVA 路径:", llava_save_path)
print("SD 路径:", sd_save_path)