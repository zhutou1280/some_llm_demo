import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from db import img_collection

# 加载CLIP模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


current_directory = os.path.dirname(os.path.abspath(__file__))

image_folder = os.path.join(current_directory, 'imgs')

# 遍历文件夹中的所有图片文件
for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # 过滤支持的图片格式
        image_path = os.path.join(image_folder, filename)

        # 打开图片并转换为嵌入向量
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs)

        # 将图片嵌入和元数据（如文件名）添加到ChromaDB
        img_collection.add(
            ids=[filename],
            documents=[image_path],            # 使用图片的路径作为文档
            embeddings=image_embeds.tolist(),  # 图片的嵌入向量
            metadatas=[{"image_name": filename}]  # 可选元数据
        )

print(f"所有图片已成功存入 ChromaDB 中。")
