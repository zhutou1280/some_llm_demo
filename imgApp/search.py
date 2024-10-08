import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from db import img_collection


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

current_directory = os.path.dirname(os.path.abspath(__file__))

query_image_path = os.path.join(current_directory, 'search.png')
# 打开查询图片并转换为嵌入
query_image = Image.open(query_image_path)
inputs = processor(images=query_image, return_tensors="pt")
with torch.no_grad():
    query_embeds = model.get_image_features(**inputs)

# 在ChromaDB中查询最相似的图片
results = img_collection.query(
    query_embeddings=query_embeds.tolist(),
    n_results=2  # 返回最相似的5个结果
)

# 输出检索结果
for result in results['documents']:
    print(f"匹配图片: {result}")
