import chromadb
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))

data_directory = os.path.join(current_directory, 'data')

chroma_client = chromadb.PersistentClient(path=data_directory)

# 检查集合是否已存在
collection_name = "img_collection"
existing_collections = chroma_client.list_collections()

img_collection = None

# 如果集合存在，获取它，否则创建新的
if collection_name in [col.name for col in existing_collections]:
    img_collection = chroma_client.get_collection(collection_name)
else:
    img_collection = chroma_client.create_collection(collection_name)

# 加载CLIP模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


current_directory = os.path.dirname(os.path.abspath(__file__))


def load_directory(image_folder):
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


def query_data(query_image_path):
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


def delete_db():
    chroma_client.delete_collection(collection_name)


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="根据命令行参数执行不同的操作")

    # 添加参数
    parser.add_argument('--load-dir', type=str, help="将该目录下的文件存储到数据库中")
    parser.add_argument('--query', type=str, help="查询某个路径的图片")
    parser.add_argument('--delete', type=bool)

    # 解析参数
    args = parser.parse_args()

    # 根据参数执行相应的逻辑
    if args.load_dir:
        load_directory(args.load_dir)
    elif args.query:
        query_data(args.query)
    elif args.delete:
        delete_db()
    else:
        print("请提供 --load-dir 或 --query 参数")


if __name__ == "__main__":
    main()
