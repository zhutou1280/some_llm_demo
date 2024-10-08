import chromadb
import os

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
