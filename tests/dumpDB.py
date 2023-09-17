"""
*** PCnewsGPT Hilfsprogramm: Inhaltsausgabe Wissensdatenbank ***
"""

"""
Load Parameters, etc.
"""
from dotenv import load_dotenv
from os import environ as os_environ
load_dotenv()
persist_directory = os_environ.get('PERSIST_DIRECTORY','db')
collection_name = os_environ.get('COLLECTION_NAME','langchain')
embeddings_model_name = os_environ.get('EMBEDDINGS_MODEL_NAME','paraphrase-multilingual-mpnet-base-v2')


"""
Initial banner Message
"""
print(f"\nPCnewsGPT Dump der Wissensdatenbank in {persist_directory} (collection '{collection_name}') V0.1.2\n")


"""
Init Chroma & Embeddings & Collections
"""
import chromadb
from chromadb.config import Settings
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory=persist_directory
                                ))
from chromadb.utils import embedding_functions
chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embeddings_model_name)
collection = chroma_client.get_collection(name=collection_name, embedding_function=chroma_ef)

"""
Print DB
"""
data = collection.get()
ids = data['ids']
embeddings = data["embeddings"]
metadatas = data["metadatas"]
documents = data["documents"]
print("--- items, ids, metadata, documents ---")
items=len(ids)
for j in range(items):
    doc = documents[j].replace('\n', '\\n')   # no newlines in text
    print(f"({j+1}/{items}): {ids[j]}, '{metadatas[j]}', text-length={len(doc)}, text='{doc}'\n")
    