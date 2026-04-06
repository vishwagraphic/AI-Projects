from chromadb.utils import embedding_functions
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# default_ef = embedding_functions.DefaultEmbeddingFunction()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# text = "Hello world."

# print(default_ef(text))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key = os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")
# collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=default_ef)
collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=openai_ef)

documents = [
    {"id": "doc1", "text": "Hello world."},
    {"id": "doc2", "text": "How are you today."},
    {"id": "doc3", "text": "Good bye, see you later"},
]

for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

query = "Hello world"

result = collection.query(query_texts=[query], n_results=2)

for idx, doc in enumerate(result["documents"][0]):
    print(result["ids"][0])
    print(result["distances"][0])
    doc_id = result["ids"][0][idx]
    distance = result["distances"][0][idx]
    print(f"Document ID: {doc_id}, Distance: {distance}, Text: {doc}")

