import chromadb
from chromadb.utils import embedding_functions
chroma_client = chromadb.Client()

collection_name = "my_collection"

default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=default_ef)


documents = [
    {"id": "doc1", "text": "Hello world."},
    {"id": "doc2", "text": "How are you today."},
    {"id": "doc3", "text": "Good bye, see you later"},
]

for document in documents:
    collection.upsert(ids=[document["id"]], documents=[document["text"]])
    result = collection.query(query_texts=["Cricket is wonderful game"], n_results=3)



for idx, doc in enumerate(result["documents"][0]):
    print(result["ids"][0])
    print(result["distances"][0])
    doc_id = result["ids"][0][idx]
    distance = result["distances"][0][idx]
    print(f"Document ID: {doc_id}, Distance: {distance}, Text: {doc}")
    