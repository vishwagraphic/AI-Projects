import os
import csv
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb 
from chromadb.utils import embedding_functions

class LLMModel:
    def __init__(self, model_type="openai"):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-3.5-turbo"
        elif model_type == "ollama":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), api_base="http://localhost:11434/v1")
            self.model_name = "ollama-gpt-3.5-turbo"
    
    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(model=self.model_name,
                                                        messages=messages,
                                                        temperature=0.0)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating completion: {e}")
            return f"Error generating completion: {e}"

class EmbeddingModel:
    def __init__(self, model_type="openai"):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.emb_func = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), 
                                                                        model_name="text-embedding-3-small")
        elif model_type == "chroma":
            self.emb_func = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            self.emb_func = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), 
                                                                        api_base="http://localhost:11434/v1",
                                                                        model_name="nomic-embed-text")
            
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "pdf_collection"
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name, embedding_function=self.emb_func)

    
        
def select_models():
    print("Select LLM model:")
    print("1. OpenAI (remote)")
    print("2. Ollama (local)")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = {"1": "openai", "2": "ollama"}[choice]
            break
    while True:
        print("Select embedding model:")
        print("1. OpenAI (remote)")
        print("2. Chroma (local)")
        print("3. Nomic (local)")
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            embedding_type = {"1": "openai", "2": "chroma", "3": "nomic"}[choice]
            break
    return llm_type, embedding_type

def generate_csv():
    data = [
        {"id": 1, "fact": "The universe is about 13.8 billion years old."},
        {"id": 2, "fact": "Light from the Sun takes about 8 minutes to reach Earth."},
        {"id": 3, "fact": "There are likely more than 2 trillion galaxies in the observable universe."},
        {"id": 4, "fact": "A single galaxy can contain billions or even trillions of stars."},
        {"id": 5, "fact": "Black holes have gravity so strong that not even light can escape them."},
        {"id": 6, "fact": "The observable universe is about 93 billion light-years in diameter."},
        {"id": 7, "fact": "Neutron stars are so dense that a teaspoon of their material would weigh billions of tons on Earth."},
        {"id": 8, "fact": "Dark matter makes up about 27% of the universe."},
        {"id": 9, "fact": "Dark energy makes up about 68% of the universe and is driving its accelerated expansion."},
        {"id": 10, "fact": "The Milky Way galaxy is estimated to contain between 100 and 400 billion stars."}
    ]
    with open("space_facts.csv", mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "fact"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)
            
    print("CSV file 'space_facts.csv' generated successfully.")
        
        
def load_csv():
    df = pd.read_csv("space_facts.csv")
    documents = df["fact"].tolist()
    return documents
    print(f"Loaded {len(documents)} documents from CSV.")
    
    
def setup_chromadb(documents, embedding_model):
    client = chromadb.Client()
    print("Creating ChromaDB collection ", documents)
    
    try:
        client.delete_collection("safe_facts")
    except:
        pass
    
    collection = client.get_or_create_collection(name="safe_facts", embedding_function=embedding_model.emb_func)
   
    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])
    print(f"Added {len(documents)} documents to ChromaDB collection 'safe_facts'.")
    return collection
    
def find_related_chunks(query, collection, n_results=2):
        result = collection.query(query_texts=[query], n_results=n_results)
        
        print("\nRelated Chunks:")
        for doc in result["documents"][0]:
            print(f" - {doc}")
            
        return list(
            zip(
                result["documents"][0],
                (
                    result["metadatas"][0]
                    if result["metadatas"][0] 
                    else [{}] * len(result["documents"][0])
                )
            )
        )
        
def augment_prompt(question, related_chunks):
    context = "\n".join([chunk for chunk, _ in related_chunks])
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    print("\nAugmented Prompt:")
    return augmented_prompt
    
def rag_pipeline(question, collection, llm_model, n_results=2):
    related_chunks = find_related_chunks(question, collection, n_results)
    augmented_prompt = augment_prompt(question, related_chunks)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
        {"role": "user", "content": augmented_prompt}
    ]
    
    answer = llm_model.generate_completion(messages)
    references = [chunk[0] for chunk in related_chunks]
    print("\nGenerated Answer:")
    return answer, references

def main():
    llm_type, embedding_type = select_models()
    llm_model = LLMModel(model_type=llm_type)
    embedding_model = EmbeddingModel(model_type=embedding_type)
    
    print(f"Selected LLM Model: {llm_type}")
    print(f"Selected Embedding Model: {embedding_type}")
    
    generate_csv()
    documents = load_csv()
    
    print("\nSetting up ChromaDB collection with documents...", documents)
    
    collection = setup_chromadb(documents, embedding_model)
    
    queries = [
        "How old is the universe?",
        "What is dark matter?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        answer, references = rag_pipeline(query, collection, llm_model)
        print(f"Answer: {answer}")
        print(f"References: {references}")
    
    
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
