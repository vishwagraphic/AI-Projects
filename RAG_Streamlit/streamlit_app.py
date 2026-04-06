import streamlit as st
import os
import csv
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb 
from chromadb.utils import embedding_functions
import pprint

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
            response = self.client.chat.completions.create(model=self.model_name.model_name,
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
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.emb_func = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), 
                                                                        model_name="text-embedding-3-small")
            
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
    return data
    
        
    
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
    context = "\n".join([chunk[0] for chunk in related_chunks])
    
    print("\nContext to be added to prompt:", context)
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
    return answer, references, augmented_prompt

    
def streamlit_app():
    st.set_page_config(page_title="Space facts Example", layout="wide")
    st.title("RAG PDF Example with Streamlit")
    
    st.sidebar.title("Model Configuration")
    
    llm_type = st.sidebar.radio("Select LLM model:", 
                                ["openai", "ollama"], 
                                format_func=lambda x: 'OpenAI GPT-4' if 'openai' in x else 'Ollama')
    
    embedding_type = st.sidebar.selectbox("Select embedding model:", 
                                          ["OpenAI (remote)", "Chroma (local)", "Nomic (local)"],
                                          format_func=lambda x: {
                                            "OpenAI (remote)": "OpenAI",
                                            "Chroma (local)": "Chroma",
                                            "Nomic (local)": "Nomic"
                                          }[x])
    
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = False
        st.session_state.facts = generate_csv()
        
        
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(documents, st.session_state.embedding_model)
        pprint.pprint(st.session_state.llm_model.__dict__)
        st.session_state['initialized'] = True
        
    if st.session_state.llm_model.model_name != llm_type or st.session_state.embedding_model.model_name != embedding_type:
        st.session_state.llm_model.model_name = LLMModel(llm_type)
        st.session_state.embeddin_model = EmbeddingModel(embedding_type)
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(documents, st.session_state.embedding_model)
           
    with st.expander("Current Collection Documents", expanded=False):
        for doc in st.session_state.collection.get(include=["documents"])["documents"]:
            st.write(f" - {doc}")
    query = st.text_input("Enter your question about space facts:")
    print(query)
    
    if query:
        with st.spinner("Generating answer..."):
            answer, references, augmented_prompt = rag_pipeline(query, st.session_state.collection, st.session_state.llm_model)
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Generated Answer")
            st.write(answer)
            
        with col2:
            st.markdown("References Used")
            for ref in references:
                st.write(f" - {ref}")
                
        with st.expander("Augmented Prompt", expanded=False):
            st.markdown("Augmented Prompt Sent to LLM")
            st.write(augment_prompt)
            
            st.markdown("LLM configuration")
            st.write(f"LLM Model: {st.session_state.llm_model.model_name}")
            st.write(f"Embedding Model: {st.session_state.embedding_model.emb_func.model_name}")
                
if __name__ == "__main__":
    streamlit_app()