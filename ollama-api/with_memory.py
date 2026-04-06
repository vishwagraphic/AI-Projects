from openai import OpenAI
import sys
from dotenv import load_dotenv
from typing import List, Dict


def intiailize_client(use_ollama:bool = False) -> OpenAI:
    if use_ollama:
        client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
    else:
        client = OpenAI()      
    return client


def create_initial_message() -> list[dict[str, str]]:   
    return [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
def chat(user_input: str, messages: list[dict[str, str]], client: OpenAI, model_name) -> str:
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(model=model_name,
                                    messages=messages + [{"role": "user", "content": user_input}]
                                    )
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        print(f"\nAssistant: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        
def summarize_messages(messages: list[dict[str, str]])-> list[dict[str, str]]:
    # Implement a summarization strategy here (e.g., using a separate model or heuristic)
    summary = "Previous message summary: " + " ".join([m['content'][:50] for m in messages[-5:]])
    return [{"role": "system", "content": summary}] + messages[-5:]


def save_conversation(messages: list[dict[str, str]], fileName: str = "conversation.json"):
    with open(fileName, 'w') as f:
        json.dump(messages, f)
        
def load_conversation(fileName: str) -> list[dict[str, str]]:
    try:
        with open(fileName, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return create_initial_message()
    
def main():
    client = intiailize_client(use_ollama=True)
    model_name = "llama3.2" if client.base_url else "gpt-3.5-turbo"
    
    messages = create_initial_message()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        
        if user_input.lower() in ["exit", "quit"]:
            break
        elif user_input.lower() == "save":
            save_conversation(messages)
            continue
        elif user_input.lower() == "load":
            messages = load_conversation()
            continue
        response = chat(user_input, messages, client, model_name)
        
        if len(messages) > 10:
            messages = summarize_messages(messages)
            
    
if __name__ == "__main__":
    main()
    