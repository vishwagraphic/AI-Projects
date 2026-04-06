from openai import OpenAI
import sys


def simple_chat_without_memory(user_input: str, use_ollama: bool = False) -> str:
    
    if use_ollama:
        client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
        model_name = "llama3.2"  # Change to your Ollama model name
    else:
        client = OpenAI()
        model_name = "gpt-3.5-turbo"
    try:
        response = client.chat.completions.create(model=model_name,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": user_input}
                                    ]
                                    )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        
def main():
    while True:
        print("Select model type:")
        print("1. OenAI (remote)")
        print("2. ollama (local)")
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        
    use_ollama = (choice == '2')
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            sys.exit(0)

        if user_input.lower() == "clear":
            continue
        
        if not user_input:
            print("Please enter a valid message.")
            continue
        
        response = simple_chat_without_memory(user_input, use_ollama=True)
        print(response)
        
        # print("\n" + "-" + 50)
        
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)