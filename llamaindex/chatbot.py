from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-4o-mini", temperature=0.1)


def chat_init():
    chat_history = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    ]
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break
        chat_history.append(ChatMessage(role=MessageRole.USER, content=user_input))
        response = llm.chat(messages = chat_history)
        answer = response.message.content
        print(f"Assistant: {answer}")
        
        chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=answer))

chat_init()
    

