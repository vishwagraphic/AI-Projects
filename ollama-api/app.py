import requests
import json
import ollama

url = "http://localhost:11434/api/generate"

data = {
        "model": "llama3.2",
        "prompt": "What is the meaning of life?",
        "max_tokens": 100,
}    
def stream_response(url, data):
   
    response = requests.post(url, json=data, stream=True)
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                result = json.loads(decoded_line)
                generated_text = result.get('response', '')
                print(generated_text, end='', flush=True)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        
#stream_response(url, data)

# print(ollama.list())

# res = ollama.chat(model="llama3.2", 
#                   messages=[{"role": "user", "content": "What is the meaning of life?"}], stream=True)

# for chunk in res:
#     print(chunk["message"]["content"], end='', flush=True)
    
    
# test = ollama.generate(model="llama3.2", 
#                   prompt="What is the meaning of life?")

# print(ollama.show("llama3.2"))

ollama.create(
    model="my-smart-assistant",
    from_="llama3.2",
    system="you are very smart assistant who can answer any question in a concise manner",
    parameters={"temperature": 0.1}
)
res = ollama.generate(model="my-smart-assistant", prompt="What is the meaning of life?")
print(res["response"])