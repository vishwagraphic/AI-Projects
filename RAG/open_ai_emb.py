from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

open_ai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=open_ai_key)

embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=[
        "The cat is on the table.",
        "The dog is in the garden."
    ]
)

print(embedding.data[0].embedding)