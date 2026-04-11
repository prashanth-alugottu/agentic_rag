
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = "ft:gpt-3.5-turbo-0125:personal::DP77y73h"

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "What is password policy?"}
    ]
)

print(response.choices[0].message.content)