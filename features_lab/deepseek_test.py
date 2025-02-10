import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

response = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Llama-70B",
    messages=[{"role":"system","content":"You are a deep thinking assistant"},{"role":"user","content":"what is 1+1?"}],
    temperature=0.0,
    top_p=0.5,
    max_tokens=10000
)

print(response.choices[0].message.content)