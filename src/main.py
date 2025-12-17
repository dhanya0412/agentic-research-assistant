from groq import Groq
from config import GROQ_API_KEY
import logging

logging.basicConfig(
    filename="logs/agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

client = Groq(api_key=GROQ_API_KEY)

def test_llm_connection():
    logging.info("Testing Groq LLM connectivity")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        temperature=0
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    test_llm_connection()
