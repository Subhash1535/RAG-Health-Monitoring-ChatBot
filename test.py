# test_groq_available_models.py
# Run this to print ONLY the working/available Groq models with your API key
# No error messages, no failed models — just clean list of valid models

from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("No GROQ_API_KEY found")
    exit()

# Current active Groq models (Dec 2025)
MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
]

print("Available Groq models with your API key:\n")

available = []

for model in MODELS:
    try:
        llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0, max_tokens=1)
        llm.invoke("OK")  # Tiny request to test
        available.append(model)
        print(model)
    except:
        pass  # Silently skip unavailable/failed

if not available:
    print("No models available (check your API key)")