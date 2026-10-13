# app/llm.py
import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")


def get_llm():
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=MODEL, temperature=0)
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=MODEL, temperature=0)


llm = get_llm()