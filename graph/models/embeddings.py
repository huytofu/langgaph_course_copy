# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# embeddings = OpenAIEmbeddings()
embeddings = OllamaEmbeddings(model="llama3.1")