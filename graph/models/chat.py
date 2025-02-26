# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# llm = ChatOpenAI(temperature=0)
llm = ChatOllama(model="deepseek-r1:70b", temperature=0)

