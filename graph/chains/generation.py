from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from graph.models.chat import llm

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
