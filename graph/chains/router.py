from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
    is_anime: bool = Field(
        ...,
        description="Whether the question is about anime.",
    )


# llm = ChatOpenAI(temperature=0)
llm = ChatOllama(model="llama3.1:70b", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
There are two vectorstores. One vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
The other vectorstore contains documents related to anime. 
Use the vectorstore for questions on these topics. For all else, use web-search.
You should also conlude if the question is related to anime or not. 
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
