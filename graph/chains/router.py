from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from graph.models.chat import llm

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class RouteVectorstore(BaseModel):
    """Route a user query to the most relevant vectorstore."""

    is_anime: bool = Field(
        ...,
        description="Given a user question determine whether the question is about anime.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, adversarial attacks or anime. 
Use the vectorstore for questions on these topics. For all else, use web-search. 
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

structured_llm_router2 = llm.with_structured_output(RouteVectorstore)
system2 = """You are an expert at determining if a user question is related to anime or not. Answer in 'yes' or 'no'.
"""
vectorstore_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system2),
        ("human", "{question}"),
    ]
)

vectorstore_router = vectorstore_prompt | structured_llm_router2