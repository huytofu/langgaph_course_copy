from typing import Any, Dict

from graph.state import GraphState
from graph.retrievers import retriever, retriever_anime


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def retrieve_anime(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE ANIME---")
    question = state["question"]

    documents = retriever_anime.invoke(question)
    return {"documents": documents, "question": question}
