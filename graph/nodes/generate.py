from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    context = "\n\n".join([doc.page_content for doc in documents])

    generation = generation_chain.invoke({"context": context, "question": question})
    retry_count += 1
    return {"documents": documents, "question": question, "generation": generation, "retry_count": retry_count}
