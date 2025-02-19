from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from graph.models.embeddings import embeddings


retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()

retriever_anime = PineconeVectorStore(
    index_name='firecrawl-index', 
    embedding=embeddings
).as_retriever()
