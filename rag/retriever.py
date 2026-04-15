from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

_db = None

def get_db():
    global _db
    if _db is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    return _db

def retrieve_context(query: str, k: int = 4) -> str:
    db = get_db()
    results = db.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context