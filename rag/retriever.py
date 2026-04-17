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

        # ✅ FIX: auto-load data if DB is empty
        try:
            if _db._collection.count() == 0:
                from rag.ingest import load_data
                load_data()

                # reload DB after inserting
                _db = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=embeddings
                )
        except:
            pass

    return _db


def retrieve_context(query: str, k: int = 4) -> str:
    db = get_db()

    try:
        results = db.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])

        # ✅ fallback if nothing found
        if not context.strip():
            return "No market data available."

        return context

    except Exception as e:
        return f"Error retrieving market data: {str(e)}"