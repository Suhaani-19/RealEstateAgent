from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

_db = None

def get_db():
    global _db

    if _db is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        try:
            _db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )

            # If empty → load data
            if _db._collection.count() == 0:
                from rag.ingest import load_data
                load_data()

                _db = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=embeddings
                )

        except Exception:
            # 💥 HARD RESET if DB is corrupted
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")

            from rag.ingest import load_data
            load_data()

            _db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )

    return _db


def retrieve_context(query: str, k: int = 4) -> str:
    db = get_db()

    try:
        results = db.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])

        return context if context.strip() else "No market data available."

    except Exception as e:
        return f"Error retrieving market data: {str(e)}"