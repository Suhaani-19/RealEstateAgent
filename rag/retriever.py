from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
import shutil

_db = None

def get_db():
    global _db

    # 💥 FORCE DELETE BAD DB ON START (important for Streamlit Cloud)
    if os.path.exists("./chroma_db"):
        try:
            # try opening → if fails, delete
            test_db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            )
            test_db._collection.count()
        except:
            shutil.rmtree("./chroma_db")

    if _db is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        _db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

        # insert default data if empty
        if _db._collection.count() == 0:
            from langchain.schema import Document

            docs = [Document(page_content="Real estate markets show steady growth with location-based demand.")]
            _db.add_documents(docs)
            _db.persist()

    return _db


def retrieve_context(query: str, k: int = 4) -> str:
    db = get_db()

    try:
        results = db.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])
        return context if context.strip() else "No market data available."
    except Exception as e:
        return f"Error retrieving market data: {str(e)}"