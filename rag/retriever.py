from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
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

            # ✅ If empty → insert default data
            if _db._collection.count() == 0:
                docs = [Document(page_content="""
                Real estate markets generally show steady growth.
                Waterfront properties have higher premiums.
                High-demand areas see faster appreciation.
                Rental yields typically range between 4-7%.
                Newer homes often have better resale value.
                """)]

                _db.add_documents(docs)
                _db.persist()

        except Exception:
            # 💥 Reset corrupted DB
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")

            docs = [Document(page_content="Real estate market trends vary by location and demand.")]
            _db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
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