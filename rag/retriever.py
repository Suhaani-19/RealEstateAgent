from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

_db = None

def get_db():
    global _db

    if _db is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 🔥 create documents safely
        try:
            with open("data/market_data.txt", "r") as f:
                text = f.read()
        except:
            text = "No market data available."

        docs = [Document(page_content=text)]

        # 🔥 IMPORTANT: explicit collection name
        _db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="market_data",   # ✅ THIS FIXES TENANT ISSUE
            persist_directory="./chroma_db"
        )

        _db.persist()  # ensure it's saved

    return _db


def retrieve_context(query: str, k: int = 4) -> str:
    try:
        db = get_db()
        results = db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Market data unavailable.\n{str(e)}"