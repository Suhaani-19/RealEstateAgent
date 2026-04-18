from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

_db = None

def get_db():
    global _db

    if _db is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        try:
            with open("data/market_data.txt", "r") as f:
                text = f.read()
        except:
            text = "No market data available."

        docs = [Document(page_content=text)]

        _db = FAISS.from_documents(docs, embeddings)

    return _db


def retrieve_context(query: str, k: int = 4) -> str:
    try:
        db = get_db()
        results = db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Market data unavailable.\n{str(e)}"