import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

_db = None

def get_db():
    global _db

    if _db is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # if DB doesn't exist → create it
        if not os.path.exists("./chroma_db"):
            os.makedirs("./chroma_db", exist_ok=True)

            # load market data
            with open("data/market_data.txt", "r") as f:
                text = f.read()

            docs = [Document(page_content=text)]

            _db = Chroma.from_documents(
                docs,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            _db.persist()

        else:
            _db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )

    return _db


def retrieve_context(query: str, k: int = 4) -> str:
    try:
        db = get_db()
        results = db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Market data unavailable.\n{str(e)}"