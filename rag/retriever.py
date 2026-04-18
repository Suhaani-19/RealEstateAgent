from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

_db = None


def get_db():
    global _db

    if _db is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        with open("data/market_data.txt", "r") as f:
            text = f.read()

        # 🔥 IMPORTANT: chunk the text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        docs = splitter.create_documents([text])

        _db = FAISS.from_documents(docs, embeddings)

    return _db


def retrieve_context(query: str, city: str, k: int = 4) -> str:
    try:
        if city.lower() == "seattle":
            db = get_db()
            results = db.similarity_search(f"{city} real estate {query}", k=k)
            return "\n\n".join([doc.page_content for doc in results])

        return f"""
{city} real estate market is growing steadily.
Average home prices vary depending on locality and demand.
Developing areas may offer better investment opportunities.
Rental yields typically range between 4-7%.
"""

    except Exception as e:
        return f"Market data unavailable.\n{str(e)}"