from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_knowledge_base():
    print("Loading documents...")
    loader = DirectoryLoader(
        "knowledge_base/",
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Creating embeddings (this takes 1-2 minutes first time)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )
    print("Knowledge base ready!")
    return db

if __name__ == "__main__":
    build_knowledge_base()