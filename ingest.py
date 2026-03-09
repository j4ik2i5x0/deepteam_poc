import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env in the project root.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
POLICY_FILE = BASE_DIR / "data" / "Policy.pdf"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"


def _validate_startup() -> None:
    """Validate that the required environment variables and files are present."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or your environment."
        )
    if not POLICY_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {POLICY_FILE}. Create the file before running."
        )


def main():
    """Load the policy document, split it into chunks, and store it in ChromaDB."""
    _validate_startup()

    print("Loading document...")
    loader = PyPDFLoader(str(POLICY_FILE))
    documents = loader.load()

    print("Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    print("Creating embeddings and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=str(CHROMA_DB_DIR)
    )

    print(f"Successfully ingested {len(docs)} chunks into {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
