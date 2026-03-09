import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env in the project root.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "0").lower() in {
    "1",
    "true",
    "yes",
}

if ENABLE_GUARDRAILS:
    from guardrails import check_input, check_output


def _validate_startup() -> None:
    """Validate that required environment variables and vector DB are present."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or your environment."
        )
    if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
        raise FileNotFoundError(
            f"ChromaDB directory not found or empty: {CHROMA_DB_DIR}. "
            "Run ingest.py first."
        )


def create_qa_chain() -> RetrievalQA:
    """Create RetrievalQA chain from existing ChromaDB (no ingestion here)."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def ask_rag(qa_chain: RetrievalQA, question: str):
    """Validate question/output with guardrails and return QA result."""
    if ENABLE_GUARDRAILS:
        is_safe, message = check_input(question)
        if not is_safe:
            return message

    result = qa_chain.invoke(question)
    output_text = result.get("result", "") if isinstance(result, dict) else str(result)

    if ENABLE_GUARDRAILS:
        is_safe, message = check_output(question, output_text)
        if not is_safe:
            return message

    return result


if __name__ == "__main__":
    _validate_startup()
    qa = create_qa_chain()
    while True:
        q = input("Ask: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        print(ask_rag(qa, q))