import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import httpx

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

API_ENDPOINT = os.getenv("API_ENDPOINT")
API_KEY = os.getenv("API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

tiktoken_cache_dir = "tiktoken_cache"
os.makedirs(tiktoken_cache_dir, exist_ok=True)

os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

assert os.path.exists(
    os.path.join(
        tiktoken_cache_dir,
        "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
    )
)

http_client = httpx.Client(verify=False)

INPUT_PATH = Path("data/patterns.json")
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "architecture_patterns"


def build_embedding_model():
    """Return embeddings"""
    
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=API_ENDPOINT,
        http_client=http_client
    )
    return embeddings


def load_patterns(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def pattern_to_document(pattern: dict[str, Any]) -> Document:
    name = pattern.get("name", "Unknown Pattern")
    summary = pattern.get("summary", "")
    best_for = pattern.get("best_for", "")
    tradeoffs = pattern.get("tradeoffs", "")
    scalability = pattern.get("scalability", "")
    team_size = pattern.get("team_size", "")
    constraints = pattern.get("constraints", "")
    anti_patterns = pattern.get("anti_patterns", "")

    page_content = "\n".join(
        [
            f"Pattern: {name}",
            f"Summary: {summary}",
            f"Best For: {best_for}",
            f"Tradeoffs: {tradeoffs}",
            f"Scalability: {scalability}",
            f"Team Size: {team_size}",
            f"Constraints: {constraints}",
            f"When Not To Use: {anti_patterns}",
        ]
    )

    metadata = {
        "name": name,
        "scalability": scalability,
        "team_size": team_size,
    }

    return Document(page_content=page_content, metadata=metadata)


def build_vector_store() -> None:
    patterns = load_patterns(INPUT_PATH)
    docs = [pattern_to_document(pattern) for pattern in patterns]

    embeddings = build_embedding_model()

    # rebuild the chroma db every time it runs
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vector_store.delete_collection()

    # Create the vector db with documents
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    
    print(
        f"Indexed {len(docs)} patterns into Chroma collection "
        f"'{COLLECTION_NAME}' at '{PERSIST_DIR}'."
    )


def main() -> None:
    build_vector_store()


if __name__ == "__main__":
    main()
