import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from prompts import (
    CLARIFYING_QUESTION_PROMPT,
    RANK_AND_RECOMMEND_PROMPT,
    REQUIREMENTS_EXTRACTION_PROMPT,
)

load_dotenv()

API_ENDPOINT = os.getenv("API_ENDPOINT")
API_KEY = os.getenv("API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "architecture_patterns"
RETRIEVER_K = 5
CLARITY_THRESHOLD = 0.55
SPEED_TARGET_SECONDS = 10.0

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


class RecommendationState(TypedDict):
    project_input: str
    parsed_requirements: str
    retrieved_docs: list[Document]
    recommendations_md: str
    confidence: float
    needs_clarification: bool
    clarifying_question: str
    ranked_recommendations: list[dict[str, Any]]
    llm: Any
    retriever: Any


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("Model response did not include a JSON object.")

    return json.loads(match.group(0))


def format_recommendations_markdown(recommendations: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(recommendations[:3], start=1):
        pattern = str(item.get("pattern", "Unknown Pattern")).strip()
        why_it_fits = str(item.get("why_it_fits", "Not provided")).strip()
        tradeoffs = str(item.get("tradeoffs", "Not provided")).strip()
        when_not_to_use = str(item.get("when_not_to_use", "Not provided")).strip()
        confidence = float(item.get("confidence", 0.0))

        blocks.append(
            "\n".join(
                [
                    f"## {index}. Pattern: {pattern}",
                    f"- Why it fits: {why_it_fits}",
                    f"- Tradeoffs: {tradeoffs}",
                    f"- When not to use: {when_not_to_use}",
                    f"- Confidence: {max(0.0, min(1.0, confidence)):.2f}",
                ]
            )
        )

    return "\n\n".join(blocks)


def parse_requirements_node(state: RecommendationState) -> RecommendationState:
    llm = state["llm"]
    prompt = (
        f"{REQUIREMENTS_EXTRACTION_PROMPT}\n\n"
        f"Project Brief:\n{state['project_input']}"
    )
    parsed = llm.invoke(prompt).content.strip()
    return {"parsed_requirements": parsed}


def retrieve_patterns_node(state: RecommendationState) -> RecommendationState:
    retriever = state["retriever"]
    docs = retriever.invoke(state["parsed_requirements"])
    return {"retrieved_docs": docs}


def reason_and_rank_node(state: RecommendationState) -> RecommendationState:
    llm = state["llm"]

    documents = []
    for idx, doc in enumerate(state["retrieved_docs"], start=1):
        documents.append(f"Retrieved Pattern {idx}:\n{doc.page_content}")
    context =  "\n\n".join(documents)

    prompt = (
        f"{RANK_AND_RECOMMEND_PROMPT}\n\n"
        f"Parsed Requirements:\n{state['parsed_requirements']}\n\n"
        f"Retrieved Context:\n{context}"
    )

    response_text = llm.invoke(prompt).content.strip()
    payload = extract_json(response_text)

    confidence = float(payload.get("confidence", 0.0))
    recommendations = payload.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []
    recommendations = recommendations[:3]
    recommendations_md = format_recommendations_markdown(recommendations)

    return {
        "confidence": max(0.0, min(1.0, confidence)),
        "ranked_recommendations": recommendations,
        "recommendations_md": recommendations_md,
    }


def clarify_or_finalize_node(state: RecommendationState) -> RecommendationState:
    confidence = state.get("confidence", 0.0)

    if confidence < CLARITY_THRESHOLD:
        llm = state["llm"]
        prompt = (
            f"{CLARIFYING_QUESTION_PROMPT}\n\n"
            f"Project Brief:\n{state['project_input']}\n\n"
            f"Parsed Requirements:\n{state['parsed_requirements']}"
        )
        question = llm.invoke(prompt).content.strip()
        return {
            "needs_clarification": True,
            "clarifying_question": question,
        }

    return {"needs_clarification": False}


def build_graph():
    graph_builder = StateGraph(RecommendationState)
    graph_builder.add_node("parse_requirements", parse_requirements_node)
    graph_builder.add_node("retrieve_patterns", retrieve_patterns_node)
    graph_builder.add_node("reason_and_rank", reason_and_rank_node)
    graph_builder.add_node("clarify_or_finalize", clarify_or_finalize_node)

    graph_builder.add_edge(START, "parse_requirements")
    graph_builder.add_edge("parse_requirements", "retrieve_patterns")
    graph_builder.add_edge("retrieve_patterns", "reason_and_rank")
    graph_builder.add_edge("reason_and_rank", "clarify_or_finalize")
    graph_builder.add_edge("clarify_or_finalize", END)

    return graph_builder.compile()


def run_recommendation(project_description: str) -> dict[str, Any]:
    started = time.perf_counter()

    # Create llm for OpenAI
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=API_KEY,
        base_url=API_ENDPOINT,
        http_client=http_client,
    )

    # Build retriever to get the similar design patterns from chroma db
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=API_ENDPOINT,
        http_client=http_client,
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    graph = build_graph()

    final_state = graph.invoke(
        {
            "project_input": project_description,
            "llm": llm,
            "retriever": retriever,
        }
    )

    confidence = float(final_state.get("confidence", 0.0))
    elapsed_seconds = round(time.perf_counter() - started, 2)
    met_speed_target = elapsed_seconds <= SPEED_TARGET_SECONDS

    if final_state.get("needs_clarification"):
        return {
            "status": "needs_clarification",
            "question": final_state.get("clarifying_question", "Can you share more details?"),
            "confidence": confidence,
            "elapsed_seconds": elapsed_seconds,
            "met_speed_target": met_speed_target,
        }

    return {
        "status": "ok",
        "recommendations_markdown": final_state.get("recommendations_md", ""),
        "confidence": confidence,
        "elapsed_seconds": elapsed_seconds,
        "met_speed_target": met_speed_target,
    }

def print_result(result: dict[str, Any]) -> None:
    print(f"Status: {result['status']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(
        f"Response Time: {result['elapsed_seconds']:.2f}s "
        f"(<=10s target: {'yes' if result['met_speed_target'] else 'no'})"
    )

    if result["status"] == "needs_clarification":
        print(f"Question: {result['question']}")
        return

    print()
    print(result["recommendations_markdown"])


def main() -> None:
    user_input = """
We are building an e-commerce platform expected to handle 2 million monthly users, 
with frequent catalog updates, flash sales, and separate teams for checkout, inventory, 
and recommendations. We need high scalability, fault isolation, and fast feature releases.
"""

    result = run_recommendation(user_input)
    print_result(result)


if __name__ == "__main__":
    main()
