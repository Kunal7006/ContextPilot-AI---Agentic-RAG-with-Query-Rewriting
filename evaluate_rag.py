"""Evaluate the existing Qdrant + Agno/Gemini RAG pipeline in LangSmith.

Run: python evaluate_rag.py
Uses the already-created LangSmith dataset and the existing Qdrant collection.
"""
import os
import re
import time
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from agno.agent import Agent
from agno.models.google import Gemini
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langsmith import Client, evaluate, traceable

load_dotenv()

DATASET_NAME = "Agentic-RAG-Thesis-Evaluation"
COLLECTION_NAME = "gemini-agentic-rag-v2"
MODEL_NAME = "gemini-3.5-flash-lite"  # Match your working Streamlit app
USE_QUERY_REWRITING = False  # First run: baseline without rewriting
K = 5

for key in ("GOOGLE_API_KEY", "QDRANT_API_KEY", "QDRANT_URL", "LANGSMITH_API_KEY"):
    if not os.getenv(key):
        raise RuntimeError(f"Missing {key} in .env")

os.environ.setdefault("GEMINI_API_KEY", os.environ["GOOGLE_API_KEY"])
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "Gemini-Agentic-RAG")


class GeminiEmbedder(Embeddings):
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = "models/gemini-embedding-001"

    def _embed(self, text: str, task_type: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=768
            ),
        )
        return result.embeddings[0].values

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, "RETRIEVAL_QUERY")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text, "RETRIEVAL_DOCUMENT") for text in texts]


qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=60,
)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=GeminiEmbedder(),
)


@traceable(name="Query Rewriting", run_type="chain")
def rewrite_question(question: str) -> str:
    agent = Agent(
        name="Query Rewriter",
        model=Gemini(id=MODEL_NAME),
        instructions=(
            "Rewrite the question to be search-friendly without changing its meaning. "
            "Return only the rewritten question."
        ),
    )
    return str(agent.run(question).content)


@traceable(name="Document Retrieval", run_type="retriever")
def retrieve_chunks(query: str) -> list[dict]:
    # Match the currently working debug configuration: similarity, no threshold.
    docs = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": K}
    ).invoke(query)
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


@traceable(name="Answer Generation", run_type="chain")
def generate_answer(question: str, rewritten_question: str, context: str) -> str:
    agent = Agent(
        name="Gemini RAG Agent",
        model=Gemini(id=MODEL_NAME),
        instructions=(
            "Answer accurately using the provided document context. "
            "If the context does not support the answer, say so."
        ),
    )
    full_prompt = (
        f"Context: {context}\n\n"
        f"Original Question: {question}\n"
        f"Rewritten Question: {rewritten_question}\n\n"
        "Please provide a comprehensive answer based on the available information."
    )
    return str(agent.run(full_prompt).content)


@traceable(name="End-to-End Thesis RAG", run_type="chain")
def rag_target(inputs: dict) -> dict:
    started = time.perf_counter()
    question = inputs["question"]
    rewritten = rewrite_question(question) if USE_QUERY_REWRITING else question
    chunks = retrieve_chunks(rewritten)
    context = "\n\n".join(item["content"] for item in chunks)
    answer = generate_answer(question, rewritten, context)
    return {
        "answer": answer,
        "retrieved_context": context,
        "retrieved_chunks": chunks,
        "rewritten_question": rewritten,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def normalize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def token_f1(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Lexical overlap only; NOT semantic answer correctness."""
    from collections import Counter
    predicted = Counter(normalize(outputs.get("answer", "")))
    expected = Counter(normalize(reference_outputs.get("answer", "")))
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"key": "reference_token_f1", "score": round(f1, 4)}


def latency_metric(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "latency_seconds", "score": outputs["latency_seconds"]}


if __name__ == "__main__":
    client = Client()
    if not client.has_dataset(dataset_name=DATASET_NAME):
        raise RuntimeError(f"LangSmith dataset not found: {DATASET_NAME}")
    print(f"Running dataset: {DATASET_NAME}")
    print(f"Query rewriting enabled: {USE_QUERY_REWRITING}")
    print("Note: each example makes Gemini API calls; monitor quota and cost.")
    result = evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[token_f1, latency_metric],
        experiment_prefix="thesis-rag-rewrite" if USE_QUERY_REWRITING else "thesis-rag-baseline",
        max_concurrency=1,
        client=client,
    )
    print("Evaluation finished. Open LangSmith > Datasets & Experiments to see results.")
