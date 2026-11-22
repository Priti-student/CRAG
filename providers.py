"""Swappable backends for CRAG. Default stack is 100% free and runs locally:

    LLM        -> Ollama (local)            CRAG_LLM=ollama
    Embeddings -> sentence-transformers     CRAG_EMBED=hf
    Evaluator  -> small cross-encoder       CRAG_EVALUATOR=crossencoder
    Web search -> DuckDuckGo (no API key)   CRAG_SEARCH=ddg

Other options: CRAG_LLM=groq|openai, CRAG_EMBED=ollama|openai,
CRAG_EVALUATOR=llm, CRAG_SEARCH=tavily.
"""
import os
from typing import Callable, List

from crag_logic import parse_score, sigmoid


def get_llm():
    kind = os.getenv("CRAG_LLM", "ollama").lower()
    model = os.getenv("CRAG_MODEL")
    if kind == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "llama3.1:8b", temperature=0)
    if kind == "groq":
        from langchain_groq import ChatGroq
        if not model:
            raise ValueError("Set CRAG_MODEL to a model name from Groq's current model list.")
        return ChatGroq(model=model, temperature=0)
    if kind == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
    raise ValueError(f"Unknown CRAG_LLM={kind}")


def get_embeddings():
    kind = os.getenv("CRAG_EMBED", "hf").lower()
    if kind == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")
    if kind == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    raise ValueError(f"Unknown CRAG_EMBED={kind}")


def get_scorer(llm) -> Callable[[str, List[str]], List[float]]:
    """Return score(question, passages) -> list of relevance scores in [0, 1]."""
    kind = os.getenv("CRAG_EVALUATOR", "crossencoder").lower()

    if kind == "crossencoder":
        # A small dedicated relevance model: closest free analogue of the paper's
        # lightweight T5 evaluator, and it needs zero LLM calls.
        import torch
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(os.getenv("CRAG_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

        def raw_scores(pairs):
            # Ask for raw logits whichever keyword this sentence-transformers version uses.
            for kw in ("activation_fn", "activation_fct"):
                try:
                    return model.predict(pairs, **{kw: torch.nn.Identity()})
                except TypeError:
                    continue
            return model.predict(pairs)

        def score(question: str, passages: List[str]) -> List[float]:
            if not passages:
                return []
            logits = raw_scores([(question, p) for p in passages])
            return [sigmoid(float(x)) for x in logits]

        return score

    if kind == "llm":
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "Rate how useful the passage is for answering the question: "
                           "1.0 = directly answers it, 0.5 = partly relevant, 0.0 = irrelevant. "
                           "Reply with ONLY a number between 0 and 1."),
                ("human", "Question: {question}\n\nPassage:\n{passage}"),
            ]) | llm | StrOutputParser()
        )

        def score(question: str, passages: List[str]) -> List[float]:
            if not passages:
                return []
            outs = chain.batch([{"question": question, "passage": p} for p in passages])
            return [parse_score(o) for o in outs]

        return score

    raise ValueError(f"Unknown CRAG_EVALUATOR={kind}")


def get_search() -> Callable[[str], List[str]]:
    """Return search(query) -> list of text snippets/pages."""
    kind = os.getenv("CRAG_SEARCH", "ddg").lower()

    if kind == "ddg":
        from ddgs import DDGS

        def search(query: str) -> List[str]:
            hits = DDGS().text(query, max_results=5)
            return [f"{h.get('title', '')}. {h.get('body', '')}".strip() for h in hits]

        return search

    if kind == "tavily":
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        def search(query: str) -> List[str]:
            res = client.search(query=query, max_results=3)
            return [r["content"] for r in res.get("results", []) if r.get("content")]

        return search

    raise ValueError(f"Unknown CRAG_SEARCH={kind}")
