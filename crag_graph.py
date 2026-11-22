"""Corrective RAG (CRAG) as a LangGraph state machine (free, local-first stack by default).

Flow (matches the paper's inference algorithm):

    correct   : retrieve -> evaluate -> refine -> generate
    ambiguous : retrieve -> evaluate -> refine -> rewrite_query -> web_search -> generate
    incorrect : retrieve -> evaluate -> rewrite_query -> web_search -> generate

Differences from the paper:
  * The paper fine-tunes a T5-large retrieval evaluator. Here the evaluator is a small
    off-the-shelf cross-encoder (or an LLM scorer). Swap `score_fn` for your own model.
  * Thresholds are tunable defaults, NOT the paper's values. Different evaluators
    produce differently-scaled scores, so tune them (the demo prints the scores).
"""
import os
from typing import Callable, List, TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from crag_logic import decide_confidence, filter_strips, recompose, split_into_strips
from providers import get_embeddings, get_llm, get_scorer, get_search

load_dotenv()

UPPER_THRESHOLD = float(os.getenv("CRAG_UPPER", 0.6))   # any doc above this -> correct
LOWER_THRESHOLD = float(os.getenv("CRAG_LOWER", 0.3))   # all docs below this -> incorrect
STRIP_THRESHOLD = float(os.getenv("CRAG_STRIP", 0.5))   # keep strips at/above this


class GraphState(TypedDict, total=False):
    question: str
    documents: List[Document]
    scores: List[float]
    confidence: str            # "correct" | "incorrect" | "ambiguous"
    internal_knowledge: str    # refined knowledge from retrieved docs
    search_query: str          # rewritten query (<= 3 keywords)
    external_knowledge: str    # refined knowledge from web search
    generation: str


def build_crag_app(
    retriever,
    llm,
    score_fn: Callable[[str, List[str]], List[float]],
    search_fn: Callable[[str], List[str]],
):
    def refine_text(question: str, texts: List[str]) -> str:
        """Decompose -> score strips -> filter -> recompose."""
        strips = [s for t in texts for s in split_into_strips(t)]
        scores = score_fn(question, strips)
        return recompose(filter_strips(strips, scores, STRIP_THRESHOLD))

    rewrite_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "Extract at most three keywords, separated by commas, from the question to "
             "use as a web search query. Return only the keywords."),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )

    answer_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using ONLY the knowledge below. If it is not enough, "
             "say you don't know instead of guessing."),
            ("human", "Knowledge:\n{knowledge}\n\nQuestion: {question}\nAnswer:"),
        ])
        | llm
        | StrOutputParser()
    )

    # ---- graph nodes ----
    def retrieve(state: GraphState):
        return {"documents": retriever.invoke(state["question"])}

    def evaluate(state: GraphState):
        scores = score_fn(state["question"], [d.page_content for d in state["documents"]])
        return {"scores": scores,
                "confidence": decide_confidence(scores, UPPER_THRESHOLD, LOWER_THRESHOLD)}

    def refine(state: GraphState):
        # Only refine documents the evaluator did not reject outright.
        relevant = [d.page_content for d, s in zip(state["documents"], state["scores"])
                    if s >= LOWER_THRESHOLD]
        return {"internal_knowledge": refine_text(state["question"], relevant)}

    def rewrite_query(state: GraphState):
        return {"search_query": rewrite_chain.invoke({"question": state["question"]}).strip()}

    def web_search(state: GraphState):
        pages = search_fn(state["search_query"])
        return {"external_knowledge": refine_text(state["question"], pages)}

    def generate(state: GraphState):
        parts = [state.get("internal_knowledge", ""), state.get("external_knowledge", "")]
        knowledge = "\n\n".join(p for p in parts if p)
        return {"generation": answer_chain.invoke(
            {"knowledge": knowledge or "(none)", "question": state["question"]})}

    # ---- routing (the CRAG action trigger) ----
    def after_evaluate(state: GraphState):
        return "rewrite_query" if state["confidence"] == "incorrect" else "refine"

    def after_refine(state: GraphState):
        return "rewrite_query" if state["confidence"] == "ambiguous" else "generate"

    g = StateGraph(GraphState)
    for name, fn in [("retrieve", retrieve), ("evaluate", evaluate), ("refine", refine),
                     ("rewrite_query", rewrite_query), ("web_search", web_search),
                     ("generate", generate)]:
        g.add_node(name, fn)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "evaluate")
    g.add_conditional_edges("evaluate", after_evaluate,
                            {"refine": "refine", "rewrite_query": "rewrite_query"})
    g.add_conditional_edges("refine", after_refine,
                            {"rewrite_query": "rewrite_query", "generate": "generate"})
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)
    return g.compile()


if __name__ == "__main__":
    # Tiny demo corpus. Replace with your own documents (PDFs, web pages, ...).
    corpus = [
        Document(page_content=(
            "Corrective RAG (CRAG) adds a lightweight retrieval evaluator in front of the "
            "generator. The evaluator scores each retrieved document. Depending on the "
            "scores, CRAG refines the documents, falls back to web search, or does both.")),
        Document(page_content=(
            "Self-RAG trains the generator to emit reflection tokens that critique its own "
            "output. It requires an instruction-tuned generator. CRAG does not.")),
        Document(page_content=(
            "FAISS is a library for efficient similarity search over dense vectors. "
            "It is commonly used as a vector store in RAG pipelines.")),
    ]
    llm = get_llm()
    retriever = FAISS.from_documents(corpus, get_embeddings()).as_retriever(search_kwargs={"k": 3})
    app = build_crag_app(retriever, llm, get_scorer(llm), get_search())

    print(app.get_graph().draw_mermaid())  # paste into https://mermaid.live to see the graph

    for q in ["What does the CRAG retrieval evaluator do?",      # expected -> correct
              "Who won the most recent FIFA World Cup?"]:        # off-corpus -> incorrect -> web
        out = app.invoke({"question": q})
        print(f"\nQ: {q}\n  confidence: {out['confidence']}  scores: {out['scores']}"
              f"\n  search_query: {out.get('search_query')}\n  answer: {out['generation']}")
