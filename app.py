
import os
import re
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, TypedDict

import lancedb
import chainlit as cl
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

DB_PATH = "./lancedbv2"
TABLE_NAME = "legal_documents"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "gpt-4o-mini"


# ============================================================
# STATE DEFINITION
# ============================================================

class AgentState(TypedDict):
    original_query: str
    rewritten_query: str
    intent: str
    intent_response: str
    metadata_filter: Dict[str, Any]
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    final_answer: str
    retry_count: int
    node_timings: Dict[str, float]


# ============================================================
# SHARED RESOURCES
# ============================================================

embedder_model = SentenceTransformer(EMBEDDING_MODEL)
main_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, max_tokens=2048)
scorer_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0, max_tokens=512)


def encode_query(text: str) -> List[float]:
    return embedder_model.encode(
        f"query: {text}", convert_to_numpy=True
    ).tolist()


def clean_llm_json(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
    return cleaned.strip()


# ============================================================
# AGENT NODES
# ============================================================

def rewrite_query_node(state):
    t0 = time.time()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at rewriting legal questions in Persian. "
         "Rewrite the question to be clearer and better suited for retrieval. "
         "Return only the rewritten question."),
        ("human", "Original question: {query}")
    ])
    chain = prompt | main_llm | StrOutputParser()
    rewritten = chain.invoke({"query": state["original_query"]})
    state["rewritten_query"] = rewritten.strip()
    state["node_timings"]["rewrite_query"] = time.time() - t0
    return state


def classify_intent_node(state):
    t0 = time.time()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify user input into: greeting, abusive, or law_question. "
         "Return JSON: {\"intent\": \"...\", \"response\": \"...\"}"),
        ("human", "User input: {query}")
    ])
    chain = prompt | main_llm | StrOutputParser()
    raw = chain.invoke({"query": state["original_query"]})
    try:
        parsed = json.loads(clean_llm_json(raw))
        state["intent"] = parsed.get("intent", "law_question")
        state["intent_response"] = parsed.get("response", "")
    except:
        state["intent"] = "law_question"
        state["intent_response"] = ""
    state["node_timings"]["classify_intent"] = time.time() - t0
    return state


def extract_metadata_node(state):
    t0 = time.time()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract legal metadata. Return JSON with legal_domain, "
         "section_number, keywords."),
        ("human", "Query: {query}")
    ])
    chain = prompt | main_llm | StrOutputParser()
    raw = chain.invoke({"query": state["rewritten_query"]})
    try:
        state["metadata_filter"] = json.loads(clean_llm_json(raw))
    except:
        state["metadata_filter"] = {}
    state["node_timings"]["extract_metadata"] = time.time() - t0
    return state


def context_retrieve_node(state):
    t0 = time.time()
    db_conn = lancedb.connect(DB_PATH)
    tbl = db_conn.open_table(TABLE_NAME)
    qvec = encode_query(state["rewritten_query"])
    search = tbl.search(qvec)
    meta = state.get("metadata_filter", {})
    domain = meta.get("legal_domain")
    if domain and domain != "null":
        try:
            search = search.where(f"legal_domain = '{domain}'")
        except:
            pass
    results = search.limit(10).to_pandas()
    state["retrieved_docs"] = [
        {
            "text": r["text"],
            "chunk_id": r.get("chunk_id", ""),
            "legal_domain": r.get("legal_domain", ""),
            "section_number": r.get("section_number", 0),
            "document_title": r.get("document_title", ""),
            "source_file": r.get("source_file", ""),
            "distance": float(r.get("_distance", 999))
        }
        for _, r in results.iterrows()
    ]
    state["node_timings"]["context_retrieve"] = time.time() - t0
    return state


def rerank_node(state):
    t0 = time.time()
    docs = state.get("retrieved_docs", [])
    if not docs:
        state["reranked_docs"] = []
        state["node_timings"]["rerank"] = time.time() - t0
        return state

    for doc in docs:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Rate relevance 0-1. Return only a number."),
                ("human", "Question: {q}\n\nDocument: {d}")
            ])
            chain = prompt | scorer_llm | StrOutputParser()
            s = chain.invoke({
                "q": state["rewritten_query"],
                "d": doc["text"][:600]
            })
            score = float(re.search(r'[\d.]+', s).group())
            doc["relevance_score"] = min(max(score, 0.0), 1.0)
        except:
            doc["relevance_score"] = 0.5

    docs.sort(key=lambda x: x["relevance_score"], reverse=True)
    state["reranked_docs"] = docs[:3]
    state["node_timings"]["rerank"] = time.time() - t0
    return state


def generate_answer_node(state):
    t0 = time.time()
    contexts = state.get("reranked_docs", [])
    if not contexts:
        state["final_answer"] = (
            "متأسفانه اطلاعات مرتبطی یافت نشد."
        )
        state["node_timings"]["generate_answer"] = time.time() - t0
        return state

    ctx_parts = []
    for i, d in enumerate(contexts, 1):
        src = d.get("source_file", "unknown")
        domain = d.get("legal_domain", "unknown")
        sec = d.get("section_number", "-")
        ctx_parts.append(
            f"--- Document {i} (source: {src}, domain: {domain}, "
            f"article: {sec}) ---\n{d['text']}"
        )
    ctx_text = "\n\n".join(ctx_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an Iranian legal expert. Answer based on documents. "
         "Answer in Persian. Cite article numbers."),
        ("human", "Question: {q}\n\nDocuments:\n{ctx}")
    ])
    chain = prompt | main_llm | StrOutputParser()
    state["final_answer"] = chain.invoke({
        "q": state["rewritten_query"],
        "ctx": ctx_text
    })
    state["node_timings"]["generate_answer"] = time.time() - t0
    return state


def end_early_node(state):
    state["final_answer"] = state.get(
        "intent_response",
        "سلام! چگونه می‌توانم کمکتان کنم؟"
    )
    return state


# ============================================================
# ROUTING
# ============================================================

def route_after_classify(state):
    if state.get("intent") in ("greeting", "abusive"):
        return "end_early"
    return "extract_metadata"


# ============================================================
# GRAPH BUILDER
# ============================================================

wf = StateGraph(AgentState)
wf.add_node("rewrite_query", rewrite_query_node)
wf.add_node("classify_intent", classify_intent_node)
wf.add_node("extract_metadata", extract_metadata_node)
wf.add_node("context_retrieve", context_retrieve_node)
wf.add_node("rerank", rerank_node)
wf.add_node("generate_answer", generate_answer_node)
wf.add_node("end_early", end_early_node)

wf.set_entry_point("rewrite_query")
wf.add_edge("rewrite_query", "classify_intent")
wf.add_conditional_edges(
    "classify_intent", route_after_classify,
    {"extract_metadata": "extract_metadata", "end_early": "end_early"}
)
wf.add_edge("extract_metadata", "context_retrieve")
wf.add_edge("context_retrieve", "rerank")
wf.add_edge("rerank", "generate_answer")
wf.add_edge("generate_answer", END)
wf.add_edge("end_early", END)

agent = wf.compile()


# ============================================================
# CHAINLIT HANDLERS
# ============================================================

@cl.on_chat_start
async def on_start():
    await cl.Message(
        content=(
            "سلام! 👋 من دستیار حقوقی هوشمند هستم.\n\n"
            "می‌توانید سوالات خود درباره قوانین کار، مالیات، جزا و سایر "
            "قوانین ایران را از من بپرسید.\n\n"
            "برای شروع، یک سوال حقوقی بپرسید! 📚⚖️"
        )
    ).send()
    cl.user_session.set("query_count", 0)


@cl.on_message
async def on_message(message: cl.Message):
    count = cl.user_session.get("query_count", 0)
    cl.user_session.set("query_count", count + 1)

    loading_msg = cl.Message(content="🔍 در حال پردازش سوال شما...")
    await loading_msg.send()

    try:
        initial_state = {
            "original_query": message.content,
            "rewritten_query": "",
            "intent": "",
            "intent_response": "",
            "metadata_filter": {},
            "retrieved_docs": [],
            "reranked_docs": [],
            "final_answer": "",
            "retry_count": 0,
            "node_timings": {}
        }

        result = agent.invoke(initial_state)

        if result.get("intent") in ("greeting", "abusive"):
            response = result.get("intent_response", "سلام!")
        else:
            response = result.get("final_answer", "متاسفانه پاسخی یافت نشد.")

            contexts = result.get("reranked_docs", [])
            if contexts:
                response += "\n\n---\n\n📄 **منابع مورد استفاده:**\n"
                for i, ctx in enumerate(contexts[:3], 1):
                    doc_title = ctx.get("document_title", "نامشخص")
                    section_num = ctx.get("section_number", "")
                    section_info = (
                        f" - ماده {section_num}" if section_num else ""
                    )
                    response += f"\n{i}. {doc_title}{section_info}"

        timings = result.get("node_timings", {})
        total_time = sum(timings.values())

        if timings:
            timing_details = " | ".join(
                [f"{k}: {v:.1f}s" for k, v in timings.items()]
            )
            response += f"\n\n⏱️ *زمان پردازش: {total_time:.1f} ثانیه*"
            response += f"\n*جزئیات: {timing_details}*"

        await loading_msg.remove()
        await cl.Message(content=response).send()

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        await loading_msg.remove()
        await cl.Message(
            content=(
                f"❌ متاسفانه خطایی رخ داد:\n\n{str(e)}\n\n"
                "لطفاً دوباره تلاش کنید."
            )
        ).send()


@cl.on_chat_end
async def end():
    query_count = cl.user_session.get("query_count", 0)
    logger.info(f"Chat ended. Total queries: {query_count}")


if __name__ == "__main__":
    print("Chainlit app loaded. Run with: chainlit run app.py -w --port 8000")
