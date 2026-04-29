"""
LLM query loop: question -> find entity -> subgraph -> LLM -> answer.
"""

import logging

from config import SchemaConfig
from extraction import call_openrouter
import db

logger = logging.getLogger(__name__)


def serialize_subgraph(subgraph: dict) -> str:
    """Serialize subgraph. Uses entity_aliases table implicitly by showing name."""
    lines = ["ENTITIES:"]
    for n in subgraph["nodes"]:
        lines.append(
            f"  - [{n['entity_type']}] {n['name']}: "
            f"{n['summary']} (depth={n.get('depth', '?')})"
        )
    lines.append("\nRELATIONSHIPS:")
    for e in subgraph["edges"]:
        lines.append(
            f"  - {e['source_name']} --[{e['relation']}]--> {e['target_name']}: {e['fact']}"
        )
    return "\n".join(lines)


def extract_keywords(question: str) -> list[str]:
    stop_words = {
        "what", "who", "where", "when", "how", "why", "which", "is", "are",
        "was", "were", "do", "does", "did", "the", "a", "an", "and", "or",
        "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
        "she", "they", "them", "this", "that", "just", "all", "if", "not",
        "no", "can", "will", "would", "should", "could", "has", "have",
        "had", "been", "be", "am", "first", "risk", "call", "about",
        "need", "tell", "show", "give", "get", "make", "know", "think",
        "down", "up", "out", "broke", "broken",
    }
    words = question.replace("?", "").replace(".", "").replace(",", "").replace("!", "").split()
    return [w for w in words if w.lower() not in stop_words and len(w) > 1]


async def find_best_entity(question: str) -> dict | None:
    keywords = extract_keywords(question)
    best = None
    best_score = 0

    for i in range(len(keywords) - 1):
        combo = f"{keywords[i]} {keywords[i + 1]}"
        results = await db.search_entities(combo, limit=3)
        for r in results:
            if combo.lower() in r["name"].lower():
                score = 90 + (10 / max(len(r["name"]), 1))
                if score > best_score:
                    best_score = score
                    best = r

    for kw in keywords:
        results = await db.search_entities(kw, limit=5)
        for r in results:
            name_lower = r["name"].lower()
            kw_lower = kw.lower()
            if name_lower == kw_lower:
                score = 100
            elif kw_lower in name_lower:
                score = 50 + (10 / max(len(r["name"]), 1))
            else:
                score = 10
            if score > best_score:
                best_score = score
                best = r

    return best


async def query_graph(
    schema: SchemaConfig,
    question: str,
    max_depth: int = 2,
    lookback_days: int | None = None,
    allowed_relations: list[str] | None = None,
) -> dict:
    await db.audit("query_started", detail={"question": question})

    entity = await find_best_entity(question)
    if not entity:
        await db.audit("query_failed", detail={"reason": "No entities found"})
        return {
            "answer": "No relevant entities found in the knowledge graph.",
            "entity_found": None,
            "subgraph_size": {"nodes": 0, "edges": 0},
        }

    subgraph = await db.traverse_subgraph(
        entity["id"], max_depth, lookback_days, allowed_relations,
    )
    subgraph_text = serialize_subgraph(subgraph)

    system_prompt = (
        f"You are an operational analyst for {schema.business_name} ({schema.business_type}).\n"
        "You reason over a knowledge graph of business entities and their relationships.\n"
        "Answer concisely. Follow graph relationships explicitly.\n"
        "Reference specific entities and relationships from the context.\n"
    )
    user_prompt = (
        f"GRAPH CONTEXT (subgraph around \"{entity['name']}\"):\n"
        f"{subgraph_text}\n\nQUESTION: {question}"
    )

    try:
        answer = await call_openrouter(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=0.3, max_tokens=2048,
        )
        await db.audit("query_completed", "entities", entity["id"], {"answer_length": len(answer)})
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        answer = f"Error querying LLM: {e}"
        await db.audit("query_failed", "entities", entity["id"], {"error": str(e)})

    return {
        "answer": answer,
        "entity_found": {"id": entity["id"], "name": entity["name"], "type": entity["entity_type"]},
        "subgraph_size": {"nodes": len(subgraph["nodes"]), "edges": len(subgraph["edges"])},
    }
