"""
Async LLM extraction pipeline.

Features:
  - Entity disambiguation via entity_aliases table
  - Dynamic schema evolution (LLM proposes new types)
  - Integer IDs throughout, priority as integer
"""

import os
import json
import logging
import httpx
from dotenv import load_dotenv

from config import SchemaConfig
import db

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_ID = os.getenv("MODEL_ID", "deepseek/deepseek-v4-flash")


def build_extraction_prompt(schema: SchemaConfig, content: str) -> str:
    return (
        f'You are ingesting operational data for a {schema.business_type} '
        f'called "{schema.business_name}".\n\n'
        f"Known entity types: [{schema.entity_types_str()}]\n\n"
        f"Known relationships:\n{schema.relationship_types_str()}\n\n"
        f"Extraction hints:\n{schema.extraction_hints_str()}\n\n"
        "Extract ALL entities and relationships from the content below.\n"
        'Use canonical names for entities (e.g., "Van V1" not "the van").\n'
        "If an entity might be known by multiple names, list alternatives.\n"
        "Relationship names must match known types when possible.\n\n"
        "If you find important entities or relationships that don't fit any known type, "
        "include them in schema_proposals so the system can learn.\n\n"
        "Respond ONLY with valid JSON (no markdown, no explanation):\n"
        "{\n"
        '  "entities": [{"name": "str", "type": "str", "summary": "str", '
        '"aliases": ["alt_name_1"]}],\n'
        '  "edges": [{"from_name": "str", "to_name": "str", "relation": "str", '
        '"fact": "str"}],\n'
        '  "schema_proposals": {\n'
        '    "new_entity_types": ["TypeName"],\n'
        '    "new_relationship_types": [{"name": "REL", "from": "Type1", "to": "Type2"}]\n'
        "  }\n"
        "}\n\n"
        f"Content:\n---\n{content}\n---"
    )


async def call_openrouter(
    messages: list[dict], temperature: float = 0.1, max_tokens: int = 4096,
) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_json_response(content: str) -> dict:
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
    return json.loads(content)


async def resolve_or_create_entity(
    name: str, entity_type: str, summary: str = "", aliases: list[str] | None = None,
) -> int:
    """Disambiguation-aware entity resolution. Returns integer entity ID."""
    # Try canonical name + type
    existing = await db.resolve_entity(name, entity_type)
    if existing:
        return existing["id"]

    # Try without type (alias might be under a different type)
    existing = await db.resolve_entity(name)
    if existing:
        if name != existing["name"]:
            await db.add_alias(existing["id"], name)
        return existing["id"]

    # Create new
    eid = await db.upsert_entity(name, entity_type, summary)

    # Register aliases
    if aliases:
        for alias in aliases:
            if alias and alias != name:
                await db.add_alias(eid, alias)

    return eid


async def extract_and_upsert(schema: SchemaConfig, event_id: int, content: str) -> dict:
    """Full extraction: LLM → entities (with disambiguation) → edges → schema proposals."""
    prompt = build_extraction_prompt(schema, content)

    await db.audit("extraction_started", "events", event_id)

    try:
        raw_response = await call_openrouter([{"role": "user", "content": prompt}])
        extracted = parse_json_response(raw_response)
    except Exception as e:
        logger.error(f"LLM extraction failed for event {event_id}: {e}")
        await db.audit("extraction_failed", "events", event_id, {"error": str(e)})
        raise

    entities_created = 0
    edges_created = 0
    name_to_id: dict[str, int] = {}

    for ent in extracted.get("entities", []):
        name = ent.get("name", "").strip()
        etype = ent.get("type", "").strip()
        if not name or not etype:
            continue

        aliases = [a.strip() for a in ent.get("aliases", []) if a.strip()]
        eid = await resolve_or_create_entity(name, etype, ent.get("summary", ""), aliases)
        name_to_id[name] = eid
        for alias in aliases:
            name_to_id[alias] = eid
        entities_created += 1

    for edge in extracted.get("edges", []):
        fn = edge.get("from_name", "").strip()
        tn = edge.get("to_name", "").strip()
        rel = edge.get("relation", "").strip()
        if not fn or not tn or not rel:
            continue

        sid = name_to_id.get(fn)
        if not sid:
            sid = await resolve_or_create_entity(fn, "Unknown")
            name_to_id[fn] = sid

        tid = name_to_id.get(tn)
        if not tid:
            tid = await resolve_or_create_entity(tn, "Unknown")
            name_to_id[tn] = tid

        await db.create_edge(sid, tid, rel, edge.get("fact", ""), event_id=event_id)
        edges_created += 1

    # Schema proposals
    proposals = extracted.get("schema_proposals", {})
    new_etypes = proposals.get("new_entity_types", [])
    new_rels = proposals.get("new_relationship_types", [])

    if new_etypes or new_rels:
        known_etypes = set(schema.entity_types)
        known_rels = {rt.name for rt in schema.relationship_types}
        novel_etypes = [t for t in new_etypes if t not in known_etypes]
        novel_rels = [r for r in new_rels if r.get("name") not in known_rels]

        if novel_etypes or novel_rels:
            await db.store_schema_proposal(event_id, novel_etypes, novel_rels)
            logger.info(f"Schema proposal: {novel_etypes} types, {len(novel_rels)} rels")

    await db.audit("extraction_completed", "events", event_id, {
        "entities": entities_created, "edges": edges_created,
    })
    logger.info(f"Extraction done for event {event_id}: {entities_created} entities, {edges_created} edges")

    return {
        "event_id": event_id,
        "entities_extracted": entities_created,
        "edges_extracted": edges_created,
        "entity_names": list(set(name_to_id.keys())),
    }
