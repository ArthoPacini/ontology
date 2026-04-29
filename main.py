"""
FastAPI app — all routes for the operational knowledge graph.
Now using integer IDs, boolean status columns, integer priorities, and audit logging.
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel

import db
from config import load_schema, reload_schema, add_to_schema, restore_schema
from extraction import extract_and_upsert
from query import query_graph
from reactor import react_to_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    schema = load_schema()
    logger.info(f"Starting Ontology for: {schema.business_name} ({schema.business_type})")
    await db.get_connection()
    logger.info("Database initialized")
    
    # Save schema snapshot on boot
    version = await db.save_schema_snapshot(schema.raw_yaml, "Server startup snapshot")
    logger.info(f"Schema history snapshot saved (v{version})")
    
    yield
    logger.info("Shutting down...")
    await db.close_connection()


app = FastAPI(
    title="Ontology — Operational Knowledge Graph",
    description="Business-agnostic knowledge graph with reactive reasoning, "
                "entity disambiguation, and dynamic schema evolution.",
    version="0.4.0",
    lifespan=lifespan,
)


# ── Request models ──────────────────────────────────────────────────────────────

class IngestRawRequest(BaseModel):
    source: str
    content: str | dict
    metadata: dict = {}

class IngestEventRequest(BaseModel):
    event_type: str
    table: str
    data: dict
    old_data: dict | None = None

class QueryRequest(BaseModel):
    question: str
    max_depth: int = 2
    lookback_days: int | None = None
    allowed_relations: list[str] | None = None

class ExecuteActionRequest(BaseModel):
    executor: str = "external"

class SchemaRestoreRequest(BaseModel):
    version: int


# ── Background pipeline ────────────────────────────────────────────────────────

async def run_pipeline(event_id: int, content: str):
    schema = load_schema()

    try:
        extraction = await extract_and_upsert(schema, event_id, content)
        logger.info(f"Extraction: {extraction['entities_extracted']} entities, {extraction['edges_extracted']} edges")
    except Exception as e:
        logger.error(f"Extraction failed for event {event_id}: {e}")
        await db.mark_event_processed(event_id)
        return

    try:
        reaction = await react_to_event(
            schema, event_id, content,
            extracted_entity_names=extraction.get("entity_names"),
        )
        if reaction:
            logger.info(f"Reaction: {reaction['trigger_summary'][:60]}... → {reaction['actions_stored']} actions")
    except Exception as e:
        logger.error(f"Reaction failed for event {event_id}: {e}")

    await db.mark_event_processed(event_id)


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    schema = load_schema()
    return {
        "service": "Ontology — Operational Knowledge Graph",
        "version": "0.4.0",
        "business": schema.business_name,
        "business_type": schema.business_type,
        "entity_types": schema.entity_types,
    }


@app.post("/ingest/raw")
async def ingest_raw(body: IngestRawRequest, background_tasks: BackgroundTasks):
    content_str = body.content if isinstance(body.content, str) else json.dumps(body.content)
    event_id = await db.store_event(body.source, content_str, body.metadata)
    background_tasks.add_task(run_pipeline, event_id, content_str)
    return {"status": "accepted", "event_id": event_id}


@app.post("/ingest/event")
async def ingest_event(body: IngestEventRequest):
    if body.event_type == "insert":
        name = body.data.get("name", body.data.get("id", "unknown"))
        etype = body.data.get("entity_type", body.table.rstrip("s").capitalize())
        summary = body.data.get("summary", "")
        meta = {k: v for k, v in body.data.items() if k not in ("name", "entity_type", "summary", "relationships", "aliases")}

        entity_id = await db.upsert_entity(name, etype, summary, meta)

        for alias in body.data.get("aliases", []):
            await db.add_alias(entity_id, alias)

        edges_created = 0
        for rel in body.data.get("relationships", []):
            target = await db.get_entity_by_name(rel["target"])
            tid = target["id"] if target else await db.upsert_entity(
                rel["target"], rel.get("target_type", "Unknown")
            )
            await db.create_edge(entity_id, tid, rel["relation"], rel.get("fact", ""))
            edges_created += 1

        return {"status": "ok", "entity_id": entity_id, "edges_created": edges_created}

    elif body.event_type == "update":
        name = body.data.get("name", "")
        if not name:
            raise HTTPException(400, "Update events require 'name' in data")
        entity = await db.get_entity_by_name(name)
        if not entity:
            entity = await db.resolve_entity(name)
        if not entity:
            raise HTTPException(404, f"Entity '{name}' not found")

        if "relationships" in body.data:
            await db.invalidate_edges(source_id=entity["id"])
            for rel in body.data["relationships"]:
                target = await db.get_entity_by_name(rel["target"])
                tid = target["id"] if target else await db.upsert_entity(
                    rel["target"], rel.get("target_type", "Unknown")
                )
                await db.create_edge(entity["id"], tid, rel["relation"], rel.get("fact", ""))

        etype = body.data.get("entity_type", entity["entity_type"])
        summary = body.data.get("summary", entity["summary"])
        await db.upsert_entity(entity["name"], etype, summary)
        return {"status": "ok", "entity_id": entity["id"]}

    elif body.event_type == "delete":
        name = body.data.get("name", "")
        entity = await db.get_entity_by_name(name)
        if entity:
            await db.invalidate_edges(source_id=entity["id"])
            await db.invalidate_edges(target_id=entity["id"])
        return {"status": "ok", "message": f"Edges invalidated for '{name}'"}

    raise HTTPException(400, f"Unknown event_type: {body.event_type}")


# ── Graph ───────────────────────────────────────────────────────────────────────

@app.get("/graph/subgraph")
async def get_subgraph(
    entity_name: str = Query(...),
    depth: int = Query(2, ge=1, le=5),
    lookback_days: int | None = Query(None, ge=1),
    relations: str | None = Query(None),
):
    entity = await db.get_entity_by_name(entity_name)
    if not entity:
        results = await db.search_entities(entity_name, limit=1)
        entity = results[0] if results else None
    if not entity:
        raise HTTPException(404, f"Entity '{entity_name}' not found")

    allowed = relations.split(",") if relations else None
    subgraph = await db.traverse_subgraph(entity["id"], depth, lookback_days, allowed)
    return {"root_entity": {"id": entity["id"], "name": entity["name"], "type": entity["entity_type"]}, "depth": depth, **subgraph}


@app.get("/entities")
async def list_entities(type: str | None = Query(None)):
    entities = await db.list_entities(entity_type=type)
    return {"count": len(entities), "entities": entities}


# ── Query & Semantic Search ─────────────────────────────────────────────────────

@app.post("/query")
async def query_endpoint(body: QueryRequest):
    schema = load_schema()
    return await query_graph(
        schema, body.question, body.max_depth, body.lookback_days, body.allowed_relations,
    )

@app.get("/search/entities")
async def search_entities_endpoint(q: str = Query(...), limit: int = Query(5, ge=1, le=50)):
    results = await db.semantic_search_entities(q, limit)
    # Fetch aliases for returned entities
    for r in results:
        r["aliases"] = await db.get_aliases(r["id"])
    return {"count": len(results), "results": results}

@app.get("/search/events")
async def search_events_endpoint(q: str = Query(...), limit: int = Query(5, ge=1, le=50)):
    results = await db.semantic_search_events(q, limit)
    return {"count": len(results), "results": results}


# ── Audit Log ───────────────────────────────────────────────────────────────────

@app.get("/audit")
async def get_audit(action: str | None = Query(None), limit: int = Query(100, le=1000)):
    logs = await db.get_audit_log(action_filter=action, limit=limit)
    return {"count": len(logs), "logs": logs}


# ── Events ──────────────────────────────────────────────────────────────────────

@app.get("/events/pending")
async def pending_events():
    events = await db.get_pending_events()
    return {"count": len(events), "events": events}


# ── Reactions ───────────────────────────────────────────────────────────────────

@app.get("/reactions")
async def list_reactions(limit: int = Query(50, ge=1, le=200)):
    reactions = await db.get_all_reactions(limit)
    for r in reactions:
        r["actions"] = await db.get_actions_for_reaction(r["id"])
    return {"count": len(reactions), "reactions": reactions}


@app.get("/reactions/pending")
async def pending_reactions():
    reactions = await db.get_pending_reactions()
    for r in reactions:
        r["actions"] = await db.get_actions_for_reaction(r["id"])
    return {"count": len(reactions), "reactions": reactions}


# ── Actions ─────────────────────────────────────────────────────────────────────

@app.get("/actions/pending")
async def pending_actions():
    actions = await db.get_pending_actions()
    return {"count": len(actions), "actions": actions}


@app.post("/actions/{action_id}/execute")
async def execute_action_endpoint(action_id: int, body: ExecuteActionRequest):
    ok = await db.execute_action(action_id, body.executor)
    if not ok:
        raise HTTPException(404, "Action not found or already processed")
    return {"status": "ok", "action_id": action_id}


@app.post("/actions/{action_id}/fail")
async def fail_action_endpoint(action_id: int, body: ExecuteActionRequest):
    ok = await db.fail_action(action_id, body.executor)
    if not ok:
        raise HTTPException(404, "Action not found or already processed")
    return {"status": "ok", "action_id": action_id}


# ── Schema Evolution ───────────────────────────────────────────────────────────

@app.get("/schema")
async def get_schema():
    return load_schema().raw


@app.get("/schema/history")
async def list_schema_history(limit: int = Query(20)):
    history = await db.get_schema_history(limit)
    return {"count": len(history), "history": history}


@app.post("/schema/restore")
async def schema_restore(body: SchemaRestoreRequest):
    ver = await db.get_schema_version(body.version)
    if not ver:
        raise HTTPException(404, "Version not found")
    
    restore_schema(ver["schema_yaml"])
    
    # Save the restoration as a new version snapshot
    new_ver = await db.save_schema_snapshot(ver["schema_yaml"], f"Rolled back to v{body.version}")
    
    return {"status": "restored", "new_version": new_ver}


@app.get("/schema/proposals")
async def list_proposals():
    proposals = await db.get_pending_proposals()
    return {"count": len(proposals), "proposals": proposals}


@app.post("/schema/proposals/{proposal_id}/approve")
async def approve_proposal_endpoint(proposal_id: int):
    proposals = await db.get_pending_proposals()
    target = next((p for p in proposals if p["id"] == proposal_id), None)
    if not target:
        raise HTTPException(404, "Proposal not found or already resolved")

    changed = add_to_schema(
        target.get("proposed_entity_types", []),
        target.get("proposed_relationship_types", []),
    )
    
    await db.approve_proposal(proposal_id)
    
    # Save new snapshot if changed
    if changed:
        schema = load_schema()
        await db.save_schema_snapshot(schema.raw_yaml, f"Approved proposal {proposal_id}")
        
    return {"status": "approved", "schema_updated": changed}


@app.post("/schema/proposals/{proposal_id}/reject")
async def reject_proposal_endpoint(proposal_id: int):
    ok = await db.reject_proposal(proposal_id)
    if not ok:
        raise HTTPException(404, "Proposal not found or already resolved")
    return {"status": "rejected"}


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
