"""
SQLite schema, connection management, and all query functions.

Design decisions:
  - INTEGER PRIMARY KEY AUTOINCREMENT on all tables (rowid = PK, fastest in SQLite)
  - entity_aliases is a separate table (easier to query/add than JSON column)
  - No status strings — boolean columns (is_processed, is_executed, etc.)
  - Priority is INTEGER (0=critical, 1=high, 2=medium, 3=low)
  - schema_history records every schema version for rollback
  - audit_log records every system action for full traceability
"""

import json
import aiosqlite
from datetime import datetime, timezone, timedelta
from typing import Any

from config import DB_PATH

# ── Priority mapping ───────────────────────────────────────────────────────────

PRIORITY_MAP = {"critical": 0, "high": 1, "medium": 2, "low": 3}
PRIORITY_LABELS = {0: "critical", 1: "high", 2: "medium", 3: "low"}


def priority_to_int(label: str) -> int:
    return PRIORITY_MAP.get(label.lower().strip(), 2)


def priority_to_label(val: int) -> str:
    return PRIORITY_LABELS.get(val, "medium")


# ── Schema DDL ──────────────────────────────────────────────────────────────────

INIT_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    source_metadata TEXT DEFAULT '{}',
    is_processed INTEGER DEFAULT 0,
    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    summary TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    alias TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(entity_id, alias)
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES entities(id),
    target_id INTEGER NOT NULL REFERENCES entities(id),
    relation TEXT NOT NULL,
    fact TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    valid_from TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    valid_until TEXT DEFAULT NULL,
    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    event_id INTEGER REFERENCES events(id),
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER REFERENCES events(id),
    trigger_summary TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    is_resolved INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    resolved_at TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reaction_id INTEGER NOT NULL REFERENCES reactions(id),
    action_type TEXT NOT NULL,
    priority INTEGER DEFAULT 2,
    target TEXT DEFAULT '',
    message TEXT DEFAULT '',
    rationale TEXT DEFAULT '',
    is_executed INTEGER DEFAULT 0,
    is_failed INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    executed_at TEXT DEFAULT NULL,
    executor TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS schema_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER REFERENCES events(id),
    proposed_entity_types TEXT DEFAULT '[]',
    proposed_relationship_types TEXT DEFAULT '[]',
    is_approved INTEGER DEFAULT 0,
    is_rejected INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    resolved_at TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS schema_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    schema_yaml TEXT NOT NULL,
    change_description TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    ref_table TEXT,
    ref_id INTEGER,
    detail TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
CREATE INDEX IF NOT EXISTS idx_edges_valid ON edges(valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_edges_event ON edges(event_id);
CREATE INDEX IF NOT EXISTS idx_edges_recorded ON edges(recorded_at);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON entity_aliases(alias);
CREATE INDEX IF NOT EXISTS idx_aliases_entity ON entity_aliases(entity_id);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
CREATE INDEX IF NOT EXISTS idx_events_recorded ON events(recorded_at);
CREATE INDEX IF NOT EXISTS idx_events_processed ON events(is_processed);
CREATE INDEX IF NOT EXISTS idx_reactions_resolved ON reactions(is_resolved);
CREATE INDEX IF NOT EXISTS idx_reactions_event ON reactions(event_id);
CREATE INDEX IF NOT EXISTS idx_actions_executed ON actions(is_executed, is_failed);
CREATE INDEX IF NOT EXISTS idx_actions_reaction ON actions(reaction_id);
CREATE INDEX IF NOT EXISTS idx_actions_priority ON actions(priority);
CREATE INDEX IF NOT EXISTS idx_proposals_pending ON schema_proposals(is_approved, is_rejected);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_ref ON audit_log(ref_table, ref_id);
CREATE INDEX IF NOT EXISTS idx_schema_history_version ON schema_history(version);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_entities USING vec0(
    id INTEGER PRIMARY KEY,
    embedding float[384]
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_events USING vec0(
    id INTEGER PRIMARY KEY,
    embedding float[384]
);
"""

# ── Connection ──────────────────────────────────────────────────────────────────

_conn: aiosqlite.Connection | None = None


async def get_connection() -> aiosqlite.Connection:
    global _conn
    if _conn is None:
        _conn = await aiosqlite.connect(str(DB_PATH))
        _conn.row_factory = aiosqlite.Row
        
        # Load sqlite-vec extension
        import sqlite_vec
        await _conn.enable_load_extension(True)
        await _conn.load_extension(sqlite_vec.loadable_path())
        await _conn.enable_load_extension(False)
        
        await _conn.executescript(INIT_SQL)
    return _conn


async def close_connection():
    global _conn
    if _conn:
        await _conn.close()
        _conn = None


# ── Helpers ─────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    d = dict(row)
    d.pop("embedding", None)
    # Convert priority int to label if present
    if "priority" in d and isinstance(d["priority"], int):
        d["priority_label"] = priority_to_label(d["priority"])
    return d


# ── Audit Log ───────────────────────────────────────────────────────────────────

async def audit(
    action: str, ref_table: str | None = None, ref_id: int | None = None,
    detail: dict | None = None,
):
    """Record any system action in the audit log."""
    conn = await get_connection()
    await conn.execute(
        "INSERT INTO audit_log (action, ref_table, ref_id, detail) VALUES (?, ?, ?, ?)",
        (action, ref_table, ref_id, json.dumps(detail or {})),
    )
    await conn.commit()


async def get_audit_log(
    action_filter: str | None = None, limit: int = 100,
) -> list[dict]:
    conn = await get_connection()
    if action_filter:
        cursor = await conn.execute(
            "SELECT * FROM audit_log WHERE action = ? ORDER BY id DESC LIMIT ?",
            (action_filter, limit),
        )
    else:
        cursor = await conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,),
        )
    return [row_to_dict(r) for r in await cursor.fetchall()]


# ── Events ──────────────────────────────────────────────────────────────────────

async def store_event(source: str, content: str, metadata: dict | None = None) -> int:
    conn = await get_connection()
    meta = json.dumps(metadata or {})
    cursor = await conn.execute(
        "INSERT INTO events (source, content, source_metadata) VALUES (?, ?, ?)",
        (source, content, meta),
    )
    eid = cursor.lastrowid
    
    from embed import get_embedding
    emb = get_embedding(content)
    if emb:
        await conn.execute("INSERT INTO vec_events(id, embedding) VALUES (?, ?)", (eid, emb))
        
    await conn.commit()
    await audit("event_ingested", "events", eid, {"source": source, "length": len(content)})
    return eid


async def mark_event_processed(event_id: int):
    conn = await get_connection()
    await conn.execute("UPDATE events SET is_processed = 1 WHERE id = ?", (event_id,))
    await conn.commit()
    await audit("event_processed", "events", event_id)


async def get_pending_events() -> list[dict]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM events WHERE is_processed = 0 ORDER BY recorded_at"
    )
    return [row_to_dict(r) for r in await cursor.fetchall()]


async def get_event(event_id: int) -> dict | None:
    conn = await get_connection()
    cursor = await conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = await cursor.fetchone()
    return row_to_dict(row) if row else None


# ── Entities ────────────────────────────────────────────────────────────────────

async def resolve_entity(name: str, entity_type: str | None = None) -> dict | None:
    """Find entity by canonical name or alias. Optional type filter."""
    conn = await get_connection()

    # Canonical name (fast path)
    if entity_type:
        cursor = await conn.execute(
            "SELECT * FROM entities WHERE name = ? AND entity_type = ?", (name, entity_type),
        )
    else:
        cursor = await conn.execute("SELECT * FROM entities WHERE name = ?", (name,))
    row = await cursor.fetchone()
    if row:
        return row_to_dict(row)

    # Search aliases table
    if entity_type:
        cursor = await conn.execute(
            """SELECT e.* FROM entities e
               JOIN entity_aliases a ON e.id = a.entity_id
               WHERE a.alias = ? AND e.entity_type = ? LIMIT 1""",
            (name, entity_type),
        )
    else:
        cursor = await conn.execute(
            """SELECT e.* FROM entities e
               JOIN entity_aliases a ON e.id = a.entity_id
               WHERE a.alias = ? LIMIT 1""",
            (name,),
        )
    row = await cursor.fetchone()
    return row_to_dict(row) if row else None


async def upsert_entity(
    name: str, entity_type: str, summary: str = "", metadata: dict | None = None,
) -> int:
    """Insert or update entity by name + type. Returns entity ID."""
    conn = await get_connection()
    now = now_iso()
    meta = json.dumps(metadata or {})

    cursor = await conn.execute(
        "SELECT id FROM entities WHERE name = ? AND entity_type = ?", (name, entity_type),
    )
    row = await cursor.fetchone()

    if row:
        eid = row["id"]
        if summary:
            await conn.execute(
                "UPDATE entities SET summary = ?, metadata = ?, updated_at = ? WHERE id = ?",
                (summary, meta, now, eid),
            )
        else:
            await conn.execute(
                "UPDATE entities SET metadata = ?, updated_at = ? WHERE id = ?",
                (meta, now, eid),
            )
        await conn.commit()
        await audit("entity_updated", "entities", eid, {"name": name})
    else:
        cursor = await conn.execute(
            """INSERT INTO entities (name, entity_type, summary, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, entity_type, summary, meta, now, now),
        )
        eid = cursor.lastrowid
        await audit("entity_created", "entities", eid, {"name": name, "type": entity_type})

    from embed import get_embedding
    emb = get_embedding(f"{name} {entity_type} {summary}")
    if emb:
        await conn.execute("DELETE FROM vec_entities WHERE id = ?", (eid,))
        await conn.execute("INSERT INTO vec_entities(id, embedding) VALUES (?, ?)", (eid, emb))

    await conn.commit()
    return eid


async def add_alias(entity_id: int, alias: str):
    """Add an alias to an entity (idempotent)."""
    conn = await get_connection()
    try:
        await conn.execute(
            "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
            (entity_id, alias),
        )
        await conn.commit()
        await audit("alias_added", "entities", entity_id, {"alias": alias})
    except Exception:
        pass  # UNIQUE constraint — alias already exists


async def get_aliases(entity_id: int) -> list[str]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias",
        (entity_id,),
    )
    return [r["alias"] for r in await cursor.fetchall()]


async def get_entity_by_name(name: str) -> dict | None:
    conn = await get_connection()
    cursor = await conn.execute("SELECT * FROM entities WHERE name = ?", (name,))
    row = await cursor.fetchone()
    return row_to_dict(row) if row else None


async def search_entities(query: str, limit: int = 10) -> list[dict]:
    """Search entities by name or aliases using LIKE."""
    conn = await get_connection()
    pattern = f"%{query}%"
    cursor = await conn.execute(
        """SELECT DISTINCT e.* FROM entities e
           LEFT JOIN entity_aliases a ON e.id = a.entity_id
           WHERE e.name LIKE ? OR a.alias LIKE ?
           ORDER BY e.name LIMIT ?""",
        (pattern, pattern, limit),
    )
    return [row_to_dict(r) for r in await cursor.fetchall()]


async def list_entities(entity_type: str | None = None) -> list[dict]:
    conn = await get_connection()
    if entity_type:
        cursor = await conn.execute(
            "SELECT * FROM entities WHERE entity_type = ? ORDER BY name", (entity_type,),
        )
    else:
        cursor = await conn.execute("SELECT * FROM entities ORDER BY entity_type, name")

    entities = [row_to_dict(r) for r in await cursor.fetchall()]

    # Batch-fetch aliases
    for ent in entities:
        ent["aliases"] = await get_aliases(ent["id"])

    return entities


# ── Edges ───────────────────────────────────────────────────────────────────────

async def create_edge(
    source_id: int, target_id: int, relation: str, fact: str = "",
    confidence: float = 1.0, event_id: int | None = None, metadata: dict | None = None,
) -> int:
    conn = await get_connection()
    now = now_iso()
    meta = json.dumps(metadata or {})
    cursor = await conn.execute(
        """INSERT INTO edges (source_id, target_id, relation, fact, confidence,
           valid_from, recorded_at, event_id, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (source_id, target_id, relation, fact, confidence, now, now, event_id, meta),
    )
    await conn.commit()
    eid = cursor.lastrowid
    await audit("edge_created", "edges", eid, {"relation": relation, "source": source_id, "target": target_id})
    return eid


async def invalidate_edges(
    source_id: int | None = None, target_id: int | None = None, relation: str | None = None,
) -> int:
    conn = await get_connection()
    now = now_iso()
    conditions = ["valid_until IS NULL"]
    params: list[Any] = []
    if source_id:
        conditions.append("source_id = ?")
        params.append(source_id)
    if target_id:
        conditions.append("target_id = ?")
        params.append(target_id)
    if relation:
        conditions.append("relation = ?")
        params.append(relation)
    where = " AND ".join(conditions)
    cursor = await conn.execute(
        f"UPDATE edges SET valid_until = ? WHERE {where}", [now] + params,
    )
    await conn.commit()
    return cursor.rowcount


# ── Graph Traversal (Temporal + Relational Filters) ─────────────────────────────

async def traverse_subgraph(
    start_entity_id: int,
    max_depth: int = 2,
    lookback_days: int | None = None,
    allowed_relations: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Recursive CTE with optional temporal and relational filters."""
    conn = await get_connection()

    cte_conditions = ["t.depth < ?", "e.valid_until IS NULL"]
    cte_params: list[Any] = [start_entity_id, max_depth]

    if lookback_days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        cte_conditions.append("e.recorded_at >= ?")
        cte_params.append(cutoff)

    if allowed_relations:
        placeholders = ",".join("?" * len(allowed_relations))
        cte_conditions.append(f"e.relation IN ({placeholders})")
        cte_params.extend(allowed_relations)

    cte_where = " AND ".join(cte_conditions)

    node_cursor = await conn.execute(
        f"""
        WITH RECURSIVE traversal(entity_id, depth) AS (
            SELECT ?, 0
            UNION
            SELECT CASE
                WHEN e.source_id = t.entity_id THEN e.target_id
                ELSE e.source_id
            END, t.depth + 1
            FROM traversal t
            JOIN edges e ON (e.source_id = t.entity_id OR e.target_id = t.entity_id)
            WHERE {cte_where}
        )
        SELECT DISTINCT ent.*, t.depth
        FROM traversal t
        JOIN entities ent ON ent.id = t.entity_id
        ORDER BY t.depth
        """,
        cte_params,
    )
    nodes = [row_to_dict(r) for r in await node_cursor.fetchall()]
    if not nodes:
        return {"nodes": [], "edges": []}

    node_ids = [n["id"] for n in nodes]
    ph = ",".join("?" * len(node_ids))

    edge_conditions = [
        f"e.source_id IN ({ph})",
        f"e.target_id IN ({ph})",
        "e.valid_until IS NULL",
    ]
    edge_params = node_ids + node_ids

    if lookback_days is not None:
        edge_conditions.append("e.recorded_at >= ?")
        edge_params.append(cutoff)
    if allowed_relations:
        placeholders = ",".join("?" * len(allowed_relations))
        edge_conditions.append(f"e.relation IN ({placeholders})")
        edge_params.extend(allowed_relations)

    edge_where = " AND ".join(edge_conditions)
    edge_cursor = await conn.execute(
        f"""
        SELECT e.*,
               s.name as source_name, s.entity_type as source_type,
               t.name as target_name, t.entity_type as target_type
        FROM edges e
        JOIN entities s ON e.source_id = s.id
        JOIN entities t ON e.target_id = t.id
        WHERE {edge_where}
        """,
        edge_params,
    )
    edges = [row_to_dict(r) for r in await edge_cursor.fetchall()]
    return {"nodes": nodes, "edges": edges}


# ── Reactions ───────────────────────────────────────────────────────────────────

async def store_reaction(event_id: int | None, trigger_summary: str, reasoning: str) -> int:
    conn = await get_connection()
    cursor = await conn.execute(
        "INSERT INTO reactions (event_id, trigger_summary, reasoning) VALUES (?, ?, ?)",
        (event_id, trigger_summary, reasoning),
    )
    await conn.commit()
    rid = cursor.lastrowid
    await audit("reaction_created", "reactions", rid, {"event_id": event_id})
    return rid


async def get_pending_reactions() -> list[dict]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM reactions WHERE is_resolved = 0 ORDER BY created_at"
    )
    return [row_to_dict(r) for r in await cursor.fetchall()]


async def get_all_reactions(limit: int = 50) -> list[dict]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM reactions ORDER BY created_at DESC LIMIT ?", (limit,),
    )
    return [row_to_dict(r) for r in await cursor.fetchall()]


async def resolve_reaction(reaction_id: int) -> bool:
    conn = await get_connection()
    cursor = await conn.execute(
        "UPDATE reactions SET is_resolved = 1, resolved_at = ? WHERE id = ? AND is_resolved = 0",
        (now_iso(), reaction_id),
    )
    await conn.commit()
    if cursor.rowcount > 0:
        await audit("reaction_resolved", "reactions", reaction_id)
    return cursor.rowcount > 0


# ── Actions ─────────────────────────────────────────────────────────────────────

async def store_action(
    reaction_id: int, action_type: str, priority: int = 2,
    target: str = "", message: str = "", rationale: str = "",
) -> int:
    conn = await get_connection()
    cursor = await conn.execute(
        """INSERT INTO actions (reaction_id, action_type, priority, target, message, rationale)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (reaction_id, action_type, priority, target, message, rationale),
    )
    await conn.commit()
    aid = cursor.lastrowid
    await audit("action_created", "actions", aid, {"type": action_type, "priority": priority})
    return aid


async def get_pending_actions() -> list[dict]:
    conn = await get_connection()
    cursor = await conn.execute(
        """SELECT a.*, r.trigger_summary, r.event_id
           FROM actions a JOIN reactions r ON a.reaction_id = r.id
           WHERE a.is_executed = 0 AND a.is_failed = 0
           ORDER BY a.priority ASC, a.created_at ASC"""
    )
    return [row_to_dict(r) for r in await cursor.fetchall()]


async def get_actions_for_reaction(reaction_id: int) -> list[dict]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM actions WHERE reaction_id = ? ORDER BY priority ASC, created_at",
        (reaction_id,),
    )
    return [row_to_dict(r) for r in await cursor.fetchall()]


async def execute_action(action_id: int, executor: str = "system") -> bool:
    conn = await get_connection()
    cursor = await conn.execute(
        "UPDATE actions SET is_executed = 1, executed_at = ?, executor = ? WHERE id = ? AND is_executed = 0 AND is_failed = 0",
        (now_iso(), executor, action_id),
    )
    await conn.commit()
    if cursor.rowcount > 0:
        await audit("action_executed", "actions", action_id, {"executor": executor})
    return cursor.rowcount > 0


async def fail_action(action_id: int, executor: str = "system") -> bool:
    conn = await get_connection()
    cursor = await conn.execute(
        "UPDATE actions SET is_failed = 1, executed_at = ?, executor = ? WHERE id = ? AND is_executed = 0 AND is_failed = 0",
        (now_iso(), executor, action_id),
    )
    await conn.commit()
    if cursor.rowcount > 0:
        await audit("action_failed", "actions", action_id, {"executor": executor})
    return cursor.rowcount > 0


# ── Schema Proposals ────────────────────────────────────────────────────────────

async def store_schema_proposal(
    event_id: int | None, proposed_entity_types: list[str], proposed_relationship_types: list[dict],
) -> int:
    conn = await get_connection()
    cursor = await conn.execute(
        """INSERT INTO schema_proposals (event_id, proposed_entity_types, proposed_relationship_types)
           VALUES (?, ?, ?)""",
        (event_id, json.dumps(proposed_entity_types), json.dumps(proposed_relationship_types)),
    )
    await conn.commit()
    pid = cursor.lastrowid
    await audit("schema_proposal_created", "schema_proposals", pid)
    return pid


async def get_pending_proposals() -> list[dict]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM schema_proposals WHERE is_approved = 0 AND is_rejected = 0 ORDER BY created_at"
    )
    rows = [row_to_dict(r) for r in await cursor.fetchall()]
    for r in rows:
        r["proposed_entity_types"] = json.loads(r.get("proposed_entity_types", "[]"))
        r["proposed_relationship_types"] = json.loads(r.get("proposed_relationship_types", "[]"))
    return rows


async def approve_proposal(proposal_id: int) -> bool:
    conn = await get_connection()
    cursor = await conn.execute(
        "UPDATE schema_proposals SET is_approved = 1, resolved_at = ? WHERE id = ? AND is_approved = 0 AND is_rejected = 0",
        (now_iso(), proposal_id),
    )
    await conn.commit()
    if cursor.rowcount > 0:
        await audit("schema_proposal_approved", "schema_proposals", proposal_id)
    return cursor.rowcount > 0


async def reject_proposal(proposal_id: int) -> bool:
    conn = await get_connection()
    cursor = await conn.execute(
        "UPDATE schema_proposals SET is_rejected = 1, resolved_at = ? WHERE id = ? AND is_approved = 0 AND is_rejected = 0",
        (now_iso(), proposal_id),
    )
    await conn.commit()
    if cursor.rowcount > 0:
        await audit("schema_proposal_rejected", "schema_proposals", proposal_id)
    return cursor.rowcount > 0


# ── Schema History ──────────────────────────────────────────────────────────────

async def save_schema_snapshot(schema_yaml: str, description: str = "") -> int:
    """Save current schema as a new version. Returns version number."""
    conn = await get_connection()
    cursor = await conn.execute("SELECT MAX(version) as v FROM schema_history")
    row = await cursor.fetchone()
    version = (row["v"] or 0) + 1

    cursor = await conn.execute(
        "INSERT INTO schema_history (version, schema_yaml, change_description) VALUES (?, ?, ?)",
        (version, schema_yaml, description),
    )
    await conn.commit()
    await audit("schema_snapshot", "schema_history", cursor.lastrowid, {"version": version})
    return version


async def get_schema_history(limit: int = 20) -> list[dict]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT id, version, change_description, created_at FROM schema_history ORDER BY version DESC LIMIT ?",
        (limit,),
    )
    return [row_to_dict(r) for r in await cursor.fetchall()]


async def get_schema_version(version: int) -> dict | None:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM schema_history WHERE version = ?", (version,),
    )
    row = await cursor.fetchone()
    return row_to_dict(row) if row else None

# ── Semantic Search ─────────────────────────────────────────────────────────────

async def semantic_search_entities(query: str, limit: int = 5) -> list[dict]:
    conn = await get_connection()
    from embed import get_embedding
    query_blob = get_embedding(query)
    if not query_blob:
        return []
    
    cursor = await conn.execute(
        """
        SELECT e.*, v.distance
        FROM vec_entities v
        JOIN entities e ON e.id = v.id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (query_blob, limit)
    )
    return [row_to_dict(r) for r in await cursor.fetchall()]

async def semantic_search_events(query: str, limit: int = 5) -> list[dict]:
    conn = await get_connection()
    from embed import get_embedding
    query_blob = get_embedding(query)
    if not query_blob:
        return []
        
    cursor = await conn.execute(
        """
        SELECT e.*, v.distance
        FROM vec_events v
        JOIN events e ON e.id = v.id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (query_blob, limit)
    )
    return [row_to_dict(r) for r in await cursor.fetchall()]
