"""
Seed script for the knowledge graph.

Phase 1: Structured entities (with aliases for disambiguation)
Phase 2: Raw events (LLM extraction + reactive reasoning)
Phase 3: External event → automatic reaction with individual actions
Phase 4: Inspect reactions and individual actions
Phase 5: Graph verification + manual query
"""

import httpx
import time
import json

BASE = "http://localhost:8000"


def post(path: str, body: dict) -> dict:
    r = httpx.post(f"{BASE}{path}", json=body, timeout=120.0)
    r.raise_for_status()
    return r.json()


def get(path: str, params: dict | None = None) -> dict:
    r = httpx.get(f"{BASE}{path}", params=params or {}, timeout=30.0)
    r.raise_for_status()
    return r.json()


def wait_for_processing(max_wait: int = 120):
    start = time.time()
    while time.time() - start < max_wait:
        pending = get("/events/pending")
        if pending["count"] == 0:
            print(f"  ✓ All events processed in {time.time() - start:.1f}s")
            return True
        print(f"  ... {pending['count']} events pending ({time.time() - start:.0f}s)")
        time.sleep(5)
    print(f"  ⚠ Timed out after {max_wait}s")
    return False


def main():
    print("=" * 64)
    print("  Ontology v0.3 — Knowledge Graph Seed Script")
    print("=" * 64)

    info = get("/")
    print(f"\n  Connected to: {info['business']} ({info['business_type']})")

    # ── Phase 1: Structured entities with aliases ───────────────────────────

    print("\n── Phase 1: Structured entities (with aliases) ──────────────────")

    entities = [
        {"event_type": "insert", "table": "vehicles", "data": {
            "name": "Van V1", "entity_type": "Vehicle",
            "summary": "White Ford Transit, plate ABC-1234, 1.5 ton capacity",
            "aliases": ["V1", "ABC-1234", "the white van"]}},
        {"event_type": "insert", "table": "vehicles", "data": {
            "name": "Van V2", "entity_type": "Vehicle",
            "summary": "Blue Fiat Ducato, plate DEF-5678, 2 ton capacity",
            "aliases": ["V2", "DEF-5678", "the blue van"]}},
        {"event_type": "insert", "table": "drivers", "data": {
            "name": "João Silva", "entity_type": "Driver",
            "summary": "Senior driver, 5 years experience, CDL holder",
            "aliases": ["João", "Silva"],
            "relationships": [
                {"target": "Van V1", "relation": "OPERATES",
                 "fact": "João is the primary operator of Van V1"}]}},
        {"event_type": "insert", "table": "drivers", "data": {
            "name": "Carlos Oliveira", "entity_type": "Driver",
            "summary": "Junior driver, 1 year experience",
            "aliases": ["Carlos"],
            "relationships": [
                {"target": "Van V2", "relation": "OPERATES",
                 "fact": "Carlos drives Van V2 on weekdays"}]}},
        {"event_type": "insert", "table": "clients", "data": {
            "name": "Padaria Central", "entity_type": "Client",
            "summary": "Downtown bakery, daily bread deliveries, high priority",
            "aliases": ["the bakery", "Central Bakery"]}},
        {"event_type": "insert", "table": "clients", "data": {
            "name": "Restaurante Bom Sabor", "entity_type": "Client",
            "summary": "South zone restaurant, receives supplies 3x/week",
            "aliases": ["Bom Sabor", "the restaurant"]}},
        {"event_type": "insert", "table": "clients", "data": {
            "name": "Mercado Livre Loja SP", "entity_type": "Client",
            "summary": "E-commerce fulfillment center, daily pickups"}},
        {"event_type": "insert", "table": "suppliers", "data": {
            "name": "Distribuidora Norte", "entity_type": "Supplier",
            "summary": "Wholesale food distributor, flour and produce"}},
        {"event_type": "insert", "table": "routes", "data": {
            "name": "North Route", "entity_type": "Route",
            "summary": "Downtown + north neighborhoods, ~45km, 8 stops"}},
        {"event_type": "insert", "table": "routes", "data": {
            "name": "South Route", "entity_type": "Route",
            "summary": "South zone + industrial area, ~60km, 12 stops"}},
    ]

    for evt in entities:
        post("/ingest/event", evt)
        aliases = evt['data'].get('aliases', [])
        alias_str = f" (aliases: {aliases})" if aliases else ""
        print(f"  ✓ {evt['data']['name']}{alias_str}")

    # ── Phase 2: Raw events (LLM extraction + reaction) ────────────────────

    print("\n── Phase 2: Raw events (extraction + reaction) ─────────────────")

    raw_events = [
        {
            "source": "spreadsheet_row",
            "content": (
                "Dispatch Log — Monday 2025-04-21:\n"
                "- Order #1042 (WB-1042): 50kg flour from Distribuidora Norte "
                "to Padaria Central. Van V1, João Silva, North Route. 09:30.\n"
                "- Order #1043 (WB-1043): 30kg produce to Restaurante Bom Sabor. "
                "Van V1, João, North Route. 10:15.\n"
                "- Order #1044 (WB-1044): 200 parcels for Mercado Livre. "
                "Van V2, Carlos Oliveira, South Route. 14:00."
            ),
        },
        {
            "source": "pdf_text",
            "content": (
                "Invoice INV-2025-0891 from Distribuidora Norte:\n"
                "  100kg Flour @ R$4.50 = R$450.00\n"
                "  50kg Produce @ R$8.00 = R$400.00\n"
                "  Total: R$850.00\n"
                "  Delivery: Padaria Central (Order #1045), "
                "Restaurante Bom Sabor (Order #1046)\n"
                "  Scheduled: 2025-04-23, Van V1, North Route"
            ),
        },
    ]

    for raw in raw_events:
        result = post("/ingest/raw", raw)
        print(f"  ✓ Event: {result['event_id']} ({raw['source']})")

    print("\n  Waiting for extraction + reaction pipeline...")
    wait_for_processing()

    # ── Phase 3: External event → reactive reasoning ───────────────────────

    print("\n── Phase 3: External event → automatic reaction ────────────────")
    print("  Simulating: Fleet system reports Van V1 breakdown\n")

    result = post("/ingest/raw", {
        "source": "fleet_webhook",
        "content": (
            "ALERT: Vehicle Van V1 (plate ABC-1234) has broken down on "
            "North Route near km 12. Driver João Silva reports engine failure. "
            "Vehicle immobilized with undelivered cargo. Tow truck ETA 45min."
        ),
        "metadata": {"severity": "critical"},
    })
    print(f"  ✓ Event: {result['event_id']}")
    print("  Waiting for extraction + reaction...")
    wait_for_processing(max_wait=90)

    # ── Phase 4: Inspect reactions and individual actions ───────────────────

    print("\n── Phase 4: Reactions & individual actions ──────────────────────")

    reactions = get("/reactions/pending")
    print(f"\n  Pending reactions: {reactions['count']}")

    for rxn in reactions["reactions"]:
        print(f"\n  ┌── Reaction {rxn['id']}")
        print(f"  │ Trigger: {rxn['trigger_summary'][:80]}")
        print(f"  │ Reasoning: {rxn['reasoning'][:120]}...")
        acts = rxn.get("actions", [])
        print(f"  │ Actions ({len(acts)}):")
        for act in acts:
            print(f"  │   [{act['priority']}] {act['action_type']} → {act['target'][:40]}")
        print(f"  └────────────")

    # Check individual pending actions
    actions = get("/actions/pending")
    print(f"\n  Total pending actions (across all reactions): {actions['count']}")
    for act in actions["actions"][:5]:
        print(f"    [{act['priority']}] {act['action_type']}: {act['target'][:40]} — {act['message'][:60]}...")

    # ── Phase 5: Graph + query ─────────────────────────────────────────────

    print("\n── Phase 5: Graph verification & query ─────────────────────────")

    ents = get("/entities")
    print(f"\n  Total entities: {ents['count']}")
    for e in ents["entities"]:
        aliases = e.get("aliases", [])
        alias_str = f" (aliases: {aliases})" if aliases else ""
        print(f"    [{e['entity_type']}] {e['name']}{alias_str}")

    q = "Van V1 just broke down. What deliveries are at risk and who do I call first?"
    print(f"\n  Q: {q}\n")
    result = post("/query", {"question": q})
    print(f"  Entity: {result.get('entity_found')}")
    print(f"  Subgraph: {result.get('subgraph_size')}")
    print(f"\n  Answer:\n  {'─' * 56}")
    for line in result["answer"].split("\n"):
        print(f"  {line}")
    print(f"  {'─' * 56}")

    # Check schema proposals
    proposals = get("/schema/proposals")
    if proposals["count"] > 0:
        print(f"\n  Schema proposals: {proposals['count']}")
        for p in proposals["proposals"]:
            print(f"    New types: {p['proposed_entity_types']}")
            print(f"    New rels:  {p['proposed_relationship_types']}")

    print("\n✅ Seed complete!")


if __name__ == "__main__":
    main()
