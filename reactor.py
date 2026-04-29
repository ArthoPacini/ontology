"""
Reactive reasoning engine — fully dynamic.
Maps string priority levels from LLM into integers using db mapping.
"""

import logging

from config import SchemaConfig
from extraction import call_openrouter, parse_json_response
from query import serialize_subgraph, extract_keywords
import db

logger = logging.getLogger(__name__)


def build_reaction_prompt(
    schema: SchemaConfig, event_content: str, subgraph_text: str,
) -> list[dict]:
    system = (
        f"You are an operational intelligence system for "
        f"{schema.business_name} ({schema.business_type}).\n\n"
        "An event has been reported. You must:\n"
        "1. Summarize what happened (trigger_summary)\n"
        "2. Analyze the impact by following graph relationships step by step (reasoning)\n"
        "3. Propose concrete actions ONLY from the available actions list below\n\n"
        f"Reasoning guidelines:\n{schema.reaction_hints_str()}\n\n"
        f"Available actions you can propose:\n{schema.actions_str()}\n\n"
        "Respond ONLY with valid JSON (no markdown, no explanation):\n"
        "{\n"
        '  "trigger_summary": "Brief description of what happened",\n'
        '  "reasoning": "Step-by-step impact analysis following graph relationships",\n'
        '  "actions": [\n'
        "    {\n"
        '      "action_type": "action_name_from_available_list",\n'
        '      "priority": "critical|high|medium|low",\n'
        '      "target": "who/what this action is for",\n'
        '      "message": "content of the notification or action details",\n'
        '      "rationale": "why this action is needed"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user = (
        f"KNOWLEDGE GRAPH CONTEXT (subgraph around affected entities):\n"
        f"{subgraph_text}\n\n"
        f"EVENT RECEIVED:\n---\n{event_content}\n---"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def react_to_event(
    schema: SchemaConfig,
    event_id: int,
    content: str,
    extracted_entity_names: list[str] | None = None,
) -> dict | None:
    await db.audit("reaction_started", "events", event_id)
    all_nodes = {}
    all_edges = {}

    search_names = list(extracted_entity_names or [])

    keywords = extract_keywords(content)
    for kw in keywords:
        results = await db.search_entities(kw, limit=3)
        for r in results:
            search_names.append(r["name"])

    seen = set()
    for name in search_names:
        if name in seen:
            continue
        seen.add(name)

        entity = await db.get_entity_by_name(name)
        if not entity:
            entity_resolved = await db.resolve_entity(name)
            if not entity_resolved:
                continue
            entity = entity_resolved

        subgraph = await db.traverse_subgraph(entity["id"], max_depth=2)
        for n in subgraph["nodes"]:
            all_nodes[n["id"]] = n
        for e in subgraph["edges"]:
            all_edges[e["id"]] = e

    if not all_nodes:
        logger.info(f"No graph entities found for event {event_id}, skipping reaction")
        await db.audit("reaction_skipped", "events", event_id, {"reason": "no_entities_found"})
        return None

    combined = {"nodes": list(all_nodes.values()), "edges": list(all_edges.values())}
    subgraph_text = serialize_subgraph(combined)

    messages = build_reaction_prompt(schema, content, subgraph_text)

    try:
        raw_response = await call_openrouter(messages, temperature=0.2, max_tokens=3000)
        result = parse_json_response(raw_response)
    except Exception as e:
        logger.error(f"Reactor LLM call failed for event {event_id}: {e}")
        await db.audit("reaction_failed", "events", event_id, {"error": str(e)})
        return None

    trigger = result.get("trigger_summary", "Unknown event")
    reasoning = result.get("reasoning", "No reasoning provided")
    action_list = result.get("actions", [])

    reaction_id = await db.store_reaction(event_id, trigger, reasoning)

    stored_actions = []
    for act in action_list:
        action_type = act.get("action_type", "unknown")
        # Convert string priority to int
        p_int = db.priority_to_int(act.get("priority", "medium"))
        
        aid = await db.store_action(
            reaction_id=reaction_id,
            action_type=action_type,
            priority=p_int,
            target=act.get("target", ""),
            message=act.get("message", ""),
            rationale=act.get("rationale", ""),
        )
        stored_actions.append({"id": aid, "action_type": action_type})

    await db.audit("reaction_completed", "reactions", reaction_id, {"actions_generated": len(stored_actions)})
    logger.info(
        f"Reaction {reaction_id}: {trigger[:60]} | "
        f"{len(stored_actions)} actions stored individually"
    )

    return {
        "reaction_id": reaction_id,
        "trigger_summary": trigger,
        "reasoning": reasoning,
        "actions_stored": len(stored_actions),
    }
