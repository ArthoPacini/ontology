"""
Schema YAML loader with history tracking.

Single schema.yaml at project root. Swap to change business domain.
Schema history is saved to the database on startup and on every change.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

SCHEMA_PATH = Path(__file__).parent / "schema.yaml"
DB_PATH = Path(__file__).parent / "graph.db"


@dataclass
class RelationshipType:
    name: str
    from_type: str
    to_type: str


@dataclass
class Action:
    name: str
    channel: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaConfig:
    business_name: str
    business_type: str
    entity_types: list[str]
    relationship_types: list[RelationshipType]
    extraction_hints: list[str]
    reaction_hints: list[str]
    actions: list[Action]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    raw_yaml: str = field(default="", repr=False)

    def entity_types_str(self) -> str:
        return ", ".join(self.entity_types)

    def relationship_types_str(self) -> str:
        return "\n".join(
            f"  - {rt.name}: {rt.from_type} -> {rt.to_type}"
            for rt in self.relationship_types
        )

    def extraction_hints_str(self) -> str:
        return "\n".join(f"  - {h}" for h in self.extraction_hints)

    def reaction_hints_str(self) -> str:
        return "\n".join(f"  - {h}" for h in self.reaction_hints)

    def actions_str(self) -> str:
        return "\n".join(
            f"  - {a.name} ({a.channel}): {a.description}"
            for a in self.actions
        )


_cached: SchemaConfig | None = None


def load_schema() -> SchemaConfig:
    global _cached
    if _cached is not None:
        return _cached

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema.yaml not found at {SCHEMA_PATH}")

    raw_yaml = SCHEMA_PATH.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_yaml)

    rels = [
        RelationshipType(name=r["name"], from_type=r["from"], to_type=r["to"])
        for r in raw.get("relationship_types", [])
    ]

    actions = [
        Action(
            name=a["name"],
            channel=a.get("channel", "unknown"),
            description=a.get("description", ""),
            parameters=a.get("parameters", {}),
        )
        for a in raw.get("actions", [])
    ]

    _cached = SchemaConfig(
        business_name=raw.get("business_name", "Unknown Business"),
        business_type=raw.get("business_type", "Generic Business"),
        entity_types=raw.get("entity_types", []),
        relationship_types=rels,
        extraction_hints=raw.get("extraction_hints", []),
        reaction_hints=raw.get("reaction_hints", []),
        actions=actions,
        raw=raw,
        raw_yaml=raw_yaml,
    )
    return _cached


def reload_schema() -> SchemaConfig:
    global _cached
    _cached = None
    return load_schema()


def add_to_schema(new_entity_types: list[str] = None, new_rels: list[dict] = None) -> bool:
    """Append new types to schema.yaml on disk and reload."""
    with open(SCHEMA_PATH, "r") as f:
        raw = yaml.safe_load(f)

    changed = False

    if new_entity_types:
        existing = set(raw.get("entity_types", []))
        for et in new_entity_types:
            if et not in existing:
                raw.setdefault("entity_types", []).append(et)
                changed = True

    if new_rels:
        existing_names = {r["name"] for r in raw.get("relationship_types", [])}
        for nr in new_rels:
            if nr["name"] not in existing_names:
                raw.setdefault("relationship_types", []).append(nr)
                changed = True

    if changed:
        with open(SCHEMA_PATH, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        reload_schema()

    return changed


def restore_schema(yaml_content: str):
    """Overwrite schema.yaml with given content and reload."""
    SCHEMA_PATH.write_text(yaml_content, encoding="utf-8")
    reload_schema()
