"""Lightweight heuristic named entity extraction & relationship graphing.

The intent is to provide a zero-dependency (beyond NetworkX & stdlib regex)
approximation suitable for early pipeline exploration. For production-grade
entity recognition substitute this module with a proper NER model.

Workflow:
    * :func:`extract_entities` collects capitalized phrase candidates.
    * :func:`aggregate_entities` aggregates counts and per-section occurrences.
    * :func:`build_relationship_graph` creates a co-occurrence graph (undirected).
    * :func:`run_ner_and_relations` orchestrates extraction & persistence.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")


def extract_entities(text: str) -> List[str]:
    """Extract naive entity candidates from text.

    Uses a regex for sequences of capitalized words. Filters extremely short
    single-letter tokens and a small stop set.

    Args:
        text: Input text block.

    Returns:
        List of entity candidate strings (may contain duplicates).
    """
    candidates = ENTITY_RE.findall(text)
    stop = {"The", "And", "For", "With", "This", "That", "From"}
    cleaned: List[str] = []
    for c in candidates:
        if any(len(part) == 1 for part in c.split()):
            continue
        if c in stop:
            continue
        cleaned.append(c)
    return cleaned


def aggregate_entities(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate entity occurrences across sections.

    Args:
        sections: List of section dicts each containing 'id' and 'text'.

    Returns:
        Dict with keys:
            counts: Mapping entity -> global frequency
            section_entities: Mapping section_id -> list of entity candidates
    """
    entity_counts: Counter[str] = Counter()
    section_entities: Dict[str, List[str]] = {}
    for s in sections:
        ents = extract_entities(s['text'])
        section_entities[str(s['id'])] = ents
        entity_counts.update(ents)
    return {"counts": dict(entity_counts), "section_entities": section_entities}


def build_relationship_graph(section_entities: Dict[str, List[str]]) -> Dict[str, Any]:
    """Build a simple undirected co-occurrence relationship graph.

    Two entities are connected if they appear together in the same section
    (counted once per section). Edge weight increments per co-occurring section.

    Args:
        section_entities: Mapping section_id -> list of entities in that section.

    Returns:
        Dict with keys 'nodes' and 'edges' suitable for JSON serialization.
    """
    G = nx.Graph()
    for ent_list in section_entities.values():
        for e in ent_list:
            if not G.has_node(e):
                G.add_node(e)
    for ent_list in section_entities.values():
        unique = list(dict.fromkeys(ent_list))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                a, b = unique[i], unique[j]
                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a, b, weight=1)
    nodes = [{"id": n, **G.nodes[n]} for n in G.nodes]
    edges = [{"source": u, "target": v, **G[u][v]} for u, v in G.edges]
    return {"nodes": nodes, "edges": edges}


def run_ner_and_relations(run_dir: str | Path, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run entity extraction and relationship graph construction, persisting output.

    Args:
        run_dir: Directory in which to write `entities.json`.
        sections: List of section dictionaries containing 'id' and 'text'.

    Returns:
        Dictionary with keys 'entities' and 'relations'.
    """
    run_dir = Path(run_dir)
    entities_info = aggregate_entities(sections)
    relations = build_relationship_graph(entities_info['section_entities'])
    data = {"entities": entities_info, "relations": relations}
    (run_dir / "entities.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
