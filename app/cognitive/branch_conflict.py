"""Observe-only branch conflict detection for fact-shaped retrieval context."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

_TAG_STRIP_RE = re.compile(r"\[[A-Z][A-Z0-9_-]*(?::[^\]]+)?\]\s*")
_FACT_PATTERNS = (
    re.compile(r"^(.+?) is famous for (.+)$", re.I),
    re.compile(r"^The type of music that (.+?) plays is (.+)$", re.I),
    re.compile(r"^(.+?) is married to (.+)$", re.I),
    re.compile(r"^(.+?) speaks the language of (.+)$", re.I),
    re.compile(r"^(.+?) works in the field of (.+)$", re.I),
    re.compile(r"^(.+?) works in the city of (.+)$", re.I),
    re.compile(r"^(.+?) worked in the city of (.+)$", re.I),
    re.compile(r"^(.+?) is employed by (.+)$", re.I),
    re.compile(r"^The chairperson of (.+?) is (.+)$", re.I),
    re.compile(r"^(.+?) plays the position of (.+)$", re.I),
    re.compile(r"^(.+?) is associated with the sport of (.+)$", re.I),
    re.compile(r"^(.+?) participates in (?:the sport of )?(.+)$", re.I),
    re.compile(r"^(.+?) was created in the country of (.+)$", re.I),
    re.compile(r"^(.+?) was created by (.+)$", re.I),
    re.compile(r"^The capital of (.+?) is (.+)$", re.I),
    re.compile(r"^The official language of (.+?) is (.+)$", re.I),
    re.compile(r"^The head of state of (.+?) is (.+)$", re.I),
    re.compile(r"^The (?:position of the )?head of state of (.+?) is (.+)$", re.I),
    re.compile(r"^The name of the current head of state in (.+?) is (.+)$", re.I),
    re.compile(r"^(.+?) is a citizen of (.+)$", re.I),
    re.compile(r"^(.+?) is affiliated with the religion of (.+)$", re.I),
    re.compile(r"^(.+?) was founded by (.+)$", re.I),
    re.compile(r"^(.+?) was founded in the city of (.+)$", re.I),
    re.compile(r"^The headquarters of (.+?) is located in the city of (.+)$", re.I),
    re.compile(r"^(.+?) is located in the city of (.+)$", re.I),
    re.compile(r"^(.+?) is located in the continent of (.+)$", re.I),
    re.compile(r"^(.+?) was born in the city of (.+)$", re.I),
    re.compile(r"^(.+?) died in the city of (.+)$", re.I),
    re.compile(r"^(.+?) was written in the language of (.+)$", re.I),
    re.compile(r"^The author of (.+?) is (.+)$", re.I),
)

_KEYED_FACT_PATTERNS = (
    ("famous_for", re.compile(r"^(.+?) is famous for (.+)$", re.I)),
    ("music_type", re.compile(r"^The type of music that (.+?) plays is (.+)$", re.I)),
    ("spouse", re.compile(r"^(.+?) is married to (.+)$", re.I)),
    ("language_spoken", re.compile(r"^(.+?) speaks the language of (.+)$", re.I)),
    ("field_of_work", re.compile(r"^(.+?) works in the field of (.+)$", re.I)),
    ("work_city", re.compile(r"^(.+?) works in the city of (.+)$", re.I)),
    ("work_city", re.compile(r"^(.+?) worked in the city of (.+)$", re.I)),
    ("employer", re.compile(r"^(.+?) is employed by (.+)$", re.I)),
    ("chairperson", re.compile(r"^The chairperson of (.+?) is (.+)$", re.I)),
    ("position_played", re.compile(r"^(.+?) plays the position of (.+)$", re.I)),
    ("associated_sport", re.compile(r"^(.+?) is associated with the sport of (.+)$", re.I)),
    ("participates_sport", re.compile(r"^(.+?) participates in (?:the sport of )?(.+)$", re.I)),
    ("created_country", re.compile(r"^(.+?) was created in the country of (.+)$", re.I)),
    ("creator", re.compile(r"^(.+?) was created by (.+)$", re.I)),
    ("capital_of", re.compile(r"^The capital of (.+?) is (.+)$", re.I)),
    ("official_language", re.compile(r"^The official language of (.+?) is (.+)$", re.I)),
    ("head_of_state", re.compile(r"^The head of state of (.+?) is (.+)$", re.I)),
    ("head_of_state_position", re.compile(r"^The (?:position of the )?head of state of (.+?) is (.+)$", re.I)),
    ("head_of_state_name", re.compile(r"^The name of the current head of state in (.+?) is (.+)$", re.I)),
    ("citizenship", re.compile(r"^(.+?) is a citizen of (.+)$", re.I)),
    ("religion", re.compile(r"^(.+?) is affiliated with the religion of (.+)$", re.I)),
    ("founder", re.compile(r"^(.+?) was founded by (.+)$", re.I)),
    ("founded_city", re.compile(r"^(.+?) was founded in the city of (.+)$", re.I)),
    ("headquarters_city", re.compile(r"^The headquarters of (.+?) is located in the city of (.+)$", re.I)),
    ("located_city", re.compile(r"^(.+?) is located in the city of (.+)$", re.I)),
    ("located_continent", re.compile(r"^(.+?) is located in the continent of (.+)$", re.I)),
    ("birth_city", re.compile(r"^(.+?) was born in the city of (.+)$", re.I)),
    ("death_city", re.compile(r"^(.+?) died in the city of (.+)$", re.I)),
    ("written_language", re.compile(r"^(.+?) was written in the language of (.+)$", re.I)),
    ("author", re.compile(r"^The author of (.+?) is (.+)$", re.I)),
)

_PREDICATE_SQL_MARKERS = {
    "famous_for": " is famous for ",
    "music_type": "the type of music that ",
    "spouse": " is married to ",
    "language_spoken": " speaks the language of ",
    "field_of_work": " works in the field of ",
    "work_city": " work",
    "employer": " is employed by ",
    "chairperson": "the chairperson of ",
    "position_played": " plays the position of ",
    "associated_sport": " associated with the sport of ",
    "participates_sport": " participates in ",
    "created_country": " was created in the country of ",
    "creator": " was created by ",
    "capital_of": "the capital of ",
    "official_language": "the official language of ",
    "head_of_state": "the head of state of ",
    "head_of_state_position": "head of state of ",
    "head_of_state_name": "current head of state in ",
    "citizenship": " is a citizen of ",
    "religion": " affiliated with the religion of ",
    "founder": " was founded by ",
    "founded_city": " was founded in the city of ",
    "headquarters_city": "the headquarters of ",
    "located_city": " located in the city of ",
    "located_continent": " located in the continent of ",
    "birth_city": " was born in the city of ",
    "death_city": " died in the city of ",
    "written_language": " was written in the language of ",
    "author": "the author of ",
}


@dataclass(frozen=True)
class BranchPath:
    concept_ids: tuple[str, ...]
    facts: tuple[str, ...]
    subjects: tuple[str, ...]
    terminal: str


@dataclass(frozen=True)
class BranchConflictReport:
    diagnostic_only: bool
    gold_used: bool
    classification: str
    branch_count: int
    terminal_count: int
    seed_subjects: tuple[str, ...]
    terminal_answers: tuple[str, ...]
    branches: tuple[BranchPath, ...]

    def to_log_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["branches"] = [
            {
                "concept_ids": list(branch.concept_ids),
                "terminal": branch.terminal,
                "hop_count": len(branch.facts),
            }
            for branch in self.branches
        ]
        return payload


@dataclass(frozen=True)
class TerminalConflictBranch:
    concept_ids: tuple[str, ...]
    terminal_subject: str
    terminal_predicate: str
    terminal_object: str
    terminal_alternative_objects: tuple[str, ...]
    retrieved_alternative_count: int
    lookup_alternative_count: int
    terminal_conflict: bool


@dataclass(frozen=True)
class TerminalConflictReport:
    diagnostic_only: bool
    gold_used: bool
    classification: str
    branch_count: int
    terminal_key_count: int
    terminal_conflict_count: int
    lookup_count: int
    seed_subjects: tuple[str, ...]
    branches: tuple[TerminalConflictBranch, ...]
    elapsed_ms: float

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "mab.terminal_conflict_trace.v1",
            "diagnostic_only": self.diagnostic_only,
            "gold_used": self.gold_used,
            "classification": self.classification,
            "branch_count": self.branch_count,
            "terminal_key_count": self.terminal_key_count,
            "terminal_conflict_count": self.terminal_conflict_count,
            "lookup_count": self.lookup_count,
            "seed_subjects": list(self.seed_subjects),
            "cost_latency": {
                "added_llm_calls": 0,
                "lookup_count": self.lookup_count,
                "elapsed_ms": self.elapsed_ms,
            },
            "branches": [
                {
                    "concept_ids": list(branch.concept_ids),
                    "terminal_subject": branch.terminal_subject,
                    "terminal_predicate": branch.terminal_predicate,
                    "terminal_object": branch.terminal_object,
                    "terminal_alternative_count": len(branch.terminal_alternative_objects),
                    "terminal_alternative_objects": list(branch.terminal_alternative_objects),
                    "retrieved_alternative_count": branch.retrieved_alternative_count,
                    "lookup_alternative_count": branch.lookup_alternative_count,
                    "terminal_conflict": branch.terminal_conflict,
                }
                for branch in self.branches
            ],
        }


@dataclass(frozen=True)
class _Edge:
    concept_id: str
    fact: str
    subject: str
    obj: str


@dataclass(frozen=True)
class _KeyedEdge:
    concept_id: str
    fact: str
    subject: str
    predicate: str
    obj: str


def _value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _strip_tags(text: str) -> str:
    return _TAG_STRIP_RE.sub("", str(text or "")).strip()


def _norm_entity(text: str) -> str:
    text = _strip_tags(text).lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'“”]+|[\"'“”.,;:!?]+$", "", text.strip())
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    return text


def _question_contains_entity(question_norm: str, entity_norm: str) -> bool:
    if not entity_norm or len(entity_norm) < 3:
        return False
    if entity_norm in {"none", "unknown", "i don't know", "n/a"}:
        return False
    escaped = re.escape(entity_norm)
    return re.search(rf"(?<![\w']){escaped}(?:'s)?(?![\w'])", question_norm) is not None


def _parse_fact(summary: str) -> tuple[str, str] | None:
    text = _strip_tags(summary).strip().rstrip(".")
    for pattern in _FACT_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        subject = _norm_entity(match.group(1))
        obj = _norm_entity(match.group(2))
        if subject and obj and subject != obj:
            return subject, obj
    return None


def _parse_fact_keyed(summary: str) -> tuple[str, str, str] | None:
    text = _strip_tags(summary).strip().rstrip(".")
    for predicate, pattern in _KEYED_FACT_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        subject = _norm_entity(match.group(1))
        obj = _norm_entity(match.group(2))
        if subject and obj and subject != obj:
            return subject, predicate, obj
    return None


def predicate_sql_marker(predicate: str) -> str | None:
    """Return a coarse SQL marker for bounded same-key diagnostic lookups."""
    return _PREDICATE_SQL_MARKERS.get(predicate)


def _classify(branches: list[BranchPath]) -> str:
    if not branches:
        return "NO_FACT_BRANCHES"
    terminals = {_norm_entity(branch.terminal) for branch in branches if branch.terminal}
    if len(branches) == 1:
        return "SINGLE_SUPPORTED_BRANCH"
    if len(terminals) <= 1:
        return "MULTI_BRANCH_SHARED_TERMINAL"
    return "MULTI_BRANCH_TERMINAL_CONFLICT"


def analyze_branch_conflicts(
    question: str,
    concepts: list[Any],
    *,
    max_hops: int = 5,
) -> BranchConflictReport:
    """Detect exact-handoff branch conflicts without changing answer behavior.

    The detector uses only question text and retrieved concept summaries. It has no
    gold-answer input and is intended for diagnostic/product-awareness telemetry.
    """
    question_norm = _norm_entity(question)
    edges_by_subject: dict[str, list[_Edge]] = {}
    seeds: list[str] = []

    for index, concept in enumerate(concepts or []):
        summary = str(_value(concept, "summary", "") or "")
        parsed = _parse_fact(summary)
        if not parsed:
            continue
        subject, obj = parsed
        concept_id = str(_value(concept, "concept_id", "") or _value(concept, "id", "") or f"idx:{index}")
        edge = _Edge(concept_id=concept_id, fact=_strip_tags(summary).rstrip("."), subject=subject, obj=obj)
        edges_by_subject.setdefault(subject, []).append(edge)
        if subject not in seeds and _question_contains_entity(question_norm, subject):
            seeds.append(subject)

    branches: list[BranchPath] = []

    def walk(entity: str, path: list[_Edge], visited: set[str], depth: int) -> None:
        outgoing = edges_by_subject.get(entity, [])
        if not outgoing or depth >= max_hops:
            if path:
                branches.append(
                    BranchPath(
                        concept_ids=tuple(edge.concept_id for edge in path),
                        facts=tuple(edge.fact for edge in path),
                        subjects=tuple(edge.subject for edge in path),
                        terminal=path[-1].obj,
                    )
                )
            return

        advanced = False
        for edge in outgoing:
            if edge.obj in visited:
                continue
            advanced = True
            walk(edge.obj, [*path, edge], {*visited, edge.obj}, depth + 1)
        if not advanced and path:
            branches.append(
                BranchPath(
                    concept_ids=tuple(edge.concept_id for edge in path),
                    facts=tuple(edge.fact for edge in path),
                    subjects=tuple(edge.subject for edge in path),
                    terminal=path[-1].obj,
                )
            )

    for seed in seeds:
        walk(seed, [], {seed}, 0)

    terminal_answers = tuple(
        dict.fromkeys(branch.terminal for branch in branches if branch.terminal)
    )
    return BranchConflictReport(
        diagnostic_only=True,
        gold_used=False,
        classification=_classify(branches),
        branch_count=len(branches),
        terminal_count=len(terminal_answers),
        seed_subjects=tuple(seeds),
        terminal_answers=terminal_answers,
        branches=tuple(branches),
    )


def _terminal_classification(branches: list[TerminalConflictBranch], lookup_count: int) -> str:
    if not branches:
        return "NO_FACT_BRANCHES"
    if not any(branch.terminal_subject and branch.terminal_predicate for branch in branches):
        return "NO_TERMINAL_KEYS"
    conflict_branches = [branch for branch in branches if branch.terminal_conflict]
    if not conflict_branches:
        return "NO_TERMINAL_CONFLICT"
    if any(branch.lookup_alternative_count for branch in conflict_branches):
        return "LOOKUP_TERMINAL_CONFLICT"
    if lookup_count:
        return "RETRIEVED_TERMINAL_CONFLICT"
    return "RETRIEVED_TERMINAL_CONFLICT"


def _answer_matches_value(answer: str, value: str) -> bool:
    answer_norm = _norm_entity(answer)
    value_norm = _norm_entity(value)
    if not answer_norm or not value_norm:
        return False
    if answer_norm == value_norm:
        return True
    if len(answer_norm) < 5 or len(value_norm) < 5:
        return False
    return answer_norm in value_norm or value_norm in answer_norm


def _surface_match_state(
    *,
    answer: str,
    terminal_object_matches: list[int],
    terminal_object_conflict_matches: list[int],
    alternative_matches: list[int],
    alternative_conflict_matches: list[int],
) -> str:
    if not _norm_entity(answer):
        return "no_answer_surface"
    if terminal_object_conflict_matches:
        return "answer_matches_conflicted_terminal_object"
    if terminal_object_matches:
        return "answer_matches_terminal_object"
    if alternative_conflict_matches:
        return "answer_matches_conflicting_alternative"
    if alternative_matches:
        return "answer_matches_terminal_alternative"
    return "no_terminal_answer_match"


def build_terminal_answer_surface_binding(
    terminal_trace: dict[str, Any] | None,
    answer_surfaces: dict[str, str | None],
) -> dict[str, Any]:
    """Bind emitted answer surfaces to terminal-conflict branches.

    This is observe-only diagnostic telemetry. It accepts no benchmark gold,
    row id, expected answer, or expected source reference. It only compares
    already-emitted answer strings against the already-built terminal trace.
    """
    branches = list((terminal_trace or {}).get("branches") or [])
    surfaces: list[dict[str, Any]] = []
    for surface_name, answer in (answer_surfaces or {}).items():
        answer_text = str(answer or "").strip()
        if not answer_text:
            continue
        terminal_object_matches: list[int] = []
        terminal_object_conflict_matches: list[int] = []
        alternative_matches: list[int] = []
        alternative_conflict_matches: list[int] = []
        matched_terminal_keys: list[str] = []
        matched_alternative_objects: list[str] = []

        for index, branch in enumerate(branches):
            terminal_object = str(branch.get("terminal_object") or "")
            terminal_subject = str(branch.get("terminal_subject") or "")
            terminal_predicate = str(branch.get("terminal_predicate") or "")
            key = f"{terminal_subject}|{terminal_predicate}"
            is_conflict = bool(branch.get("terminal_conflict"))
            if _answer_matches_value(answer_text, terminal_object):
                terminal_object_matches.append(index)
                matched_terminal_keys.append(key)
                if is_conflict:
                    terminal_object_conflict_matches.append(index)
            for alternative in branch.get("terminal_alternative_objects") or []:
                alternative_text = str(alternative or "")
                if not _answer_matches_value(answer_text, alternative_text):
                    continue
                alternative_matches.append(index)
                matched_alternative_objects.append(alternative_text)
                if is_conflict:
                    alternative_conflict_matches.append(index)

        surfaces.append(
            {
                "surface": surface_name,
                "answer": answer_text,
                "normalized_answer": _norm_entity(answer_text),
                "match_state": _surface_match_state(
                    answer=answer_text,
                    terminal_object_matches=terminal_object_matches,
                    terminal_object_conflict_matches=terminal_object_conflict_matches,
                    alternative_matches=alternative_matches,
                    alternative_conflict_matches=alternative_conflict_matches,
                ),
                "terminal_object_branch_indexes": terminal_object_matches,
                "terminal_object_conflict_branch_indexes": terminal_object_conflict_matches,
                "terminal_alternative_branch_indexes": sorted(set(alternative_matches)),
                "terminal_alternative_conflict_branch_indexes": sorted(
                    set(alternative_conflict_matches)
                ),
                "matched_terminal_keys": sorted(set(matched_terminal_keys)),
                "matched_alternative_objects": sorted(set(matched_alternative_objects)),
            }
        )

    return {
        "schema_version": "mab.answer_surface_binding.v1",
        "diagnostic_only": True,
        "gold_used": False,
        "surface_count": len(surfaces),
        "terminal_branch_count": len(branches),
        "surfaces": surfaces,
        "summary": {
            "any_terminal_object_match": any(
                surface["terminal_object_branch_indexes"] for surface in surfaces
            ),
            "any_conflicted_terminal_object_match": any(
                surface["terminal_object_conflict_branch_indexes"]
                for surface in surfaces
            ),
            "any_terminal_alternative_match": any(
                surface["terminal_alternative_branch_indexes"] for surface in surfaces
            ),
            "any_conflicted_terminal_alternative_match": any(
                surface["terminal_alternative_conflict_branch_indexes"]
                for surface in surfaces
            ),
            "match_states": sorted({surface["match_state"] for surface in surfaces}),
        },
        "cost_latency": {
            "added_llm_calls": 0,
        },
    }


def analyze_terminal_conflicts(
    question: str,
    concepts: list[Any],
    *,
    same_key_lookup: Callable[[str, str], list[Any]] | None = None,
    max_hops: int = 5,
    max_alternatives_per_key: int = 5,
) -> TerminalConflictReport:
    """Trace same-subject/same-predicate terminal conflicts without answer actuation.

    The detector uses only question text, retrieved/activated concept summaries,
    and an optional caller-injected same-key lookup. It accepts no benchmark gold,
    row id, expected source reference, or answer-selection input.
    """
    started = time.perf_counter()
    question_norm = _norm_entity(question)
    edges_by_subject: dict[str, list[_KeyedEdge]] = {}
    edges_by_key: dict[tuple[str, str], list[_KeyedEdge]] = {}
    seeds: list[str] = []

    for index, concept in enumerate(concepts or []):
        summary = str(_value(concept, "summary", "") or "")
        parsed = _parse_fact_keyed(summary)
        if not parsed:
            continue
        subject, predicate, obj = parsed
        concept_id = str(_value(concept, "concept_id", "") or _value(concept, "id", "") or f"idx:{index}")
        edge = _KeyedEdge(
            concept_id=concept_id,
            fact=_strip_tags(summary).rstrip("."),
            subject=subject,
            predicate=predicate,
            obj=obj,
        )
        edges_by_subject.setdefault(subject, []).append(edge)
        edges_by_key.setdefault((subject, predicate), []).append(edge)
        if subject not in seeds and _question_contains_entity(question_norm, subject):
            seeds.append(subject)

    terminal_paths: list[list[_KeyedEdge]] = []

    def walk(entity: str, path: list[_KeyedEdge], visited: set[str], depth: int) -> None:
        outgoing = edges_by_subject.get(entity, [])
        if not outgoing or depth >= max_hops:
            if path:
                terminal_paths.append(path)
            return
        advanced = False
        for edge in outgoing:
            if edge.obj in visited:
                continue
            advanced = True
            walk(edge.obj, [*path, edge], {*visited, edge.obj}, depth + 1)
        if not advanced and path:
            terminal_paths.append(path)

    for seed in seeds:
        walk(seed, [], {seed}, 0)

    lookup_cache: dict[tuple[str, str], list[_KeyedEdge]] = {}
    lookup_count = 0

    def lookup_edges(subject: str, predicate: str) -> list[_KeyedEdge]:
        nonlocal lookup_count
        key = (subject, predicate)
        if key in lookup_cache:
            return lookup_cache[key]
        lookup_cache[key] = []
        if same_key_lookup is None:
            return []
        lookup_count += 1
        raw_rows = same_key_lookup(subject, predicate) or []
        parsed_edges: list[_KeyedEdge] = []
        for index, row in enumerate(raw_rows[: max(0, max_alternatives_per_key * 3)]):
            summary = str(_value(row, "summary", "") or "")
            parsed = _parse_fact_keyed(summary)
            if not parsed:
                continue
            row_subject, row_predicate, row_obj = parsed
            if row_subject != subject or row_predicate != predicate:
                continue
            concept_id = str(_value(row, "concept_id", "") or _value(row, "id", "") or f"lookup:{index}")
            parsed_edges.append(
                _KeyedEdge(
                    concept_id=concept_id,
                    fact=_strip_tags(summary).rstrip("."),
                    subject=row_subject,
                    predicate=row_predicate,
                    obj=row_obj,
                )
            )
            if len(parsed_edges) >= max_alternatives_per_key:
                break
        lookup_cache[key] = parsed_edges
        return parsed_edges

    branches: list[TerminalConflictBranch] = []
    for path in terminal_paths:
        terminal = path[-1]
        key = (terminal.subject, terminal.predicate)
        retrieved_edges = edges_by_key.get(key, [])
        lookup_candidates = lookup_edges(*key)
        retrieved_ids = {edge.concept_id for edge in retrieved_edges}
        lookup_unique = [edge for edge in lookup_candidates if edge.concept_id not in retrieved_ids]
        objects = tuple(
            dict.fromkeys(
                edge.obj
                for edge in [*retrieved_edges, *lookup_unique]
                if edge.obj
            )
        )[:max_alternatives_per_key]
        branches.append(
            TerminalConflictBranch(
                concept_ids=tuple(edge.concept_id for edge in path),
                terminal_subject=terminal.subject,
                terminal_predicate=terminal.predicate,
                terminal_object=terminal.obj,
                terminal_alternative_objects=objects,
                retrieved_alternative_count=len({edge.obj for edge in retrieved_edges}),
                lookup_alternative_count=len({edge.obj for edge in lookup_unique}),
                terminal_conflict=len(set(objects)) > 1,
            )
        )

    terminal_keys = {
        (branch.terminal_subject, branch.terminal_predicate)
        for branch in branches
        if branch.terminal_subject and branch.terminal_predicate
    }
    conflict_count = len(
        {
            (branch.terminal_subject, branch.terminal_predicate)
            for branch in branches
            if branch.terminal_conflict
        }
    )
    return TerminalConflictReport(
        diagnostic_only=True,
        gold_used=False,
        classification=_terminal_classification(branches, lookup_count),
        branch_count=len(branches),
        terminal_key_count=len(terminal_keys),
        terminal_conflict_count=conflict_count,
        lookup_count=lookup_count,
        seed_subjects=tuple(seeds),
        branches=tuple(branches),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
    )
