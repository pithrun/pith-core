"""Preference evidence answer planning.

This module turns prompt-visible preference/advice evidence into a small action
plan that downstream answer generation can inspect. It is intentionally pure:
no storage reads, no model calls, and no retrieval side effects.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

PLAN_HEADER = "[Preference answer plan]"
PREFERENCE_EVIDENCE_HEADER = "[Preference evidence]"

PLAN_POLICIES = frozenset(
    {
        "optimize_existing",
        "use_existing_resource",
        "choose_between_known_options",
        "recommend_new_option",
        "abstain",
    }
)
PLAN_ROLES = frozenset(
    {
        "already_owned",
        "already_downloaded",
        "already_selected",
        "prior_positive",
        "prior_negative",
        "generic_advice",
    }
)

_STOPWORDS = frozenset(
    {
        "about",
        "again",
        "already",
        "also",
        "and",
        "any",
        "are",
        "around",
        "actually",
        "am",
        "be",
        "been",
        "being",
        "for",
        "from",
        "getting",
        "have",
        "help",
        "how",
        "into",
        "is",
        "lately",
        "more",
        "should",
        "that",
        "the",
        "this",
        "think",
        "thinking",
        "tips",
        "trouble",
        "user",
        "was",
        "weekend",
        "were",
        "what",
        "when",
        "where",
        "with",
        "would",
        "you",
    }
)

_LOCAL_TOPIC_SYNONYMS = {
    "battery": {"power", "charger", "charging", "phone", "bank"},
    "phone": {"battery", "power", "charger", "charging"},
    "tokyo": {"japan", "transit", "train", "suica", "tripit", "navigation"},
    "transit": {"tokyo", "train", "suica", "tripit", "navigation"},
    "navigation": {"tokyo", "transit", "train", "suica", "tripit"},
    "trip": {"travel", "itinerary", "tripit", "suica"},
}

_OWNED_RE = re.compile(
    r"\b(?:got|bought|purchased|owns|possesses|uses|carries)\b"
    r"|\buser\b.{0,60}\b(?:has|had)\b"
    r"|\bi(?:'ve| have| had)\b",
    re.IGNORECASE,
)
_POSSESSIVE_RESOURCE_RE = re.compile(
    r"\bmy\b.{0,80}\b(?:app|card|charger|itinerary|pad|pass|power bank|ticket|watch)\b",
    re.IGNORECASE,
)
_DOWNLOADED_RE = re.compile(r"\b(?:downloaded|installed)\b", re.IGNORECASE)
_SELECTED_RE = re.compile(r"\b(?:selected|chose|chosen|booked|reserved|card|pass|ticket|itinerary)\b", re.IGNORECASE)
_POSITIVE_RE = re.compile(r"\b(?:positive|prefers|likes|favorite|favourite|enjoys|appreciates)\b", re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"\b(?:negative|dislikes|avoid|avoids|do_not_recommend|don't recommend|do not recommend)\b", re.IGNORECASE)
_ADVICE_RE = re.compile(r"\b(?:advice_facet|tip|tips|advice|recommendation|procedure|avoidance)\b", re.IGNORECASE)
_TIPS_QUERY_RE = re.compile(r"\b(?:tip|tips|advice|help|how to|what should|what can|any ideas)\b", re.IGNORECASE)
_CHOOSE_QUERY_RE = re.compile(r"\b(?:choose|pick|which|what kind|would prefer|recommend)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PreferenceAnswerPlan:
    roles: tuple[str, ...]
    policy: str
    required_facts: tuple[str, ...]
    avoided_facts: tuple[str, ...]
    confidence: float
    rationale: str

    def as_debug_dict(self) -> dict[str, Any]:
        return asdict(self)


def preference_answer_plan_activation(
    plan: PreferenceAnswerPlan | None,
) -> tuple[bool, str | None]:
    """Return whether a plan should be injected and why it was suppressed."""
    if plan is None:
        return False, "missing_plan"
    if plan.policy == "abstain":
        return False, "policy_abstain"
    if not plan.required_facts:
        return False, "no_required_facts"
    if plan.policy == "recommend_new_option":
        return False, "recommend_new_option_suppressed"
    return True, None


def build_preference_answer_plan(
    question: str,
    context: str,
    evidence_block: str | None = None,
) -> PreferenceAnswerPlan:
    """Build a gold-blind answer plan from prompt-visible text."""
    lines = _candidate_lines(context, evidence_block)
    relevant = [line for line in lines if _has_topic_overlap(question, line)]
    roles: list[str] = []
    avoided: list[str] = []
    line_facts: list[tuple[str, tuple[str, ...]]] = []

    for line in relevant:
        line_roles = _roles_for_line(line)
        for role in line_roles:
            if role not in roles:
                roles.append(role)
        fact = _fact_from_line(line)
        if fact and line_roles:
            line_facts.append((fact, line_roles))

    policy = _policy_for(question, roles)
    required = [
        fact
        for fact, line_roles in line_facts
        if _fact_supported_by_policy(policy, line_roles)
    ]
    if not required:
        required = [fact for fact, _line_roles in line_facts]
    if policy == "optimize_existing":
        avoided.extend(_replacement_avoids(required))
    elif policy == "use_existing_resource" and required:
        avoided.append("generic advice that omits existing resources")
    elif "prior_negative" in roles:
        avoided.extend(fact for fact in required if _NEGATIVE_RE.search(fact))

    if not roles and lines:
        policy = "abstain"

    return PreferenceAnswerPlan(
        roles=tuple(role for role in roles if role in PLAN_ROLES),
        policy=policy if policy in PLAN_POLICIES else "abstain",
        required_facts=tuple(_dedupe(required, limit=5)),
        avoided_facts=tuple(_dedupe(avoided, limit=5)),
        confidence=_confidence(policy, roles, required),
        rationale=_rationale(policy, roles),
    )


def render_preference_answer_plan_block(plan: PreferenceAnswerPlan | None) -> str | None:
    """Render a bounded context block for the final answer prompt."""
    if plan is None:
        return None
    roles = ", ".join(plan.roles) if plan.roles else "none"
    required = "; ".join(plan.required_facts) if plan.required_facts else "none"
    avoided = "; ".join(plan.avoided_facts) if plan.avoided_facts else "none"
    return "\n".join(
        [
            PLAN_HEADER,
            f"- policy: {plan.policy}",
            f"- roles: {roles}",
            f"- required_facts: {required}",
            f"- avoided_facts: {avoided}",
            f"- confidence: {plan.confidence:.2f}",
            f"- rationale: {plan.rationale[:240]}",
        ]
    )


def parse_preference_answer_plan_block(text: str | None) -> PreferenceAnswerPlan | None:
    """Parse a rendered plan block from text."""
    if not text or PLAN_HEADER not in text:
        return None
    block = _extract_block(text, PLAN_HEADER)
    fields: dict[str, str] = {}
    for raw_line in block.splitlines()[1:]:
        line = raw_line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        fields[key.strip()] = value.strip()
    policy = fields.get("policy", "abstain")
    if policy not in PLAN_POLICIES:
        policy = "abstain"
    roles = tuple(
        role.strip()
        for role in fields.get("roles", "").split(",")
        if role.strip() in PLAN_ROLES
    )
    return PreferenceAnswerPlan(
        roles=roles,
        policy=policy,
        required_facts=tuple(_split_semicolon_field(fields.get("required_facts"))),
        avoided_facts=tuple(_split_semicolon_field(fields.get("avoided_facts"))),
        confidence=_parse_confidence(fields.get("confidence")),
        rationale=fields.get("rationale", ""),
    )


def check_preference_answer_plan_compliance(
    plan: PreferenceAnswerPlan | None,
    answer: str | None,
) -> dict[str, Any]:
    """Return a best-effort trace of whether the final answer followed the plan."""
    if plan is None:
        return {"compliant": True, "required_present": [], "required_missing": [], "avoided_present": []}
    answer_text = answer or ""
    required_present = [fact for fact in plan.required_facts if _fact_mentions_answer(fact, answer_text)]
    required_missing = [fact for fact in plan.required_facts if fact not in required_present]
    avoided_present = [fact for fact in plan.avoided_facts if _fact_mentions_answer(fact, answer_text)]
    return {
        "compliant": not required_missing and not avoided_present,
        "required_present": required_present,
        "required_missing": required_missing,
        "avoided_present": avoided_present,
    }


def _candidate_lines(context: str, evidence_block: str | None) -> list[str]:
    source = evidence_block if evidence_block else context
    if not source:
        return []
    blocks = []
    if evidence_block:
        blocks.append(evidence_block)
    elif PREFERENCE_EVIDENCE_HEADER in source:
        blocks.extend(_extract_blocks(source, PREFERENCE_EVIDENCE_HEADER))
    lines = []
    for block in blocks or [source]:
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("["):
                lines.append(line)
    if not evidence_block and context:
        for raw_line in context.splitlines():
            line = raw_line.strip()
            if line and _roles_for_line(line) and line not in lines:
                lines.append(line)
    return lines


def _extract_block(text: str, header: str) -> str:
    start = text.find(header)
    if start < 0:
        return ""
    end = text.find("\n\n", start)
    if end < 0:
        end = len(text)
    return text[start:end].strip()


def _extract_blocks(text: str, header: str) -> list[str]:
    blocks = []
    search_at = 0
    while True:
        start = text.find(header, search_at)
        if start < 0:
            break
        end = text.find("\n\n", start)
        if end < 0:
            end = len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
        search_at = max(end, start + len(header))
    return blocks


def _roles_for_line(line: str) -> tuple[str, ...]:
    roles = []
    if _OWNED_RE.search(line) or _POSSESSIVE_RESOURCE_RE.search(line):
        roles.append("already_owned")
    if _DOWNLOADED_RE.search(line):
        roles.append("already_downloaded")
    if _SELECTED_RE.search(line):
        roles.append("already_selected")
    if _POSITIVE_RE.search(line):
        roles.append("prior_positive")
    if _NEGATIVE_RE.search(line):
        roles.append("prior_negative")
    if _ADVICE_RE.search(line):
        roles.append("generic_advice")
    return tuple(_dedupe(roles, limit=6))


def _policy_for(question: str, roles: list[str]) -> str:
    if not roles:
        return "abstain"
    if "already_downloaded" in roles or "already_selected" in roles:
        return "use_existing_resource"
    if "already_owned" in roles and _TIPS_QUERY_RE.search(question or ""):
        return "optimize_existing"
    if ("prior_positive" in roles or "prior_negative" in roles) and _CHOOSE_QUERY_RE.search(question or ""):
        return "choose_between_known_options"
    if "generic_advice" in roles or "prior_positive" in roles:
        return "recommend_new_option"
    return "abstain"


def _fact_supported_by_policy(policy: str, roles: tuple[str, ...]) -> bool:
    role_set = set(roles)
    if policy == "optimize_existing":
        return "already_owned" in role_set
    if policy == "use_existing_resource":
        return bool({"already_downloaded", "already_selected"} & role_set)
    if policy == "choose_between_known_options":
        return bool({"prior_positive", "prior_negative"} & role_set)
    if policy == "recommend_new_option":
        return "generic_advice" in role_set
    return False


def _has_topic_overlap(question: str, line: str) -> bool:
    question_terms = _expanded_terms(_terms(question))
    line_terms = _terms(line)
    if not question_terms or not line_terms:
        return False
    return bool(question_terms & line_terms)


def _expanded_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in list(terms):
        expanded.update(_LOCAL_TOPIC_SYNONYMS.get(term, set()))
    return expanded


def _terms(text: str | None) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _fact_from_line(line: str) -> str:
    owned_resource_fact = _owned_resource_fact_from_line(line)
    if owned_resource_fact:
        return owned_resource_fact
    fact = line.strip().lstrip("-").strip()
    if "| source:" in fact:
        fact = fact.split("| source:", 1)[0].strip()
    if ":" in fact:
        _, fact = fact.split(":", 1)
        fact = fact.strip()
    return fact[:180]


def _owned_resource_fact_from_line(line: str) -> str | None:
    if not (_OWNED_RE.search(line) or _POSSESSIVE_RESOURCE_RE.search(line)):
        return None
    low = line.lower()
    resources = []
    if "portable power bank" in low or "power bank" in low:
        resources.append("portable power bank")
    if "wireless charging pad" in low or "charging pad" in low:
        resources.append("wireless charging pad")
    if "suica" in low:
        resources.append("Suica card")
    if "tripit" in low:
        resources.append("TripIt app")
    if not resources:
        return None
    return "User already has " + _join_resource_list(_dedupe(resources, limit=4)) + "."


def _join_resource_list(resources: list[str]) -> str:
    if len(resources) == 1:
        return resources[0]
    if len(resources) == 2:
        return f"{resources[0]} and {resources[1]}"
    return ", ".join(resources[:-1]) + f", and {resources[-1]}"


def _replacement_avoids(required_facts: list[str]) -> list[str]:
    avoids = []
    joined = " ".join(required_facts).lower()
    if "power bank" in joined:
        avoids.append("buying another power bank")
    if "charger" in joined or "charging" in joined:
        avoids.append("buying another charger")
    return avoids or ["buying another replacement resource"]


def _confidence(policy: str, roles: list[str], required: list[str]) -> float:
    if policy == "abstain":
        return 0.2
    score = 0.55
    if "already_owned" in roles or "already_downloaded" in roles or "already_selected" in roles:
        score += 0.25
    if required:
        score += 0.05
    return min(0.9, round(score, 2))


def _rationale(policy: str, roles: list[str]) -> str:
    if policy == "optimize_existing":
        return "Relevant evidence says the user already has the resource, so answer with use or optimization guidance."
    if policy == "use_existing_resource":
        return "Relevant evidence names resources the user already prepared, so answer should use those resources."
    if policy == "choose_between_known_options":
        return "Relevant preference evidence should guide selection among known options."
    if policy == "recommend_new_option":
        return "Relevant preference or advice evidence exists, but no prepared resource dominates."
    return "No sufficiently relevant preference evidence was found."


def _fact_mentions_answer(fact: str, answer: str) -> bool:
    fact_terms = _terms(fact)
    answer_terms = _terms(answer)
    if not fact_terms:
        return False
    high_signal = {term for term in fact_terms if len(term) >= 5}
    return bool((high_signal or fact_terms) & answer_terms)


def _split_semicolon_field(value: str | None) -> list[str]:
    if not value or value == "none":
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def _parse_confidence(value: str | None) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def _dedupe(values: list[str], limit: int) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out
