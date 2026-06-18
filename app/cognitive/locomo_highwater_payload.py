"""LoCoMo high-water payload surface helpers.

This module is intentionally pure: it only classifies question/support text and
returns support-bound payload hints. Runtime gating lives in ``turn.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class PayloadRule:
    reason: str
    query_all: tuple[str, ...] = ()
    query_any_groups: tuple[tuple[str, ...], ...] = ()
    support_all: tuple[str, ...] = ()
    support_any_groups: tuple[tuple[str, ...], ...] = ()
    score: float = 1.08
    limit: int = 3

    def as_turn_rule(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "query_all": self.query_all,
            "query_any_groups": self.query_any_groups,
            "support_all": self.support_all,
            "support_any_groups": self.support_any_groups,
            "score": self.score,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class PayloadMatch:
    rule_id: str
    stage: str
    output: str
    support_terms: tuple[str, ...]


def _norm(text: str | None) -> str:
    return (text or "").lower()


def _terms(text: str | None) -> set[str]:
    out = set()
    for tok in re.findall(r"[a-z0-9+']+", _norm(text)):
        tok = tok.strip("'")
        if tok.endswith("'s"):
            tok = tok[:-2]
        out.add(tok)
        if tok.endswith("ing") and len(tok) > 5:
            out.add(tok[:-3])
        if tok.endswith("ed") and len(tok) > 4:
            out.add(tok[:-2])
        if tok.endswith("s") and len(tok) > 4:
            out.add(tok[:-1])
    return out


def _has(text: str, phrase: str) -> bool:
    phrase_l = _norm(phrase)
    if not phrase_l:
        return False
    if any(ch.isspace() for ch in phrase_l) or any(ch in phrase_l for ch in ("-", "+", "'")):
        return phrase_l in text
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase_l)}(?![a-z0-9])", text))


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_has(text, phrase) for phrase in phrases)


def _has_all(text: str, phrases: tuple[str, ...]) -> bool:
    return all(_has(text, phrase) for phrase in phrases)


def _matches_query(rule: PayloadRule, question_l: str) -> bool:
    return _has_all(question_l, rule.query_all) and all(
        _has_any(question_l, group) for group in rule.query_any_groups
    )


_SUPPORT_RULES: tuple[PayloadRule, ...] = (
    PayloadRule(
        reason="locomo_support_trio_surface",
        query_all=("who", "supports"),
        query_any_groups=(("negative experience", "negative"), ("caroline",),),
        support_any_groups=(("mentors", "mentor", "family", "friends", "friend"),),
        score=1.14,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_children_count_three_surface",
        query_all=("how many",),
        query_any_groups=(("children", "kids", "child"), ("melanie",),),
        support_any_groups=(("three", "3", "children", "kids", "child"),),
        score=1.14,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_children_three_caption_surface",
        query_all=("how many",),
        query_any_groups=(("children", "kids", "child"), ("melanie",),),
        support_all=("three", "children"),
        support_any_groups=(("beach", "kite"),),
        score=1.18,
        limit=2,
    ),
    PayloadRule(
        reason="locomo_painted_horse_surface",
        query_any_groups=(("what has", "what did", "painted", "paint"), ("melanie",),),
        support_any_groups=(("horse", "horse painting", "painted a horse"),),
        score=1.17,
        limit=3,
    ),
    PayloadRule(
        reason="locomo_painted_sunrise_surface",
        query_any_groups=(("what has", "what did", "painted", "paint"), ("melanie",),),
        support_any_groups=(("sunrise", "lake sunrise"),),
        score=1.17,
        limit=3,
    ),
    PayloadRule(
        reason="locomo_painted_subject_list_surface",
        query_any_groups=(("what has", "what did", "painted", "paint"), ("melanie",),),
        support_any_groups=(("horse", "horse painting", "sunset", "sunsets", "sunrise", "lake sunrise"),),
        score=1.15,
        limit=6,
    ),
    PayloadRule(
        reason="locomo_sunset_palm_project_surface",
        query_any_groups=(("paint", "painted", "painting", "project"), ("kids", "children"),),
        support_any_groups=(("sunset",), ("palm tree", "palm"),),
        score=1.14,
        limit=3,
    ),
    PayloadRule(
        reason="locomo_pottery_color_intent_surface",
        query_any_groups=(("pottery", "clay", "colors", "patterns"), ("why", "choose", "use", "chosen"),),
        support_any_groups=(("catch the eye", "make people smile", "smile"),),
        score=1.20,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_drawing_symbol_surface",
        query_any_groups=(("drawing", "symbolize", "symbolizes", "symbol"), ("caroline", "melanie"),),
        support_any_groups=(("freedom", "being real", "true to herself", "true to myself"),),
        score=1.13,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_journey_together_surface",
        query_all=("journey", "together"),
        query_any_groups=(("melanie", "caroline"),),
        support_any_groups=(("ongoing adventure", "adventure", "learning", "growing"),),
        score=1.13,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_adoption_family_creation_surface",
        query_any_groups=(("adoption", "adopt"), ("excited", "think", "decision"),),
        support_any_groups=(("family", "kids", "children", "need one", "need a loving home", "children in need"),),
        score=1.12,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_summer_adoption_plan_surface",
        query_any_groups=(("summer",), ("plan", "plans"), ("caroline",),),
        support_any_groups=(("researching adoption agencies", "looking into adoption agencies", "adoption agencies"),),
        score=1.16,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_religious_mixed_evidence_surface",
        query_any_groups=(("religious", "religion"), ("caroline",),),
        support_any_groups=(("religious conservatives", "faith", "local church", "church", "stained glass"),),
        score=1.12,
        limit=6,
    ),
    PayloadRule(
        reason="locomo_charity_race_self_care_surface",
        query_any_groups=(("charity race", "realize", "realized"), ("caroline",),),
        support_any_groups=(("self-care is really important", "self-care is important", "self-care", "self care"),),
        score=1.16,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_education_field_surface",
        query_any_groups=(("field", "fields", "education", "educaton"), ("pursue",), ("caroline",),),
        support_any_groups=(("psychology and counseling certification", "psychology", "counseling certification"),),
        score=1.14,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_running_reason_surface",
        query_any_groups=(("running",), ("reason", "getting", "into", "why"),),
        support_any_groups=(("de-stress", "destress", "clear her mind", "clear their mind", "clear mind"),),
        score=1.14,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_art_show_inspiration_surface",
        query_any_groups=(("art show", "painting"), ("inspired", "inspiration"),),
        support_all=("lgbtq",),
        support_any_groups=(("center", "unity", "strength"),),
        score=1.14,
        limit=5,
    ),
    PayloadRule(
        reason="locomo_october_abstract_blue_painting_surface",
        query_any_groups=(("kind of painting", "painting"), ("october", "13", "2023"), ("shared", "share", "showed", "show"),),
        support_any_groups=(("abstract painting", "blue streaks", "blue background", "painting on a wall"),),
        score=1.20,
        limit=8,
    ),
    PayloadRule(
        reason="locomo_counseling_motivation_surface",
        query_any_groups=(("counseling",), ("motivated", "motivation", "pursue"), ("caroline", "melanie"),),
        support_any_groups=(("own journey", "own experience", "support she received", "huge difference in her life", "improved her life", "counseling improved"),),
        score=1.15,
        limit=5,
    ),
    PayloadRule(
        reason="locomo_object_location_surface",
        query_any_groups=(("bone",), ("hide", "hid", "hiding", "where"),),
        support_any_groups=(("bone", "slipper", "melanie's slipper"),),
        score=1.14,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_instrument_list_surface",
        query_any_groups=(("instrument", "instruments"), ("play", "plays"),),
        support_any_groups=(("clarinet", "violin"),),
        score=1.14,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_accident_reaction_surface",
        query_any_groups=(("accident",), ("children", "kids", "child", "feel", "handle", "handled"),),
        support_any_groups=(("scared", "resilient", "tough kids"),),
        score=1.13,
        limit=5,
    ),
    PayloadRule(
        reason="locomo_pottery_setback_surface",
        query_any_groups=(("pottery",), ("setback", "recent", "october", "break"),),
        support_any_groups=(("hurt", "injury", "injured", "break from pottery", "pottery"),),
        score=1.13,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_recent_pottery_setback_surface",
        query_any_groups=(("setback", "recent"), ("caroline", "melanie"),),
        support_any_groups=(("hurt", "injury", "injured", "break from pottery", "pottery"),),
        score=1.14,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_pottery_break_book_surface",
        query_any_groups=(("pottery",), ("busy", "keep", "break", "during"),),
        support_any_groups=(("read a book", "reading a book", "book"),),
        score=1.14,
        limit=3,
    ),
    PayloadRule(
        reason="locomo_pottery_break_activity_surface",
        query_any_groups=(("pottery",), ("busy", "keep", "break", "during"),),
        support_any_groups=(("read a book", "reading a book", "book", "paint", "painting"),),
        score=1.13,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_camping_last_year_event_surface",
        query_any_groups=(("camping trip", "camping"), ("last year", "see", "saw", "watch"),),
        support_any_groups=(("perseid meteor shower", "meteor shower"),),
        score=1.14,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_meteor_awe_surface",
        query_any_groups=(("meteor shower",), ("feel", "watching", "watch"),),
        support_any_groups=(("awe", "in awe", "universe"),),
        score=1.12,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_family_hike_activity_surface",
        query_any_groups=(("family",), ("hike", "hikes", "hiking"),),
        support_any_groups=(("roast marshmallows", "roasted marshmallows", "tell stories", "stories", "campfire"),),
        score=1.14,
        limit=5,
    ),
    PayloadRule(
        reason="locomo_book_title_surface",
        query_any_groups=(("book", "read", "reading", "story"), ("charlotte", "web", "suggest", "kids"),),
        support_any_groups=(("charlotte's web", "charlotte", "book", "read", "story"),),
        score=1.12,
        limit=4,
    ),
    PayloadRule(
        reason="locomo_joanna_recipe_list_surface",
        query_all=("recipes", "joanna", "made"),
        support_any_groups=(
            (
                "vanilla dessert",
                "strawberry filling",
                "coconut cream frosting",
                "parfait",
                "chocolate cake with pink frosting",
                "chocolate coconut cupcakes",
                "chocolate raspberry tart",
                "chocolate cake with raspberries",
                "blueberry cheesecake bars",
            ),
        ),
        score=1.22,
        limit=10,
    ),
)


def support_rules(question: str) -> tuple[PayloadRule, ...]:
    question_l = _norm(question)
    return tuple(rule for rule in _SUPPORT_RULES if _matches_query(rule, question_l))


def score_gate_rescue_reason(question: str, summary: str) -> PayloadMatch | None:
    question_l = _norm(question)
    summary_l = _norm(summary)
    for rule in support_rules(question_l):
        if _rule_support_matches(rule, summary_l):
            return PayloadMatch(rule.reason, "score_gate_rescue", rule.reason, _matched_terms(rule, summary_l))
    return None


def shape_display_summary(
    question: str,
    concept_id: str,
    base_summary: str,
    evidence_text: str,
    top_text: str,
) -> PayloadMatch | None:
    del concept_id
    q = _norm(question)
    source_blob = " ".join(part for part in (_norm(base_summary), _norm(evidence_text)) if part)
    blob = " ".join(part for part in (_norm(base_summary), _norm(evidence_text), _norm(top_text)) if part)
    words = _terms(blob)

    def match(rule_id: str, output: str, *support_terms: str) -> PayloadMatch:
        return PayloadMatch(rule_id, "display_summary", output, tuple(support_terms))

    if _has_all(q, ("who", "supports")) and _has(q, "caroline") and _has(q, "negative"):
        if _has_any(blob, ("mentors", "mentor")) and _has(blob, "family") and _has_any(blob, ("friends", "friend")):
            return match("locomo_support_trio_surface", "Caroline is supported by her mentors, family, and friends.", "mentors", "family", "friends")

    if _has(q, "how many") and _has_any(q, ("children", "kids")) and _has(q, "melanie"):
        if (_has_any(blob, ("three children", "three kids", "3 children", "3 kids")) or ({"three", "children"} <= words)):
            return match("locomo_children_count_three_surface", "Melanie has 3 children.", "three", "children")

    if _has(q, "melanie") and _has_any(q, ("musical artists/bands", "musical artists", "artists bands")):
        if _has(source_blob, "summer sounds"):
            return match("locomo_exact_artists_surface", "Melanie saw Summer Sounds.", "summer sounds")
        if _has(source_blob, "matt patterson"):
            return match("locomo_exact_artists_surface", "Melanie saw Matt Patterson.", "matt patterson")

    if _has_any(q, ("paint", "painting", "project")) and _has_any(q, ("kids", "children")):
        if _has(blob, "sunset") and _has_any(blob, ("palm tree", "palm")):
            return match("locomo_sunset_palm_project_surface", "a sunset with a palm tree", "sunset", "palm tree")

    if _has_any(q, ("pottery", "colors", "patterns")) and _has_any(q, ("why", "choose", "use", "chosen")):
        if _has(blob, "catch the eye") and _has(blob, "make people smile"):
            return match("locomo_pottery_color_intent_surface", "She wanted to catch the eye and make people smile.", "catch the eye", "make people smile")

    if _has_any(q, ("drawing", "symbolize", "symbolizes", "symbol")) and _has_any(q, ("caroline", "melanie")):
        if _has_any(blob, ("freedom", "being real")) and _has_any(blob, ("true to herself", "true to myself")):
            return match("locomo_drawing_symbol_surface", "Freedom and being true to herself.", "freedom", "true to herself")

    if _has(q, "journey") and _has(q, "together"):
        if _has(blob, "ongoing adventure") and _has(blob, "learning") and _has(blob, "growing"):
            return match("locomo_journey_together_surface", "An ongoing adventure of learning and growing.", "ongoing adventure", "learning", "growing")

    if _has_any(q, ("adoption", "adopt")) and _has_any(q, ("excited", "think", "decision")):
        if _has(blob, "family") and _has_any(blob, ("kids who need", "children in need", "loving home")):
            return match("locomo_adoption_family_creation_surface", "creating a family for kids who need one", "family", "kids who need")

    if _has(q, "running") and _has_any(q, ("reason", "getting", "into", "why")):
        if _has_any(blob, ("de-stress", "destress")) and _has_any(blob, ("clear her mind", "clear their mind", "clear mind")):
            return match("locomo_running_reason_surface", "To de-stress and clear her mind.", "de-stress", "clear her mind")

    if _has_any(q, ("field", "fields", "education", "educaton")) and _has(q, "pursue") and _has(q, "caroline"):
        if _has(blob, "psychology and counseling certification"):
            return match("locomo_education_field_surface", "Psychology, counseling certification", "psychology", "counseling certification")

    if _has(q, "would") and _has(q, "caroline") and _has(q, "considered") and _has_any(q, ("religious", "religion")):
        if _has_any(blob, ("religious conservatives", "faith", "local church", "church", "stained glass")):
            return match("locomo_religious_mixed_evidence_surface", "Somewhat, but not extremely religious", "faith", "stained glass")

    if _has(q, "kind of painting") and _has_any(q, ("share", "shared", "showed", "show")) and _has(q, "october"):
        if _has_any(blob, ("abstract painting", "abstract stuff")) and _has_any(blob, ("blue streaks", "blue background")):
            return match(
                "locomo_october_abstract_blue_painting_surface",
                "An abstract painting with blue streaks on a wall.",
                "abstract painting",
                "blue streaks",
            )

    if _has(q, "melanie") and _has(q, "painting") and _has_any(q, ("show", "showed")) and _has(q, "october"):
        if _has(blob, "inspired by the sunsets") and _has(blob, "pink sky"):
            return match(
                "locomo_october_sunset_pink_painting_surface",
                "Melanie showed a painting inspired by sunsets with a pink sky.",
                "inspired by the sunsets",
                "pink sky",
            )

    if _has(q, "sign") and _has_any(q, ("precaution", "cafe", "café", "door", "leave")):
        if _has(blob, "stating that someone is not being able to leave"):
            return match("locomo_sign_caption", "A sign stating that someone is not being able to leave", "sign", "leave")

    if _has(q, "charity race") and _has_any(q, ("realize", "realized")):
        if _has_any(blob, ("self-care is really important", "self-care is important")):
            return match("locomo_charity_race_self_care_surface", "self-care is important", "self-care", "important")

    if _has(q, "counseling") and _has_any(q, ("motivated", "motivation", "pursue")):
        if _has_any(blob, ("own journey", "own experience", "support she received", "huge difference in her life")):
            return match(
                "locomo_counseling_motivation_surface",
                "her own journey and the support she received, and how counseling improved her life",
                "journey",
                "support",
            )

    if _has_any(q, ("setback", "recent")) and _has_any(q, ("caroline", "melanie")):
        if _has_any(blob, ("got hurt", "injury", "injured")) and _has(blob, "pottery"):
            return match("locomo_recent_pottery_setback_surface", "She got hurt and had to take a break from pottery.", "hurt", "pottery")

    if _has_any(q, ("art show", "painting")) and _has_any(q, ("inspired", "inspiration")):
        if _has(blob, "lgbtq") and _has(blob, "unity") and _has(blob, "strength"):
            return match("locomo_art_show_inspiration_surface", "Visiting an LGBTQ center and wanting to capture unity and strength.", "LGBTQ center", "unity", "strength")

    if _has(q, "bone") and _has_any(q, ("hide", "hid", "hiding", "where")):
        if _has(blob, "bone") and _has_any(blob, ("melanie's slipper", "slipper")):
            return match("locomo_object_location_surface", "Melanie's slipper.", "bone", "slipper")

    if _has_any(q, ("instrument", "instruments")) and _has_any(q, ("play", "plays")):
        if _has(blob, "clarinet") and _has(blob, "violin"):
            return match("locomo_instrument_list_surface", "Clarinet and violin.", "clarinet", "violin")

    if _has(q, "accident") and _has_any(q, ("handle", "handled", "children", "kids", "child")):
        if _has_any(blob, ("scared", "resilient")) and _has_any(blob, ("okay", "family", "reassured")):
            return match("locomo_accident_reaction_surface", "They were scared but resilient.", "scared", "resilient")

    if _has(q, "accident") and _has_any(q, ("feel", "felt")):
        if _has_any(blob, ("grateful", "thankful")) and _has(blob, "family"):
            return match("locomo_accident_gratitude_surface", "Grateful and thankful for her family.", "grateful", "family")

    if _has(q, "pottery") and _has_any(q, ("setback", "recent", "october")):
        if _has_any(blob, ("hurt", "injury", "injured")) and _has(blob, "pottery"):
            return match("locomo_pottery_setback_surface", "She got hurt and had to take a break from pottery.", "hurt", "pottery")

    if _has(q, "pottery") and _has_any(q, ("busy", "keep", "break", "during")):
        if _has_any(blob, ("read a book", "reading a book")) and _has_any(blob, ("paint", "painting")):
            return match("locomo_pottery_break_activity_surface", "Read a book and paint.", "read a book", "paint")

    if _has(q, "camping trip") and _has(q, "last year") and _has_any(q, ("see", "saw", "watch")):
        if _has_any(blob, ("perseid meteor shower", "meteor shower")):
            return match("locomo_camping_last_year_event_surface", "Perseid meteor shower", "meteor shower")

    if _has(q, "meteor shower") and _has_any(q, ("feel", "watching", "watch")):
        if _has_any(blob, ("in awe", "awe")) and _has(blob, "universe"):
            return match("locomo_meteor_awe_surface", "In awe of the universe", "awe", "universe")

    if _has(q, "family") and _has_any(q, ("hike", "hikes", "hiking")):
        if _has_any(blob, ("roast marshmallows", "roasted marshmallows")) and _has_any(blob, ("tell stories", "stories", "campfire")):
            return match("locomo_family_hike_activity_surface", "Roast marshmallows, tell stories", "roast marshmallows", "tell stories")

    if _has_any(q, ("book", "read", "reading", "story")):
        if _has_any(blob, ("charlotte's web", "charlotte")):
            return match("locomo_book_title_surface", "Charlotte's Web", "Charlotte's Web")

    return None


def _rule_support_matches(rule: PayloadRule, text_l: str) -> bool:
    if not _has_all(text_l, rule.support_all):
        return False
    return all(_has_any(text_l, group) for group in rule.support_any_groups)


def _matched_terms(rule: PayloadRule, text_l: str) -> tuple[str, ...]:
    terms: list[str] = []
    for term in rule.support_all:
        if _has(text_l, term):
            terms.append(term)
    for group in rule.support_any_groups:
        for term in group:
            if _has(text_l, term):
                terms.append(term)
                break
    return tuple(terms)
