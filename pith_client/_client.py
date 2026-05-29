"""Synchronous Pith client using requests."""
from __future__ import annotations

from typing import Any

import requests as _requests

from ._base import DEFAULT_BASE_URL, ClientConfig, _extract_detail
from .exceptions import (
    PithAPIError,
    PithAuthError,
    PithConnectionError,
    PithTimeoutError,
)
from .models import (
    CKO,
    BackgroundTasksResponse,
    BeliefDiffResponse,
    BenchmarkResponse,
    CheckpointResponse,
    CKOListResponse,
    Concept,
    ConceptWriteResponse,
    ConversationTurnResponse,
    HealthResponse,
    HealthTrendResponse,
    ImportResponse,
    LearningMetricsResponse,
    LearnResponse,
    # Tier 2
    LinkResponse,
    # Tier 3
    MetricsDashboard,
    MetricsSummaryResponse,
    MigrationResponse,
    OrientResponse,
    Question,
    SearchResult,
    SessionInfo,
    # Tier 1
    SessionResponse,
    StatsResponse,
    ThreadsResponse,
    TracesResponse,
    ValidationResult,
)


class PithClient:
    """Synchronous Pith API client.

    Usage::

        from pith_client import PithClient

        pith = PithClient(api_key="your-key")
        pith.session_start(context_hint="onboarding")
        resp = pith.conversation_turn("What do I know about X?")
        for c in resp.activated_concepts:
            print(c.summary, c.confidence)
        pith.session_end()

    All methods return typed Pydantic models. The client raises
    :class:`PithAPIError` on server errors, :class:`PithAuthError`
    on 401/403, and :class:`PithTimeoutError` on timeouts.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        timeout: float = 180.0,
    ):
        self._cfg = ClientConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
        )
        self._session = _requests.Session()
        self._session.headers.update(self._cfg.headers)

    # ── HTTP helpers ──────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None,
             timeout: float | None = None) -> Any:
        url = self._cfg.url(path)
        try:
            resp = self._session.get(
                url, params=params,
                timeout=timeout or self._cfg.timeout)
        except _requests.ConnectionError as exc:
            raise PithConnectionError(url, str(exc)) from exc
        except _requests.Timeout as exc:
            raise PithTimeoutError(
                timeout or self._cfg.timeout, path) from exc
        return self._handle(resp, path)

    def _post(self, path: str, body: dict | None = None,
              params: dict | None = None,
              timeout: float | None = None) -> Any:
        url = self._cfg.url(path)
        try:
            resp = self._session.post(
                url, params=params, json=body or {},
                timeout=timeout or self._cfg.timeout)
        except _requests.ConnectionError as exc:
            raise PithConnectionError(url, str(exc)) from exc
        except _requests.Timeout as exc:
            raise PithTimeoutError(
                timeout or self._cfg.timeout, path) from exc
        return self._handle(resp, path)

    def _put(self, path: str, body: dict | None = None,
             timeout: float | None = None) -> Any:
        url = self._cfg.url(path)
        try:
            resp = self._session.put(
                url, json=body or {},
                timeout=timeout or self._cfg.timeout)
        except _requests.ConnectionError as exc:
            raise PithConnectionError(url, str(exc)) from exc
        except _requests.Timeout as exc:
            raise PithTimeoutError(
                timeout or self._cfg.timeout, path) from exc
        return self._handle(resp, path)

    def _delete(self, path: str, params: dict | None = None,
                timeout: float | None = None) -> Any:
        url = self._cfg.url(path)
        try:
            resp = self._session.delete(
                url, params=params,
                timeout=timeout or self._cfg.timeout)
        except _requests.ConnectionError as exc:
            raise PithConnectionError(url, str(exc)) from exc
        except _requests.Timeout as exc:
            raise PithTimeoutError(
                timeout or self._cfg.timeout, path) from exc
        return self._handle(resp, path)

    @staticmethod
    def _handle(resp: _requests.Response, path: str) -> Any:
        if resp.status_code in (401, 403):
            try:
                detail = _extract_detail(resp.json())
            except Exception:
                detail = resp.text
            raise PithAuthError(detail)
        if resp.status_code >= 400:
            try:
                detail = _extract_detail(resp.json())
            except Exception:
                detail = resp.text
            raise PithAPIError(resp.status_code, detail)
        return resp.json()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Tier 1: Core (15 methods) ─────────────────────────────

    def health(self) -> HealthResponse:
        """GET /pith_health — cognitive health analysis."""
        return HealthResponse(**self._get(
            "/pith_health", timeout=self._cfg.health_timeout))

    def stats(self) -> StatsResponse:
        """GET /pith_stats — system overview."""
        return StatsResponse(**self._get("/pith_stats"))

    def session_start(
        self,
        context_hint: str = "",
        agent_id: str = "default",
    ) -> SessionResponse:
        """POST /session_start — begin a session."""
        body: dict[str, Any] = {}
        if context_hint:
            body["context_hint"] = context_hint
        if agent_id != "default":
            body["agent_id"] = agent_id
        return SessionResponse(**self._post(
            "/session_start", body,
            timeout=self._cfg.session_timeout))

    def session_end(
        self, session_id: str | None = None,
    ) -> SessionResponse:
        """POST /session_end — end session, trigger reflection."""
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        return SessionResponse(**self._post(
            "/session_end", body,
            timeout=self._cfg.session_timeout))

    def conversation_turn(
        self,
        message: str,
        previous_response: str = "",
        previous_message: str = "",
        extracted_concepts_json: str = "[]",
        session_id: str | None = None,
        max_concepts: int = 14,
        origin_id: str | None = None,
        current_task_id: str | None = None,
        context_authority_mode: str = "balanced",
    ) -> ConversationTurnResponse:
        """POST /conversation_turn — the main learning+retrieval call."""
        body: dict[str, Any] = {
            "message": message,
            "extracted_concepts_json": extracted_concepts_json,
            "max_concepts": max_concepts,
        }
        if previous_response:
            body["previous_response"] = previous_response
        if previous_message:
            body["previous_message"] = previous_message
        if session_id:
            body["session_id"] = session_id
        if origin_id:
            body["origin_id"] = origin_id
        if current_task_id:
            body["current_task_id"] = current_task_id
        if context_authority_mode != "balanced":
            body["context_authority_mode"] = context_authority_mode
        return ConversationTurnResponse(**self._post(
            "/conversation_turn", body))

    def session_learn(
        self,
        content: str,
        knowledge_area: str = "general",
        source: str = "sdk",
        agent_id: str | None = None,
    ) -> LearnResponse:
        """POST /session_learn — explicit teaching."""
        body: dict[str, Any] = {
            "content": content,
            "knowledge_area": knowledge_area,
            "source": source,
        }
        if agent_id:
            body["agent_id"] = agent_id
        return LearnResponse(**self._post("/session_learn", body))

    def search(
        self,
        query: str,
        max_results: int = 5,
        min_confidence: float = 0.0,
        context: str | None = None,
        goal: str | None = None,
        ka_boost: list[str] | None = None,
    ) -> SearchResult:
        """POST /pith_search — semantic search over concepts.

        Matches ``SearchQuery`` model in ``app/core/models.py``.
        """
        body: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "min_confidence": min_confidence,
        }
        if context:
            body["context"] = context
        if goal:
            body["goal"] = goal
        if ka_boost:
            body["ka_boost"] = ka_boost
        return SearchResult(**self._post("/pith_search", body))

    def get_concept(
        self, concept_id: str, version: str = "latest",
    ) -> Concept:
        """GET /pith_get_concept — retrieve a single concept by ID."""
        return Concept(**self._get(
            "/pith_get_concept",
            params={"concept_id": concept_id, "version": version}))

    def propose_concept(
        self,
        concept_id: str,
        summary: str,
        knowledge_area: str = "general",
        confidence: float = 0.5,
        evidence: list[str] | None = None,
        concept_type: str = "observation",
    ) -> ConceptWriteResponse:
        """POST /pith_propose_concept — create new knowledge."""
        body: dict[str, Any] = {
            "concept_id": concept_id,
            "summary": summary,
            "knowledge_area": knowledge_area,
            "confidence": confidence,
            "concept_type": concept_type,
        }
        if evidence:
            body["evidence"] = evidence
        return ConceptWriteResponse(**self._post("/pith_propose_concept", body))

    def evolve_concept(
        self,
        concept_id: str,
        new_summary: str | None = None,
        new_evidence: list[str] | None = None,
        confidence_change: float = 0.0,
    ) -> ConceptWriteResponse:
        """POST /pith_evolve_concept — update existing knowledge."""
        body: dict[str, Any] = {"concept_id": concept_id}
        if new_summary:
            body["new_summary"] = new_summary
        if new_evidence:
            body["new_evidence"] = new_evidence
        if confidence_change:
            body["confidence_change"] = confidence_change
        return ConceptWriteResponse(**self._post("/pith_evolve_concept", body))

    def reflect(self) -> dict[str, Any]:
        """POST /pith_reflect — run consolidation cycle."""
        return self._post("/pith_reflect")

    def checkpoint(
        self,
        task_id: str,
        description: str,
        action: str = "save",
        status: str = "active",
        done: list[str] | None = None,
        active: str = "",
        next_items: list[str] | None = None,
        blockers: list[str] | None = None,
        context: dict[str, Any] | None = None,
        concept_refs: list[str] | None = None,
        session_id: str | None = None,
    ) -> CheckpointResponse:
        """POST /checkpoint — save/load/touch/complete execution state."""
        body: dict[str, Any] = {
            "action": action,
            "task_id": task_id,
            "description": description,
            "status": status,
        }
        if done is not None:
            body["done"] = done
        if active:
            body["active"] = active
        if next_items is not None:
            body["next"] = next_items
        if blockers is not None:
            body["blockers"] = blockers
        if context is not None:
            body["context"] = context
        if concept_refs is not None:
            body["concept_refs"] = concept_refs
        if session_id:
            body["session_id"] = session_id
        return CheckpointResponse(**self._post("/checkpoint", body))

    def orient(self) -> OrientResponse:
        """GET /pith_orient — present-moment orientation."""
        return OrientResponse(**self._get("/pith_orient"))

    def set_goal(
        self, goal: str, context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /pith_set_goal — goal-directed retrieval."""
        params: dict[str, Any] = {"goal": goal}
        if context:
            params["context"] = context
        return self._post("/pith_set_goal", params=params)

    def sessions_list(
        self,
        status: str | None = None,
        limit: int = 20,
        since: str | None = None,
    ) -> list[SessionInfo]:
        """GET /sessions_list — browse session history."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if since:
            params["since"] = since
        raw = self._get("/sessions_list", params=params)
        if isinstance(raw, list):
            return [SessionInfo(**item) for item in raw]
        if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
            return [SessionInfo(**item) for item in raw["sessions"]]
        raise PithAPIError(500, "Unexpected sessions_list response shape")

    # ── Tier 2: Extended (17 methods) ─────────────────────────

    # Graph & Associations

    def related_concepts(
        self, concept_id: str, max_depth: int = 2,
    ) -> list[str]:
        """GET /pith_related_concepts — graph walk from a concept."""
        return self._get(
            "/pith_related_concepts",
            params={"concept_id": concept_id, "max_depth": max_depth})

    def link_concepts(
        self,
        concept_a: str,
        concept_b: str,
        relation: str = "related_to",
        strength: float = 0.5,
    ) -> LinkResponse:
        """POST /pith_link_concepts — create manual association.

        Sends an ``Association`` body per ``app/core/models.py``.
        """
        return LinkResponse(**self._post("/pith_link_concepts", {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "relation": relation,
            "strength": strength,
        }))

    def set_always_activate(
        self, concept_id: str, value: bool = True,
    ) -> dict[str, Any]:
        """POST /pith_set_always_activate — pin critical concepts.

        Server ``AlwaysActivateRequest`` uses field ``value``, not
        ``always_activate``.
        """
        return self._post("/pith_set_always_activate", {
            "concept_id": concept_id,
            "value": value,
        })

    def activate_context(
        self, context: str, boost: float = 0.5,
    ) -> dict[str, Any]:
        """POST /pith_activate_context — pre-warm retrieval.

        Server expects a semantic context string and boost factor,
        NOT a list of concept IDs.
        """
        return self._post(
            "/pith_activate_context",
            params={"context": context, "boost": boost},
        )

    # Reasoning & Validation

    def questions(self, limit: int = 10) -> list[Question]:
        """GET /pith_questions — surface uncertain knowledge."""
        raw = self._get("/pith_questions", params={"limit": limit})
        if isinstance(raw, list):
            return [Question(**q) for q in raw]
        return []

    def validate_response(
        self,
        response_text: str,
        constraint_set: dict[str, Any],
        skip_validation: bool = False,
    ) -> ValidationResult:
        """POST /validate_response — check against constraints.

        ``constraint_set`` is REQUIRED by the server. Pass the
        ``constraint_set`` dict from a ``ConversationTurnResponse``.
        """
        return ValidationResult(**self._post(
            "/validate_response", {
                "response_text": response_text,
                "constraint_set": constraint_set,
                "skip_validation": skip_validation,
            }))

    def belief_diff(
        self,
        t1: str,
        t2: str,
        knowledge_area: str | None = None,
    ) -> BeliefDiffResponse:
        """POST /belief_diff — compare belief states over time.

        ``t1`` and ``t2`` are ISO-8601 datetime strings (both required).
        """
        body: dict[str, Any] = {"t1": t1, "t2": t2}
        if knowledge_area:
            body["knowledge_area"] = knowledge_area
        return BeliefDiffResponse(**self._post("/belief_diff", body))

    def import_conversation(
        self,
        conversation_text: str,
        source_id: str = "manual_import",
        knowledge_area: str = "imported",
        chunk_size: int = 200,
    ) -> ImportResponse:
        """POST /pith_import_conversation — import historical data.

        Server processes raw conversation text (not structured turns).
        """
        return ImportResponse(**self._post(
            "/pith_import_conversation",
            params={
                "conversation_text": conversation_text,
                "source_id": source_id,
                "knowledge_area": knowledge_area,
                "chunk_size": chunk_size,
            }))

    # Compound Knowledge Objects (CKOs)

    def cko_create(
        self,
        title: str,
        concept_ids: list[str],
        synthesis: str,
        knowledge_area: str = "general",
        cko_type: str = "analysis",
    ) -> CKO:
        """POST /pith/cko — create a compound knowledge object.

        ``synthesis`` is REQUIRED (server ``CKOCreateRequest``).
        """
        return CKO(**self._post("/pith/cko", {
            "title": title,
            "concept_ids": concept_ids,
            "synthesis": synthesis,
            "knowledge_area": knowledge_area,
            "cko_type": cko_type,
        }))

    def cko_get(self, cko_id: str) -> CKO:
        """GET /pith/cko/{id} — retrieve a CKO."""
        return CKO(**self._get(f"/pith/cko/{cko_id}"))

    def cko_search(
        self,
        query_area: str | None = None,
        max_results: int = 3,
    ) -> CKOListResponse:
        """POST /pith/cko/search — search CKOs."""
        body: dict[str, Any] = {"max_results": max_results}
        if query_area:
            body["query_area"] = query_area
        return CKOListResponse(**self._post("/pith/cko/search", body))

    def cko_update(
        self, cko_id: str, **kwargs: Any,
    ) -> CKO:
        """PUT /pith/cko/{id} — update a CKO."""
        return CKO(**self._put(f"/pith/cko/{cko_id}", kwargs))

    def cko_lifecycle(self) -> dict[str, Any]:
        """POST /pith/cko/lifecycle — run CKO lifecycle management."""
        return self._post("/pith/cko/lifecycle")

    def cko_list(
        self, knowledge_area: str | None = None,
        limit: int = 20,
    ) -> CKOListResponse:
        """GET /pith/cko — list all CKOs."""
        params: dict[str, Any] = {"limit": limit}
        if knowledge_area:
            params["knowledge_area"] = knowledge_area
        return CKOListResponse(**self._get("/pith/cko", params=params))

    # Threads & Traces

    def threads(self, **kwargs: Any) -> ThreadsResponse:
        """POST /pith_threads — narrative thread management."""
        return ThreadsResponse(**self._post("/pith_threads", kwargs))

    def traces(self, **kwargs: Any) -> TracesResponse:
        """POST /pith_traces — cognitive trace retrieval."""
        return TracesResponse(**self._post("/pith_traces", kwargs))

    def learning_metrics(self) -> LearningMetricsResponse:
        """GET /learning_metrics — learning pipeline health."""
        return LearningMetricsResponse(**self._get("/learning_metrics"))

    # ── Tier 3: Platform (7 methods) ──────────────────────────

    def observability(self) -> dict[str, Any]:
        """GET /pith/observability — unified system health snapshot.

        Returns cognitive health, stats, learning metrics, performance,
        and background task status in a single call.
        """
        return self._get("/pith/observability")

    def metrics_dashboard(
        self, since: str | None = None,
    ) -> MetricsDashboard:
        """GET /metrics/dashboard — the Critical 8 metrics."""
        params: dict[str, Any] = {}
        if since:
            params["since"] = since
        return MetricsDashboard(**self._get(
            "/metrics/dashboard", params=params or None))

    def metrics_bg_tasks(
        self, since: str | None = None,
    ) -> BackgroundTasksResponse:
        """GET /metrics/bg_tasks — background task health."""
        params: dict[str, Any] = {}
        if since:
            params["since"] = since
        return BackgroundTasksResponse(**self._get(
            "/metrics/bg_tasks", params=params or None))

    def metrics_summary(self, days: int = 7) -> MetricsSummaryResponse:
        """GET /metrics/summary — aggregated metrics."""
        return MetricsSummaryResponse(**self._get(
            "/metrics/summary", params={"days": days}))

    def metrics_health_trend(
        self, days: int = 7,
    ) -> HealthTrendResponse:
        """GET /metrics/health_trend — health score time series."""
        return HealthTrendResponse(**self._get(
            "/metrics/health_trend", params={"days": days}))

    def auto_associate_batch(
        self, **kwargs: Any,
    ) -> dict[str, Any]:
        """POST /auto_associate_batch — bulk graph enrichment."""
        return self._post("/auto_associate_batch", kwargs)

    def benchmark(self, **kwargs: Any) -> BenchmarkResponse:
        """POST /pith/benchmark — governance benchmark."""
        return BenchmarkResponse(**self._post(
            "/pith/benchmark", kwargs))

    def migrate_epistemic(
        self, **kwargs: Any,
    ) -> MigrationResponse:
        """POST /migrate_epistemic_networks — epistemic migration."""
        return MigrationResponse(**self._post(
            "/migrate_epistemic_networks", kwargs))
