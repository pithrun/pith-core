"""Shared learning-result interpretation; also embedded into standalone hooks."""


def learning_count(value):
    """Return a nonnegative integer, or None for malformed explicit evidence."""
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed < 0 or str(value).strip() != str(parsed):
        return None
    return parsed


def classify_learning_result(result):
    """Transport failure/pending precedes capture evidence; explicit zero wins."""
    if not isinstance(result, dict):
        return "failed"
    states = {str(result.get(key) or "") for key in ("status", "processing_state", "persistence_state")}
    if result.get("error") is True or states & {"failed", "error"}:
        return "failed"
    if states & {"processing", "queued", "retry", "unknown", "unknown_pending"}:
        return "processing" if "processing" in states else "unknown_pending"
    if states & {"partial", "deferred"}:
        return "partial"
    if states & {"rejected", "degraded_terminal_session"}:
        return "failed"
    if "degraded_zero_learning" in states:
        return "degraded_zero_learning"
    summary = result.get("summary")
    if isinstance(summary, dict):
        return classify_learning_result(summary)
    capture = str(result.get("learning_capture_state") or "")
    accepted_key = "accepted_learning_events" if "accepted_learning_events" in result else "learning_events"
    accepted = learning_count(result.get(accepted_key, 0))
    errors = learning_count(result.get("errors", 0))
    if accepted is None or errors is None:
        return "failed"
    if capture in {"error", "rejected", "degraded_terminal_session"}:
        return "failed"
    if capture in {"partial", "deferred"} or errors > 0:
        return "partial" if accepted > 0 or capture in {"partial", "deferred"} else "failed"
    if capture and capture not in {"accepted", "zero_learning"}:
        return "failed"
    if accepted > 0 and capture != "zero_learning":
        return "committed"
    if states & {"committed"} or accepted_key in result or capture:
        return "degraded_zero_learning"
    return str(result.get("status") or result.get("processing_state") or "ok")


def learning_result_body(operation, body):
    """Select the completed learning operation, never a session lifetime total."""
    if not isinstance(body, dict):
        return {}
    key = {"conversation_turn": "auto_learned", "session_end": "last_exchange_learning"}.get(operation)
    nested = body.get(key) if key else None
    return nested if isinstance(nested, dict) else body
