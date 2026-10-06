"""Bounded original-input accounting for session learning. No persistence or retry."""

from app.core.models import ClientLearningItem, ClientLearningRange, ClientLearningReceipt


class LearningReceiptBuilder:
    def __init__(self, input_count, cap):
        self.input_count = input_count
        self.admitted = min(input_count, max(0, cap))
        self.items = {
            i: ClientLearningItem(
                input_index=i, status="deferred", reason="not_processed", persistence_evidence="not_attempted"
            )
            for i in range(self.admitted)
        }
        self.ranges = (
            [ClientLearningRange(start_index=self.admitted, end_index_exclusive=input_count, reason="client_cap")]
            if input_count > self.admitted
            else []
        )

    def mark(self, index, status, reason, persistence="not_attempted", concept_id=None):
        if index is None:
            return
        if index not in self.items:
            raise ValueError("invalid server-owned client input index")
        self.items[index] = ClientLearningItem(
            input_index=index, status=status, reason=reason, persistence_evidence=persistence, concept_id=concept_id
        )

    def processed(self, index, result):
        action = result.get("action")
        concept = result.get("learned_concept") or result.get("evolved_concept")
        concept_id = getattr(concept, "concept_id", None)
        if action in {"created", "evolved"}:
            if not concept_id:
                self.mark(index, "error", "missing_saved_identity", "unknown")
                return "error"
            self.mark(index, action, action, "reported_saved", concept_id)
            return action
        if action == "skipped_per_call_cap":
            self.mark(index, "deferred", action, "unknown")
            return "deferred"
        if action in {"gated", "rejected_contradiction"}:
            self.mark(index, "rejected", action, "unknown")
            return "rejected"
        if action in {
            "skipped_confidence_heuristic",
            "skipped_confidence",
            "skipped_short_summary",
            "skipped_no_evidence",
            "skipped_duplicate",
            "skipped_saturated",
            "skipped_evidence_cap",
        }:
            self.mark(index, "skipped", action, "unknown")
            return "skipped"
        self.mark(index, "error", "unknown_processing_action", "unknown", concept_id)
        return "error"

    def capture_state(self, accepted, errors, rejected=0, deferred=False):
        statuses = {item.status for item in self.items.values()}
        has_rejection = rejected > 0 or "rejected" in statuses
        has_error = errors > 0 or "error" in statuses
        has_deferred = deferred or bool(self.ranges) or "deferred" in statuses
        if accepted > 0:
            return "partial" if has_error or has_rejection or has_deferred else "accepted"
        if has_error:
            return "error"
        if has_rejection:
            return "rejected"
        return "deferred" if has_deferred else "zero_learning"

    def build(self):
        return ClientLearningReceipt(
            input_count=self.input_count, items=list(self.items.values()), deferred_ranges=self.ranges
        )
