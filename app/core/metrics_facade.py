"""
app.core.metrics_facade — thin re-export shim for app.ops.metrics.

DEBT-244: cognitive/ and retrieval/ modules must not import app.ops.metrics
directly (Contract 3 / Contract 5 violations). This facade lives in app.core
(the foundation layer, importable by everyone) and simply re-exports the
public surface of app.ops.metrics.

Usage:
    # Instead of:
    from app.ops.metrics import metrics
    # Use:
    from app.core.metrics_facade import metrics

The facade holds no logic — it is purely a layer-boundary adapter.
Any future migration to a different metrics backend only requires updating
this one file.
"""

from app.ops.metrics import (  # noqa: F401  (re-export)
    MetricsCollector,
    metrics,
)
from app.ops.traces import (  # noqa: F401  (re-export)
    create_trace,
    resolve_predictions_for_concept,
)

__all__ = ["MetricsCollector", "metrics", "create_trace", "resolve_predictions_for_concept"]
