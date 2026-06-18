# pith-client

Python SDK for the [Pith](https://github.com/pithrun/pith-core) Cognitive Runtime — a persistent knowledge layer for AI systems.

## Install

```bash
pip install pith-client
```

Requires Python 3.10+. The sync client uses `requests`; the async client uses `httpx`.

## Quick Start

```python
from pith_client import PithClient

pith = PithClient(api_key="your-key")

# Start a session
pith.session_start(context_hint="exploring pith SDK")

# Store knowledge via conversation
resp = pith.conversation_turn(
    message="How does the retrieval pipeline work?",
    origin_id="api-thread:retrieval-design-42",
    extracted_concepts_json='[{"summary": "Pith uses embedding-based retrieval", "confidence": 0.7, "knowledge_area": "architecture"}]',
)

# Access activated concepts
for concept in resp.activated_concepts:
    print(f"{concept.summary} (confidence: {concept.confidence})")

# Search knowledge
results = pith.search("retrieval pipeline", max_results=5)
for r in results.results:
    print(f"  {r.concept_id}: {r.summary}")

# End session (triggers reflection)
pith.session_end()
```

### Async Usage

```python
import asyncio
from pith_client import AsyncPithClient

async def main():
    async with AsyncPithClient(api_key="your-key") as pith:
        await pith.session_start()
        resp = await pith.conversation_turn("What patterns have I learned?")
        print(resp.orientation_summary)
        await pith.session_end()

asyncio.run(main())
```

## Configuration

```python
pith = PithClient(
    base_url="http://localhost:8000",  # Default
    api_key="your-api-key",            # X-API-Key header
    timeout=180.0,                     # Request timeout (seconds)
)
```

Environment variable support:

```bash
export PITH_API_KEY="your-key"
export PITH_BASE_URL="http://localhost:8000"
```

## SDK Contract Notes

`propose_concept()` now requires an explicit `concept_id` because the live server
requires it:

```python
result = pith.propose_concept(
    concept_id="example_concept_001",
    summary="Example concept",
    evidence=["sdk example"],
)
```

`propose_concept()` and `evolve_concept()` return write-envelope metadata, not a
hydrated `Concept` body. If you need the full concept after a write, do a follow-up
read with `get_concept(result.concept_id)`.

## API Surface

The SDK exposes 40 methods organized into three tiers matching the server's capability structure.

### Tier 1: Core (15 methods)

The essential loop — session management, learning, retrieval.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `health()` | GET /pith_health | Cognitive health analysis |
| `stats()` | GET /pith_stats | System overview |
| `session_start()` | POST /session_start | Begin a session |
| `session_end()` | POST /session_end | End session + reflect |
| `conversation_turn()` | POST /conversation_turn | Main learning + retrieval call |
| `session_learn()` | POST /session_learn | Explicit teaching |
| `search()` | POST /pith_search | Semantic concept search |
| `get_concept()` | GET /pith_get_concept | Retrieve by ID |
| `propose_concept()` | POST /pith_propose_concept | Create knowledge |
| `evolve_concept()` | POST /pith_evolve_concept | Update knowledge |
| `reflect()` | POST /pith_reflect | Run consolidation |
| `checkpoint()` | POST /checkpoint | Save execution state |
| `orient()` | GET /pith_orient | Present-moment orientation |
| `set_goal()` | POST /pith_set_goal | Goal-directed retrieval |
| `sessions_list()` | GET /sessions_list | Browse session history |

### Tier 2: Extended (17 methods)

Graph operations, reasoning, validation, CKOs.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `related_concepts()` | GET /pith_related_concepts | Graph walk |
| `link_concepts()` | POST /pith_link_concepts | Manual association |
| `set_always_activate()` | POST /pith_set_always_activate | Pin concepts |
| `activate_context()` | POST /pith_activate_context | Pre-warm retrieval |
| `questions()` | GET /pith_questions | Surface uncertain knowledge |
| `validate_response()` | POST /validate_response | Check against constraints |
| `belief_diff()` | POST /belief_diff | Compare belief states |
| `import_conversation()` | POST /pith_import_conversation | Import historical data |
| `cko_create()` | POST /pith/cko | Create compound knowledge object |
| `cko_get()` | GET /pith/cko/{id} | Retrieve CKO |
| `cko_search()` | POST /pith/cko/search | Search CKOs |
| `cko_update()` | PUT /pith/cko/{id} | Update CKO |
| `cko_lifecycle()` | POST /pith/cko/lifecycle | CKO lifecycle management |
| `cko_list()` | GET /pith/cko | List CKOs |
| `threads()` | POST /pith_threads | Workstream/thread operations |
| `traces()` | POST /pith_traces | Cognitive trace list/search |
| `learning_metrics()` | GET /learning_metrics | Learning pipeline health |

### Tier 3: Platform (8 methods)

Observability, benchmarking, migration.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `metrics_dashboard()` | GET /metrics/dashboard | The Critical 8 metrics |
| `observability()` | GET /pith/observability | Unified system health snapshot |
| `metrics_bg_tasks()` | GET /metrics/bg_tasks | Background task health |
| `metrics_summary()` | GET /metrics/summary | Aggregated metrics |
| `metrics_health_trend()` | GET /metrics/health_trend | Health time series |
| `auto_associate_batch()` | POST /auto_associate_batch | Bulk graph enrichment |
| `benchmark()` | POST /pith/benchmark | Governance benchmark |
| `migrate_epistemic()` | POST /migrate_epistemic_networks | Epistemic migration |

## Response Models

All methods return typed Pydantic models with `extra="allow"` for forward compatibility — new server fields are preserved without requiring a client upgrade.

```python
resp = pith.conversation_turn("hello")

# Typed access
print(resp.activation_count)          # int
print(resp.orientation_summary)       # str | None
print(resp.constraint_set)            # ConstraintSet | None
print(resp.processing_time_ms)        # float

# Forward-compatible — unknown fields are accessible
print(resp.some_future_field)         # Works if server sends it
```

Key models: `SessionResponse`, `ConversationTurnResponse`, `LearnResponse`, `SearchResult`, `Concept`, `ConstraintSet`, `CheckpointResponse`, `LinkResponse`, `CKO`, `ValidationResult`, `BeliefDiffResponse`.

## Error Handling

```python
from pith_client import PithClient, PithAPIError, PithAuthError, PithTimeoutError

try:
    pith.search("test")
except PithAuthError as e:
    print(f"Authentication failed: {e}")
except PithTimeoutError as e:
    print(f"Request timed out after {e.timeout}s")
except PithAPIError as e:
    print(f"Server error {e.status_code}: {e.detail}")
```

## Context Manager

Both clients support context managers for automatic cleanup:

```python
with PithClient(api_key="k") as pith:
    pith.session_start()
    # ... work ...
    pith.session_end()
# Connection pool closed automatically

async with AsyncPithClient(api_key="k") as pith:
    await pith.session_start()
    # ... work ...
    await pith.session_end()
```

## License

MIT
