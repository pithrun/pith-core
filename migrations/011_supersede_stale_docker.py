"""Migration 011: Supersede stale Docker/Node.js architecture concepts.

Targets concepts identified in RETRIEVAL_GAP_SPEC §2 E3 that reference
eliminated technologies (Docker, Node.js wrapper) at high confidence.

Spec: RETRIEVAL_GAP_SPEC_v1.md (Fix 5)
Backlog: DATA-057
"""
from app.staleness import sweep_stale_technology_refs


def migrate():
    """Run the stale technology sweep."""
    # First dry run to log what will change
    dry_result = sweep_stale_technology_refs(dry_run=True)
    print(f"Dry run: {dry_result['matched']} concepts matched")
    for d in dry_result["details"][:10]:
        print(f"  {d['id']}: conf={d['confidence']:.2f}, pattern='{d['pattern']}'")

    # Execute
    result = sweep_stale_technology_refs(dry_run=False)
    print(f"Superseded: {result['superseded']} concepts")
    return result


if __name__ == "__main__":
    migrate()
