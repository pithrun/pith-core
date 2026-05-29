"""Domain seeder for Pith (Layer 3).

Seeds cognitive_domains and domain_area_mapping tables.
Analogous to seed_firmware.py — idempotent, runs on server startup.

Domain definitions from DOMAINS_AND_DIRECTIVES_SPEC.md Section 2.2.
"""

import json
import logging

from app.storage import _db, get_metadata, set_metadata

logger = logging.getLogger(__name__)

DOMAINS_VERSION = "1.0.0"

# --- Domain Catalog ---
# (domain_id, name, description, triggers_list, strategic_priority)
DOMAIN_CATALOG = [
    (
        "pith_strategy",
        "Pith Strategy",
        "Product, business, and competitive strategy for Pith",
        [
            "pricing",
            "market",
            "competitor",
            "ICP",
            "agency",
            "GHL",
            "funding",
            "YC",
            "revenue",
            "beta",
            "customer",
            "GTM",
            "positioning",
        ],
        0.5,
    ),
    (
        "pith_engineering",
        "Pith Engineering",
        "Technical architecture and implementation of the Pith system",
        [
            "code",
            "build",
            "API",
            "schema",
            "database",
            "SQLite",
            "MCP",
            "server",
            "spec",
            "migration",
            "pipeline",
            "retrieval",
            "governance",
        ],
        0.5,
    ),
    (
        "quality_assurance",
        "Quality Assurance",
        "Testing, quality, debugging, security, performance",
        ["bug", "test", "fix", "error", "broken", "slow", "security", "audit", "regression", "trace", "benchmark"],
        0.5,
    ),
    (
        "methodology",
        "Methodology",
        "How we work — processes, reviews, learning, project management",
        [
            "sprint",
            "retrospective",
            "review",
            "checkpoint",
            "session",
            "workflow",
            "plan",
            "spec",
            "design",
            "protocol",
        ],
        0.5,
    ),
    (
        "operations",
        "Operations",
        "Runtime operations, tooling, deployment, integration",
        ["deploy", "backup", "cron", "process", "PID", "WAL", "Docker", "container", "maintenance", "scheduler"],
        0.5,
    ),
    (
        "uncategorized",
        "Uncategorized",
        "Catch-all for concepts not yet mapped to a domain",
        [],  # Never explicitly activated
        0.0,
    ),
]

# --- Area Mappings ---
# (domain_id, knowledge_area, activation_weight)
AREA_MAPPINGS = [
    # pith_strategy
    ("pith_strategy", "product_strategy", 0.3),
    ("pith_strategy", "business_strategy", 0.3),
    ("pith_strategy", "competitive_analysis", 0.3),
    ("pith_strategy", "ip_protection", 0.2),
    ("pith_strategy", "product_operations", 0.2),
    # pith_engineering
    ("pith_engineering", "architecture", 0.3),
    ("pith_engineering", "implementation", 0.3),
    ("pith_engineering", "architecture_gaps", 0.3),
    ("pith_engineering", "engineering_patterns", 0.3),
    ("pith_engineering", "design_principles", 0.25),
    ("pith_engineering", "pith_engineering", 0.3),
    # quality_assurance
    ("quality_assurance", "testing", 0.3),
    ("quality_assurance", "system_quality", 0.3),
    ("quality_assurance", "debugging", 0.3),
    ("quality_assurance", "security", 0.25),
    ("quality_assurance", "performance", 0.25),
    ("quality_assurance", "cognitive_safety", 0.2),
    # methodology
    ("methodology", "process", 0.3),
    ("methodology", "review_methodology", 0.3),
    ("methodology", "learning", 0.25),
    ("methodology", "project_status", 0.2),
    ("methodology", "documentation", 0.2),
    ("methodology", "specification", 0.3),
    # operations
    ("operations", "operations", 0.3),
    ("operations", "tooling", 0.25),
    ("operations", "integration", 0.25),
    ("operations", "protocol", 0.2),
    ("operations", "tool_routing", 0.2),
    # uncategorized
    ("uncategorized", "general", 0.0),
]


def seed_domains() -> dict:
    """Seed domain tables with current catalog. Idempotent."""
    current_version = get_metadata("domains_version")

    if current_version == DOMAINS_VERSION:
        logger.info(
            f"seed_domains: v{DOMAINS_VERSION} already seeded, "
            f"skipping ({len(DOMAIN_CATALOG)} domains, {len(AREA_MAPPINGS)} mappings)"
        )
        return {
            "action": "skipped",
            "version": DOMAINS_VERSION,
            "reason": "already_seeded",
        }

    logger.info(f"seed_domains: seeding v{DOMAINS_VERSION} (previous: {current_version or 'none'})")

    domains_seeded = 0
    mappings_seeded = 0

    with _db() as conn:
        for domain_id, name, desc, triggers, priority in DOMAIN_CATALOG:
            try:
                conn.execute(
                    """
                    INSERT INTO cognitive_domains
                        (domain_id, name, description, activation_triggers, strategic_priority)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(domain_id) DO UPDATE SET
                        name = excluded.name,
                        description = excluded.description,
                        activation_triggers = excluded.activation_triggers,
                        strategic_priority = excluded.strategic_priority,
                        updated_at = datetime('now')
                """,
                    (domain_id, name, desc, json.dumps(triggers), priority),
                )
                domains_seeded += 1
            except Exception as e:
                logger.error(f"seed_domains: failed to seed domain '{domain_id}': {e}")

        for domain_id, area, weight in AREA_MAPPINGS:
            try:
                conn.execute(
                    """
                    INSERT INTO domain_area_mapping
                        (domain_id, knowledge_area, activation_weight)
                    VALUES (?, ?, ?)
                    ON CONFLICT(domain_id, knowledge_area) DO UPDATE SET
                        activation_weight = excluded.activation_weight
                """,
                    (domain_id, area, weight),
                )
                mappings_seeded += 1
            except Exception as e:
                logger.error(f"seed_domains: failed to seed mapping '{domain_id}/{area}': {e}")

    # Only record version if at least some data was seeded successfully
    if domains_seeded > 0 or mappings_seeded > 0:
        set_metadata("domains_version", DOMAINS_VERSION)

    logger.info(f"seed_domains: seeded {domains_seeded} domains, {mappings_seeded} mappings")

    return {
        "action": "seeded",
        "version": DOMAINS_VERSION,
        "previous_version": current_version,
        "domains_seeded": domains_seeded,
        "mappings_seeded": mappings_seeded,
    }
