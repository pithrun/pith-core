# Pith Benchmark Notes

## Current Launch Evidence

Pith's public benchmark claims use external benchmark families with score-bearing artifacts. Internal CogGov-Bench results are QA evidence only and are not public benchmark claims.

| Benchmark family | Run class | Score | N | Evidence status |
|---|---|---:|---:|---|
| MemoryAgentBench / FactConsolidation | Standard multi-hop, MH6K post-merge replay | 95.0 EM / 95.0 F1 | 100 | Score-bearing comparable artifact |
| MemoryAgentBench / FactConsolidation | Standard multi-hop, MH32K post-merge replay | 83.0 EM / 83.0 F1 | 100 | Score-bearing comparable artifact |
| MemoryAgentBench / FactConsolidation | Standard multi-hop, MH64K post-merge replay | 84.0 EM / 84.0 F1 | 100 | Score-bearing comparable artifact |
| MemoryAgentBench / FactConsolidation | MH262 high-water addendum | 68.0 EM / 68.2 F1 | 100 | Score-bearing high-water artifact |
| LoCoMo-Plus official Cognitive all401 | `pith-memory-adapter / boundary_checked_v1` | 100.00% judge score | 401 | Complete score-bearing post-merge artifact |

## Artifact References

| Claim | Run ID / artifact | SHA-256 |
|---|---|---|
| MAB MH6K post-merge replay | `mab_fc_mh_6k_production_20260527T145219_2026-05-27T14-54-18.json` | `5058f7778e1df689f7242761101bd53b93ff7f04a10e143d98015854be87598a` |
| MAB MH32K post-merge replay | `mab_fc_mh_32k_production_20260527T145421_2026-05-27T14-56-43.json` | `4e3a6748dccfdbe5ee75eb13d91f0ee536b9555fa973f639f14e18503f5189de` |
| MAB MH64K post-merge replay | `mab_fc_mh_64k_production_20260527T145646_2026-05-27T14-59-07.json` | `b76d78c35e3235a765a19e684e702372885b6faa8c637e29a80f830b3a1c0d47` |
| MAB MH262 high-water addendum | `mab_fc_mh_262k_production_20260526T231510_2026-05-26T23-18-34.json` | `ecfb205d8fb4c57020e44399dc4959a9ced230d04395c83f8be58fdfe9dc861a` |
| LoCoMo-Plus all401 summary | `locomo_plus_debug_launch_98a2d648_20260514T052519Z/summary.json` | `d36008ba07a605975f871bc5a1a0dea40cb2ef0458c544930aaafd266e3169f6` |
| LoCoMo-Plus all401 manifest | `locomo_plus_debug_launch_98a2d648_20260514T052519Z/manifest.json` | `a8b45aaadcaca63db4494b3314b2a5b0a82b6dce751c0cec82ab628daa4b9670` |
| LoCoMo-Plus SHA index | `locomo_plus_boundary_checked_score_adapter_postmerge_20260514/SHA256SUMS.txt` | `f3447df8bb62dea126e74570ab93a2891b9860aac9b5643fd8e09297bdcc1f61` |

## Claim Boundaries

These numbers are allowed only with their benchmark family, run class, and artifact reference. Do not combine them into one blended "Pith benchmark score."

The MAB scores are exact-match and F1 results for FactConsolidation variants. The LoCoMo-Plus score is a judge score for the official Cognitive all401 lane using the `pith-memory-adapter / boundary_checked_v1` answer path.

The launch evidence supports:

- narrow technical benchmark pages
- founder-led demos
- benchmark appendices with run IDs and hashes

It does not support:

- independent third-party reproduction claims
- industry-standard governance benchmark claims
- broad "best" or "highest-scoring" claims without a cited comparator packet
- any public CogGov-Bench score claim

## Reproduction Status

The score-bearing artifacts are archived and hash-indexed. Public clean-machine reproduction instructions are still being packaged, so use "artifact-verified" rather than "independently reproduced."

Before using any benchmark number in landing-page copy, ads, launch posts, or competitor comparisons, run a claim review against this file and the current public release claim ledger.
