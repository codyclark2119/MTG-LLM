# Reddit r/MTGRules eval set

- Source: Javier-Jimenez99/reddit-mtgrules-qa (Hugging Face, CC-BY-SA-4.0), itself scraped from Reddit's r/MTGRules
- Filter: score >= 3, prompt >= 30 chars, response >= 20 chars, LLM-filtered for genuine on-topic rulings (Reddit score alone rewards jokes as much as accuracy — see script docstring), evenly sampled down to 100 records
- License: CC-BY-SA-4.0 — attribution required, derivatives must share-alike
- These are real community answers, not verified by an actual Magic rules judge — correctness is not guaranteed even after LLM filtering. Explicit rule-number citations in the original answers are checked against the currently-pinned CR (2026-08-07) and flagged in `cited_rule_ids_stale` when they don't resolve — Reddit posts span many CR revisions, so a cited number may have since been renumbered or reused. 1/100 records have at least one stale citation.
- `retrieved_rule_ids`: best-effort RAG grounding against the current chunk index (Section 6), for cross-checking the community answer against current rules text — not a guarantee of correctness either.
