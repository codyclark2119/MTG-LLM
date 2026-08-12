# Gold set schema

Human-authored, judge-sourced rules Q&A. This is the highest-trust data in
the project: everything else is either machine-generated (synthetic set),
community prose (reddit set), or first-party but not question-shaped
(WotC rulings).

One JSON object per line in `data/gold/gold_questions.jsonl`. Validate with:

```bash
python scripts/validate_gold.py            # check
python scripts/validate_gold.py --to-eval  # emit eval/gold_questions.eval.jsonl
```

## Why a rubric instead of one reference answer

Sections 9.6–9.9 measured what goes wrong when scoring against a single
prose answer: the judge rewarded length (r = +0.21), and two independent
judges agreed only r = +0.43 with a mean disagreement of 1.18 points on a
1–5 scale. Both problems come from asking "how similar is this to my one
phrasing?" — a question with no objective answer.

`key_points` and `common_errors` replace that with something checkable:
*did the response assert these facts, and did it avoid these mistakes?*
That is length-neutral by construction (a one-line answer hitting every
key point scores full marks) and far more reproducible across judges.

## Fields

| Field | Required | Notes |
|---|---|---|
| `id` | yes | stable slug, e.g. `sba-lethal-damage-01`. Never reuse. |
| `question` | yes | the question as a **player** would ask it — natural, first-person, messy is fine |
| `paraphrases` | no | other natural phrasings of the *same* question; used to test robustness to wording (see below) |
| `answer` | yes | the ruling in the judge's own words, 1–4 sentences |
| `key_points` | yes | list of assertions a correct answer **must** contain. Scoring unit. |
| `common_errors` | no | wrong beliefs a response must **not** assert. Powerful — encodes what players actually get wrong |
| `rule_citations` | yes | CR rule IDs supporting the ruling, e.g. `["704.5g"]`. Validated against the pinned CR |
| `cards` | no | card names referenced. Validated against the Oracle pool |
| `category` | yes | one of the Section 7.1 categories (validator lists them) |
| `difficulty` | yes | `basic` \| `intermediate` \| `advanced` |
| `format_context` | no | e.g. `commander`, `two-player`, `limited` — some rulings differ by format |
| `source` | yes | `judge:<name-or-initials>`, `jr:<judge-rulings-url>`, `wotc-ruling`, `cr-derived` |
| `verified_by` | no | second judge who confirmed it — mark disputed rulings clearly |
| `cr_version` | yes | CR effective date this was checked against, e.g. `2026-08-07` |
| `notes` | no | edge cases, why players get it wrong, related interactions |

### Fields for scenario-derived records

Judge Simulation Worksheet scenarios (`data/judge_worksheets/scenarios.jsonl`)
are Competitive-REL tournament situations. Their *policy* answer (GRV, HCE,
backups) lives in the IPG/MTR, which is **not** in this project's corpus —
verified directly: the CR contains zero occurrences of "Game Rule Violation"
or "Competitive REL". But each scenario is built on a Comprehensive Rules
question that *is* in scope, and those are worth extracting because they are
real table situations rather than questions generated from rules text.

| Field | Required | Notes |
|---|---|---|
| `scenario` | no | the full table situation, when the question only makes sense with it |
| `worksheet_id` | no | provenance, e.g. `jsw-58` |
| `policy_note` | no | the infraction/fix, parked verbatim for when IPG/MTR enters the corpus. Never scored against the CR-only system |

A worksheet scenario is **seed material, not a gold record** — the rules
question inside it is implicit, and making it explicit is the judgement
step. See `data/gold/worked_examples.jsonl` for drafts.

## Keeping the language natural

The synthetic set failed partly because every question was generated from
rules text, so it *read* like rules text and never resembled how players
actually write. Two things to preserve here:

- **Write `question` the way a player asks it**, not the way a rules
  document states it. "my opponent blocked and then flashed in a creature,
  does it deal damage?" is more useful than "describe the timing of
  blocker declaration relative to instant-speed permanent entry."
- **Use `paraphrases` for wording variation.** Store one canonical
  `question` plus 2–4 natural rewordings of the *same* underlying ruling —
  terse, verbose, misspelled, jargon-heavy, beginner-worded. The rubric is
  shared across all of them, so the same ruling can be tested under many
  phrasings without duplicating judgement work. This directly measures
  whether the model tracks meaning or surface form.

## Example

```json
{
  "id": "sba-deathtouch-lethal-01",
  "question": "my 5/5 got hit by a 1/1 deathtouch creature. it only did 1 damage so does my creature actually die?",
  "paraphrases": [
    "Does deathtouch kill a creature even if the damage is way less than its toughness?",
    "1 damage from a deathtoucher vs a 5/5 — dead or not?"
  ],
  "answer": "Yes. Any nonzero damage from a source with deathtouch counts as lethal, so the 5/5 is destroyed the next time state-based actions are checked.",
  "key_points": [
    "the creature is destroyed",
    "any nonzero amount of deathtouch damage counts as lethal damage",
    "this happens as a state-based action, checked when a player would next receive priority"
  ],
  "common_errors": [
    "claiming the creature survives because 1 < 5 toughness",
    "claiming deathtouch destroys immediately rather than via a state-based action"
  ],
  "rule_citations": ["704.5g", "702.2b"],
  "cards": [],
  "category": "state-based actions",
  "difficulty": "basic",
  "source": "judge:LX",
  "cr_version": "2026-08-07",
  "notes": "Players often expect damage to compare against toughness numerically; deathtouch replaces that comparison."
}
```

## Collecting from judges

Judges should not hand-write JSONL. Have them fill a spreadsheet or a
plain markdown table with these columns and convert it — one column per
field above, `key_points` / `common_errors` / `rule_citations` as
semicolon-separated lists. `scripts/validate_gold.py` reports exactly
which rows fail and why, so the round-trip with a non-technical
contributor stays short.

## Coverage targets

The current eval is 54% definition recall, with one turn-structure
question and two state-based-action questions — per-category conclusions
are impossible at that size. Aim for a **minimum of 8–10 per category**
across the eight Section 7.1 categories before adding depth anywhere, and
skew `difficulty` toward `intermediate`/`advanced`: `basic` definition
recall is the one thing the existing corpora already cover well.
