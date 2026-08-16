# Board positions

A position is a gold record whose question is a **board** instead of a sentence.
It asks the model to decide, not to explain. Everything in [SCHEMA.md](SCHEMA.md)
about rubrics still applies — this covers only what is different.

Positions live in `data/gold/positions.jsonl`, **not** in `gold_questions.jsonl`.
The two files are kept apart on purpose: `label_store.CATEGORIES` drives
stratified sampling and validation for the rules gold set, and an in-flight
n≈100 two-judge comparison depends on that set's composition.

Author them at `#/position` in the console (`python scripts/webui.py --lan`).
The right pane renders exactly what the model will see, live.

---

## The one thing to get right

`common_errors` **is** the blunder list. Blunder rate — the whole Gate 3 metric —
is just how often a judge says an answer committed one:

```
blundered  ==  errors_made is non-empty
```

That is the same `rubric_correctness()` that scores rules questions, unchanged.
So a position with no `common_errors` cannot contribute to the gate, and the
validator rejects it.

### Lead with the mistake, not with the play

Measured, in Section 16.12. Six of eight disputed judge calls landed on two of
eight positions, and both had the same defect: a `common_errors` line whose
**opening clause was also true of the correct line.**

| | |
| --- | --- |
| correct line | "**Cast Lightning Strike** on Grizzly Bears, then attack." |
| bad error | "**Casts Lightning Strike** at the opponent's face instead of removing the blocker" |
| good error | "**Leaves Grizzly Bears alive**, pointing Lightning Strike at the opponent instead" |

A judge extracting claims matches the opening and fires before it reaches the
qualifier that makes the play wrong. Rewriting three lines this way closed the
two judges' blunder-rate gap from **28 points to 6**.

It did *not* fix per-call disagreement (9 disputes → 8), so this removes a
systematic bias rather than making the metric reliable. Both halves matter.

The form warns when it catches this, but the check is deliberately conservative
— it only fires on a **verbatim** restatement and will miss subtler overlap. A
clean form means "no obvious defect", not "good rubric".

### An error must be a play someone would actually make

`common_errors` are traps, not negations of the key points. "Fails to cast
Lightning Strike" is a negation. "Attacks into the untapped blocker" is a trap.

---

## Writing the board

One permanent per line. Suffixes, in any order:

```
Mountain x3                          three untapped Mountains
Grizzly Bears 2/2                    current power/toughness
Serra Angel 4/4 tapped attacking     combat status is a permanent's status
Llanowar Elves 1/1 sick              summoning sick
```

`2/2` is the **current** P/T, after counters and pumps — a 2/2 with a +1/+1
counter is a `3/3`. Combat math is an entire category, so this is the number
the model reasons from.

Card names are validated against Oracle and must match **exactly**; the form
suggests the exact name when it can resolve yours.

**Attacking creatures are not stack entries.** An attacking creature is a
battlefield permanent with the attacking status (506.3). The first draft of the
seed fixtures modelled them as `stack` items and the Oracle name check caught it
by rejecting "Serra Angel is attacking you" as a card that does not exist.

### legal_actions

One per line, in the action grammar (`scripts/gameplay/actions.py`). Every entry
is parsed and a malformed one is rejected, because the closed arm matches model
output against this list — an entry that cannot parse can never be matched, so
every answer naming it would be scored illegal.

**Do not list `PASS`.** It is always accepted, listed or not. The system prompt
mandates a trailing `PASS` on every answer, so it is a protocol terminator
rather than a play you are offering — and scoring it as a play failed Gate 1 on
the one position where passing priority is not a legal game action at all, a
mulligan decision (Section 16.13). List only the plays that are genuinely
available.

Supplying them is optional but worth it: the closed arm is the measurement of
whether the model's problem is *not knowing what is possible* or *not knowing
what is good*. So far it is the second.

---

## Categories

`mulligan`, `land sequencing`, `combat math`, `blocking`, `removal timing`,
`trigger ordering`, `race vs stabilize`.

`trigger ordering` has no seed fixture — a position where trigger order actually
changes the outcome needs a real card interaction, and a contrived one is worse
than none. It is the position analogue of `definition recall` in the rules set.

---

## Workflow

```bash
# author in the console, then:
python scripts/gameplay/positions.py                     # validate everything
python scripts/gameplay/eval_positions.py                # scores with TWO judges
```

**Run both judges from the first position, not at the end.** Gates 2 and 3 both
reversed between judges on byte-identical answers. The gate is specified as
"≤25% blunder rate holding under both judges", and the second half of that
sentence is the load-bearing part.

Then read `positions_seed_agreement.md`:

- **Cohen's kappa** on the blunder call, not raw agreement — two judges each
  blundering 60% of the time agree half the time by chance.
- **The disputed-call list.** Treat a position the judges split on as a
  **position that needs rewriting**, the same way Section 14.6 treated a
  machine-drafted rubric. Disagreement localizes: on the seed set, six of eight
  disputes sat on two positions and six positions drew none at all.

The judge is deterministic — re-judging identical answers reproduced 24/24 calls
exactly — so a number that changes means something real changed, never noise.

---

## Sizing

**n ≈ 40**, not the n ≈ 100 the rules comparison needs. Blunder rate is a
proportion and the expected effects are large: separating 60% from 30% needs
~42 per group unpaired, and this design is paired.

That is still 8–12 hours, because a position costs far more to author than a
rubric. Gates 1 and 2 do not wait for 40 — they are measurable at n ≈ 8.

---

## Seed fixtures

`data/gold/positions_seed.jsonl` holds 8 machine-drafted positions that verify
the renderer, parser, validator and scoring path end to end. They carry a
`seed_note` field and the report warns whenever one is present in a scored run.

They are **not** gate evidence. Section 14.6 measured hand-authored rubrics
lifting inter-judge agreement from r = +0.30 to +0.62, and there is no reason
positions are exempt. Regenerate them with
`python scripts/gameplay/make_seed_positions.py`.
