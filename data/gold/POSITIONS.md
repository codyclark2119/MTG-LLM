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

### Write the false CLAIM, not the player's behaviour

**This supersedes the "lead with the mistake" rule below** (Section 21.35). Both
are about the same failure and this one is more direct.

The judge is asked which of these the candidate *asserted*. `key_points` are
claims, so that question is answerable — the judge credits the reference answer
with 90% of its own key points. `common_errors` were written as behaviours
("Adds Centaur Courser to the block"), which turns the question into *did this
text describe a player doing that?* Measured cost: the 7B judge charged the
reference answer — which definitionally cannot commit an error — with one on
**40%** of questions, and blunder rate is defined on exactly that field.

| | |
| --- | --- |
| correct line | "Block with Sedge Scorpion alone." |
| behaviour (old) | "Adds Centaur Courser to the block, spending a 3/3 to save 3 life" |
| **claim (now)** | "**Centaur Courser should be added to the block alongside Sedge Scorpion**" |

The test: **could a wrong answer contain this sentence verbatim?** If not, it
describes a mistake instead of being one. `lint_common_errors` warns on the
third-person-verb opening that marks the old form.

A claim will often start with the same card as the correct line — both sentences
are *about* that card — and that is fine. What distinguishes them is the
predicate, which is always present. The restatement check below is gated to the
behaviour form for exactly this reason.

<details><summary>The superseded rule, kept because its measurement stands</summary>

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

Both examples are behaviour-form; under the claim rule the error becomes
"Lightning Strike should be aimed at the opponent", which has no opening clause
to match early because it has no qualifier.

</details>

It did *not* fix per-call disagreement (9 disputes → 8), so this removes a
systematic bias rather than making the metric reliable. Both halves matter.

The form warns when it catches this, but the check is deliberately conservative
— it only fires on a **verbatim** restatement and will miss subtler overlap. A
clean form means "no obvious defect", not "good rubric".

### If the line has two steps, stopping after the first is a blunder

Measured in Section 21.3, and it was the single largest source of judge
disagreement on the first real gate run.

**69% of model answers contain at most one real action.** Emitting one play and
passing is what these models *do*. So on a position whose correct line needs two
steps, the most likely wrong answer is the first half of the right one — and if
the blunder list has no entry for that, the answer commits no listed error and
scores as clean.

That is exactly what happened. On a board where the line was "Shock the blocker,
then attack for exactly lethal", the answer was:

```
CAST Shock TARGET Grizzly Bears
PASS
```

Half the line. Lethal left on an empty board. One judge fired no errors, which
is *correct by the rubric* and scores a thrown-away win as un-blundered; the
other fired all three to signal the answer was bad. Four of the five multi-step
positions had this gap and carried 11 of 31 disputed calls between them. The
fifth already had a partial-execution entry — and drew **zero** disputes.

So: **every multi-step line needs an entry for abandoning it halfway.** Where
the board is lethal, say so in those terms — leaving the winning step untaken
hands a beaten opponent another turn, which in play is a fatal blunder rather
than a missed optimisation. Where the board is *not* lethal, write the smaller
loss it actually is; borrowing severity makes the rubric describe a different
board.

Note this is the **inverse** of the defect above. That one is an entry whose
opening clause is also true of the correct line. This one is a real failure mode
with no entry at all, and the lint cannot see it — a missing line never trips a
check on the lines that are there.

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

### phase

**XMage's spelling, from `mage.constants.PhaseStep`** — mirrored in
`common.PHASE_STEPS`, which is the only list of steps in the project:

```
untap step            begin combat step        end of combat step
upkeep                declare attackers step   postcombat main step
draw step             declare blockers step    end turn step
precombat main step   first combat damage      cleanup step
                      combat damage step
```

Plus `opening hand`, which is not a step — a mulligan happens before the turn
begins, so XMage has no constant for it and neither does the CR.

The vocabulary was ours until Section 21.102 and had 9 spellings across 32
positions. Use `python scripts/gameplay/positions.py --canonicalize-phases
--dry-run` after any bulk import; it converts the CR's wording ("end step",
"beginning of combat") and refuses `main`, which names two steps.

**`legal_actions` is scoped to the TURN, not to this step.** A board at
`precombat main step` that offers `ATTACK Serra Angel` is correct: the answer is
expected to write `END PHASE precombat main step`, `PHASE declare attackers
step`, and then attack. That is what `PHASE` and `END PHASE` are in the grammar
for, and `check_reference` enforces it — a reference line's attack must appear
in `legal_actions` even though it cannot be made in the step the board states.

Reading the field the other way — one step, one list — made a rules engine
report eight positions as impossible when none of them were (21.101).

It is still worth asking whether a board *should* span two steps. A position
asks one question (21.93), and a main phase that exists only to be left is
scaffolding; that is what the `shorten` review flag is for. Impossible and
long-winded look identical in a diff and are not the same problem.

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
what is good*. So far it is the second. An empty list is valid only when the
position intentionally has no legal play; the closed prompt then states that
`PASS` is the only response. Do not use an empty list to hide an incomplete
position.

An empty list also changes what `PROTOCOL_ERRORS` entry 1 means. "Only passes"
is a blunder when a play was available and the *correct* answer when none was,
so `protocol_findings` does not charge entry 1 on a board whose `legal_actions`
is present and empty. A list that is *absent* is left alone — that is "not
stated", not "nothing is legal" (Section 21.85).

## `reference_actions` — the 100%-correct line

Optional. The ideal answer for this board in the action grammar, one action per
line, exactly as a perfect model would emit it:

```json
"reference_actions": ["PHASE Declare Attackers Step",
                      "TAP Forest FOR {G}", "CAST Ambush Viper", "PASS"]
```

Authored in the `#/adjudicate` form and promoted with:

```bash
python scripts/gameplay/positions.py --ingest-references submissions.jsonl --dry-run
```

**It is refused unless a parser agrees with it.** `check_reference` runs the
line through the same parser and the same `PROTOCOL_ERRORS` checks every arm
answer goes through: it must parse, every play must be in `legal_actions`, and
no decidable protocol entry may fire. `validate_position` re-checks stored
lines, so a reference that was clean when written and stops being clean when
the rubric grows — `PROTOCOL_ERRORS` is append-only — fails loudly instead of
quietly becoming wrong.

Same shape as a scenario step's `reference_actions` (`turns.py`) on purpose:
one name, one meaning.

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
# author in the console, or draft a batch and promote it:
python scripts/gameplay/positions.py --ingest drafts.jsonl --dry-run
python scripts/gameplay/positions.py --ingest drafts.jsonl --author "Your Name"

python scripts/gameplay/positions.py                     # validate everything
python scripts/gameplay/eval_positions.py --second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

`--ingest` refuses three things, each a measurement it would otherwise corrupt:
an id that already exists (a "rewrite" is a different board wearing an old id,
and every score filed under it would then describe a board that no longer
exists), a record carrying `seed_note`, and any batch where a single record
fails validation. `--dry-run` renders every incoming board, because a position
can only be reviewed as the model will see it.

**Comparing two runs requires the same arm count.** The judge grades every
candidate for a position in one batched call, so adding an arm changes every
other arm's score — measured in Section 21.5, where a byte-identical arm moved
23 points. Use `--arms` to match counts rather than ignoring a column.

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
