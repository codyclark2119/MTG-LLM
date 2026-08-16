"""Constants shared across the pipeline.

Three kinds of value live here, each because a divergent copy has caused
or nearly caused a real bug:

**Prompts.** Section 8.7 traced a fine-tune failure to training examples
whose format did not match what inference sent. The prompts were later
copy-pasted verbatim into four scripts, and the only thing keeping them in
agreement was that nobody had edited one of them yet. Training data and
evaluation must be built from the *same* string, so there is now one.

**The CR version.** The Comprehensive Rules are pinned to a dated release.
That date was hardcoded in three argparse defaults and baked into a
filename, so updating the corpus meant finding every copy. A gold record
claiming `cr_version: 2026-08-07` while validated against a different file
is a silent correctness failure, and nothing checked it.

**Rule-id patterns.** Eight modules defined `CROSS_REF_RE`, under one name
with two incompatible meanings: six used a *search* pattern to pull ids out
of prose, two used an *anchored* pattern to validate a whole string. Both
are needed; sharing a name meant copying one into a file that wanted the
other silently changed behavior. They are named apart here.

Import as a plain module — scripts run as `python scripts/<name>.py`, so
their own directory is already on `sys.path`:

    from common import SYSTEM_PROMPT, CR_VERSION, RULE_ID_RE
"""

import json
import os
import re
from pathlib import Path

# --- Where the repository lives ---------------------------------------------
#
# Every canonical path below is anchored here rather than to the current
# working directory. They used to be bare relative paths, which meant every
# script silently required being run from the repo root: `python
# /path/to/scripts/card_lookup.py "Llanowar Elves"` from anywhere else died on
# `FileNotFoundError: data/cards/processed/card_chunks.jsonl`. Nothing said so,
# and the web console's runner only worked because it happened to inherit the
# right cwd.
#
# Paths a *user* supplies on the command line stay relative to their cwd, which
# is what anyone would expect.
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Comprehensive Rules pinning -------------------------------------------

# The CR release every derived artifact is validated against. Changing this
# means re-running ingest and re-validating the gold set; `cr_version` on a
# gold record refers to this value.
CR_VERSION = "2026-08-07"
CR_TEXT_PATH = REPO_ROOT / f"data/raw/MagicCompRules_{CR_VERSION.replace('-', '')}.txt"

# --- Rule-id patterns -------------------------------------------------------

# Find rule ids inside prose: "...as a state-based action (704.5g) and..."
RULE_ID_RE = re.compile(r"\b\d{3}\.\d+[a-z]?\b")
# Validate that a whole string is exactly one rule id, e.g. from a `rule_citations` list.
RULE_ID_EXACT_RE = re.compile(r"^\d{3}\.\d+[a-z]?$")

# --- Prompts ----------------------------------------------------------------
#
# CHANGING THESE INVALIDATES THE TRAINED ADAPTER. The v2 adapter was trained
# on RAG_SYSTEM_PROMPT with "Rules text:"-shaped context; an adapter is only
# valid for the format it saw. Edit these and you must retrain, or accept
# that the adapter is being evaluated out of distribution (Section 13.5
# measured exactly that failure for card-formatted context).

SYSTEM_PROMPT = (
    "You are a Magic: The Gathering rules expert. Answer precisely and "
    "cite comprehensive rule numbers."
)

RAG_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    " Use ONLY the provided rules text to answer; do not rely on outside knowledge."
)

CARDS_RAG_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    " Use ONLY the provided card text and rules text to answer; do not rely on "
    "outside knowledge. The card text is authoritative for what each named card does."
)

# --- Canonical data paths ---------------------------------------------------

RULES_PATH = REPO_ROOT / "data/processed/rules.jsonl"
GLOSSARY_PATH = REPO_ROOT / "data/processed/glossary.jsonl"
CHUNKS_PATH = REPO_ROOT / "data/processed/chunks.jsonl"
INDEX_PATH = REPO_ROOT / "data/processed/chunk_embeddings.npz"

ORACLE_CARDS_PATH = REPO_ROOT / "data/cards/raw/oracle_cards.jsonl"
CARD_CHUNKS_PATH = REPO_ROOT / "data/cards/processed/card_chunks.jsonl"
RULING_CHUNKS_PATH = REPO_ROOT / "data/cards/processed/ruling_chunks.jsonl"

RAW_RULINGS_PATH = REPO_ROOT / "data/cards/raw/rulings.jsonl"

GOLD_PATH = REPO_ROOT / "data/gold/gold_questions.jsonl"
POSITIONS_PATH = REPO_ROOT / "data/gold/positions.jsonl"
RULESGURU_SNAPSHOT = REPO_ROOT / "data/gold/rulesguru/questions.jsonl"
GOLD_CANDIDATES_PATH = REPO_ROOT / "data/gold/rulesguru/gold_candidates.jsonl"
# Anchored like the rest. `author_rubrics.py` had this as a bare relative
# Path, which Section 17.2's repo-root pass missed: run from anywhere else it
# wrote the worksheet under the *current* directory while printing the
# in-repo path, so a contributor's completed file could sit somewhere the
# ingest never looks.
WORKSHEETS_DIR = REPO_ROOT / "data/gold/worksheets"

PROCESSED_DIR = REPO_ROOT / "data/processed"
DATASETS_DIR = REPO_ROOT / "data/datasets"

# --- Gameplay -----------------------------------------------------------------
#
# A board position asks the model to DECIDE, not to explain, so it gets its own
# system prompt rather than reusing the rules-expert one. Note what that means
# for the adapter: this prompt and the rendered board below are shapes the v2/v3
# adapter has never seen in training, so the fine-tuned arms are evaluated out
# of distribution here by construction — the same effect Section 13.5 measured
# when card-formatted context met a rules-only adapter. Expect the adapter to
# underperform base on positions until position-shaped examples enter the SFT
# mix, and read a poor finetuned score as evidence about the training data
# rather than about the method.

# The grammar the model is told to speak. gameplay/actions.py implements the
# parser for it; gameplay/test_actions.py asserts the two agree, because a verb
# added here and not there would be a documented action the parser scores as
# malformed.
# NO SQUARE BRACKETS. The first version wrote optional operands as
# "CAST <card> [TARGET <a>, <b>]", and the model reproduced the brackets
# verbatim — "CAST Lightning Strike [ TARGET Grizzly Bears]" — which parses the
# card name as "Lightning Strike [" and fails to match any legal action. That
# turned a correct play into an illegal one and would have been read as the
# model naming actions that do not exist. Optional forms get their own line
# instead, so there is no meta-syntax left to copy.
ACTION_GRAMMAR = (
    "PLAY <card>                       play a land\n"
    "CAST <card>                       cast a spell that has no targets\n"
    "CAST <card> TARGET <a>, <b>       cast a spell, naming its targets\n"
    "ACTIVATE <permanent>: <ability>   activate an ability\n"
    "ATTACK <creature>, <creature>     declare all attackers at once\n"
    "BLOCK <blocker> -> <attacker>     one assignment per line\n"
    "ORDER TRIGGERS <a>, <b>           in the order they should resolve\n"
    "MULLIGAN                          ship the opening hand\n"
    "KEEP                              keep the opening hand as it is\n"
    "KEEP BOTTOM <a>, <b>              keep, naming the cards put on the bottom\n"
    "PASS                              take no further action"
)

GAMEPLAY_SYSTEM_PROMPT = (
    "You are piloting a Magic: The Gathering deck. You are given a board "
    "position and must decide what to do.\n\n"
    "Answer with the actions you take, ONE PER LINE, using exactly this "
    f"grammar:\n\n{ACTION_GRAMMAR}\n\n"
    "Write card names exactly as the position shows them, with no brackets or "
    "quotation marks around them. Give your reasoning first if you want to, "
    "then the actions. End with a single PASS."
)


def build_rag_messages(question: str, context: str | None = None,
                       preformatted: bool = False) -> list[dict]:
    """The single definition of a prompt's shape, for training AND inference.

    Section 8.7's fine-tune failure was a train/inference mismatch: the model
    was trained on bare question→answer while inference wrapped the question
    in retrieved rules text, so it had never seen an example where the answer
    was supposed to come from the prompt. The fix was to make the two match —
    but the two constructions then lived in different files, agreeing only
    because `build_grounded_messages` carried a docstring asking the next
    editor to keep them "byte-identical in shape".

    An invariant that important should not be a comment. Both callers build
    their messages here, so they cannot drift.

    `preformatted` marks context that already carries its own section headers
    (retrieve_hybrid emits "Cards referenced:" / "Rules text:"). Sniffing for
    the prefix instead of passing this explicitly once double-labeled every
    no-card prompt as "Rules text:\\nRules text:\\n...".
    """
    if not context:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    body = context if preformatted else f"Rules text:\n{context}"
    # The rules-only prompt says "use ONLY the provided rules text", which
    # would instruct the model to ignore card text handed to it in the same
    # turn — the unsatisfiable instruction Section 9.5 flagged.
    system = CARDS_RAG_SYSTEM_PROMPT if "Cards referenced:" in context else RAG_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{body}\n\nQuestion: {question}"},
    ]


def _render_permanent(p: dict) -> str:
    """One battlefield entry: name, current P/T, counters, auras, tap state."""
    bits = [p["card"]]
    if p.get("pt"):
        # Author-supplied CURRENT power/toughness, after counters and pumps.
        # Not derived from Oracle, because a 2/2 with a +1/+1 counter is a 3/3
        # and combat math is an entire position category.
        bits.append(p["pt"])
    notes = []
    for kind, n in (p.get("counters") or {}).items():
        notes.append(f"{n} {kind} counter{'s' if n != 1 else ''}")
    if p.get("attachments"):
        notes.append("attached: " + ", ".join(p["attachments"]))
    if p.get("summoning_sick"):
        notes.append("summoning sick")
    if notes:
        bits.append("(" + "; ".join(notes) + ")")
    bits.append("— tapped" if p.get("tapped") else "— untapped")
    # Combat status is a property of the permanent, not a stack object: an
    # attacking creature is on the battlefield with the attacking status (506.3),
    # it is not "on the stack". Modelling it as a stack entry renders a false
    # board to the model.
    if p.get("attacking"):
        bits.append("** ATTACKING **")
    if p.get("blocking"):
        bits.append(f"** BLOCKING {p['blocking']} **")
    return " ".join(bits)


def _render_zone(items: list, indent: str = "  ") -> list[str]:
    """Render a zone, collapsing identical entries into 'Mountain x4'.

    Lands repeat, and spelling out four identical Mountains costs tokens that
    matter: a rendered position plus card text plus rules chunks already runs
    close to the 2048-token training window (see configs/phase1_lora_v3.yaml).
    """
    if not items:
        return [f"{indent}(empty)"]
    counts: dict[str, int] = {}
    for line in items:
        counts[line] = counts.get(line, 0) + 1
    return [f"{indent}{line}" + (f" x{n}" if n > 1 else "") for line, n in counts.items()]


def render_position(pos: dict) -> str:
    """Render a structured position into the text the model actually sees.

    Derived, never authored. Storing the rendered text would let a renderer
    improvement apply only to positions written after it — the same trap
    Section 8.7 hit when the prompt's shape lived in two files. Every position
    renders through here, so a change applies retroactively to all of them.
    """
    you = pos["players"]["you"]
    opp = pos["players"]["opp"]
    active = pos.get("active_player", "you")
    whose = "your" if active == "you" else "opponent's"

    lines = [f"=== Turn {pos['turn']} — {whose} {pos['phase']} ==="]
    lines.append(
        f"You: {you['life']} life, {len(you.get('hand') or [])} cards in hand, "
        f"{you.get('library_count', '?')} in library"
    )
    lines.append(
        f"Opponent: {opp['life']} life, {opp.get('hand_count', '?')} cards in hand, "
        f"{opp.get('library_count', '?')} in library"
    )

    for side, label in (("you", "Your battlefield"), ("opp", "Opponent's battlefield")):
        perms = [_render_permanent(p) for p in pos.get("battlefield", [])
                 if p.get("controller") == side]
        lines.append(f"\n{label}:")
        lines.extend(_render_zone(perms))

    lines.append("\nYour hand:")
    lines.extend(_render_zone(list(you.get("hand") or [])))

    for side, label in (("you", "Your graveyard"), ("opp", "Opponent's graveyard")):
        gy = list((pos.get("graveyards") or {}).get(side) or [])
        if gy:
            lines.append(f"\n{label}:")
            lines.extend(_render_zone(gy))

    for side, label in (("you", "Your exile"), ("opp", "Opponent's exile")):
        ex = list((pos.get("exile") or {}).get(side) or [])
        if ex:
            lines.append(f"\n{label}:")
            lines.extend(_render_zone(ex))

    stack = pos.get("stack") or []
    if stack:
        lines.append("\nStack (top last):")
        lines.extend(f"  {s}" for s in stack)

    if pos.get("mana_available"):
        lines.append(f"\nMana available to you: {pos['mana_available']}")

    if pos.get("known_information"):
        lines.append("\nKnown:")
        lines.extend(f"  {k}" for k in pos["known_information"])

    if pos.get("deck_note"):
        lines.append(f"\nYour deck: {pos['deck_note']}")

    lines.append(
        f"\n{'You have' if pos.get('priority', 'you') == 'you' else 'Your opponent has'} priority."
    )
    return "\n".join(lines)


def build_position_messages(pos: dict, context: str | None = None,
                            closed: bool = False) -> list[dict]:
    """Prompt for a board position — the gameplay analogue of build_rag_messages.

    `closed` shows the enumerated legal actions and asks the model to choose
    among them, turning open-ended generation into ranking. `open` (the default)
    withholds the list. The gap between the two arms separates not knowing what
    is possible from not knowing what is good.
    """
    board = render_position(pos)
    parts = []
    if context:
        parts.append(context)
    parts.append(f"Position:\n{board}")

    if closed:
        legal = pos.get("legal_actions") or []
        if not legal:
            raise ValueError(f"position {pos.get('id')} has no legal_actions for the closed arm")
        listing = "\n".join(f"  {a}" for a in legal)
        parts.append(
            "These are the only legal actions available to you. Choose from "
            f"this list and reply with the ones you take:\n{listing}"
        )
        parts.append("Which do you take?")
    else:
        parts.append("What do you do?")

    system = GAMEPLAY_SYSTEM_PROMPT
    if context and "Cards referenced:" in context:
        system += (
            "\n\nThe card text provided is authoritative for what each card does."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def pearson_r(pairs: list[tuple[float, float]]) -> float:
    """Correlation over (judge A, judge B) score pairs. NaN under 3 pairs.

    One definition: `eval.py --compare` and `eval_positions.compare_judges`
    both report inter-judge correlation, and this is the number Section 14.6's
    +0.30 -> +0.62 result is stated in. Two copies of it would be the
    duplicated-helper trap on the project's headline statistic.
    """
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


HAND_AUTHORED_PREFIX = "hand-authored"


# Verbs a PLAYER does. A capitalized word immediately in front of one of these
# is a player; a capitalized word anywhere else is Magic vocabulary — a card,
# a creature type, a keyword — and none of our business.
_PLAYER_VERBS = (
    "controls control casts cast has have had plays play played attacks attack "
    "blocks block targets target draws draw discards discard sacrifices sacrifice "
    "activates activate taps tap owns own gains gain loses lose wants want "
    "responds respond chooses choose declares declare puts put moves move "
    "exiles exile destroys destroy counters counter reveals reveal wins win"
).split()
_PLAYER_NAME = re.compile(
    r"\b([A-Z][a-z]{2,})\s+(?:" + "|".join(_PLAYER_VERBS) + r")\b")


def stray_names(question: str, answer: str, lines: list[str]) -> list[str]:
    """Player names a rubric uses that its question never introduces.

    RulesGuru re-randomizes player names per request, which is why the
    snapshot is frozen — and it means the names in any one question are
    arbitrary. Writing a rubric against them is easy to get subtly wrong:
    two named players, and the rubric attributes the action to the other one,
    or to a name carried over from the question before it.

    Matching on capitalization alone does NOT work, and two attempts proved
    it in both directions. Magic capitalizes ordinary vocabulary, so checking
    against the question flagged "Swamps" on a correct rubric; widening the
    reference to include the verified answer fixed that but then flagged
    "Elemental" on a rubric whose only sin was naming the token's creature
    type — which is exactly the added precision a good rubric is supposed to
    have. A check that punishes sharper writing is worse than no check.

    So this matches on GRAMMAR instead: a capitalized word directly in front
    of a verb only a player performs. "Bianca controls no Swamps" matches;
    "One 1/1 red Elemental creature token is created" does not, because
    Elemental is followed by a noun.

    Reported, never blocked. Missing a stray name phrased some other way is
    the acceptable failure; nagging about correct card vocabulary is not.
    """
    known = set(_PLAYER_NAME.findall(f"{question or ''} {answer or ''}"))
    seen: list[str] = []
    for line in lines or []:
        for name in _PLAYER_NAME.findall(line or ""):
            if name not in known and name not in seen:
                seen.append(name)
    return seen


def is_hand_authored(record: dict) -> bool:
    """True only when a person wrote this rubric.

    `rubric_source` is prose, and the predicate used to be `bool(...)` — is
    the field set at all. That silently counted 20 of 39 gold records as
    finished whose provenance string reads, in full, *"assistant-authored
    from the RulesGuru-verified answer; not judge-reviewed"*. Non-empty was
    standing in for hand-written, and the two agree only until something
    machine-generated starts filling the field in — which is exactly what
    `rulesguru_to_gold.py` does.

    The cost was invisible rather than loud: `--fix-gold` and the console's
    work queue both skipped those records, so the highest-leverage work in
    the project (Section 14.6: hand rubrics take inter-judge agreement from
    r = +0.30 to +0.62) was hidden from the two tools whose job is to surface
    it. Both callers now use this one definition.

    Writers of the field: `label_store` (two paths), `webui`, and
    `author_rubrics --ingest` all prefix "hand-authored".
    """
    return str(record.get("rubric_source") or "").startswith(HAND_AUTHORED_PREFIX)


def read_jsonl(path: Path, missing_ok: bool = True) -> list[dict]:
    """Read a .jsonl file, skipping blank lines.

    There were five near-identical copies of this (three named `load_jsonl`,
    plus `read_jsonl` and `load_positions`) and eight more inline
    comprehensions, and they did NOT agree: three of them omitted the
    `if line.strip()` guard, so a trailing blank line — exactly what appending
    by hand or with a text editor leaves behind — raised
    `JSONDecodeError: Expecting value: line 2 column 1`.

    That is the same failure `load_rule_ids` was consolidated to fix in
    Section 15.3, reappearing one level up. One definition, so the guard cannot
    be missing from some callers and present in others.
    """
    if missing_ok and not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Write via a temp file in the same directory, then replace.

    A half-written gold file is worse than no gold file: validation fails on a
    truncated final line and the loss is hand-authored work. Lives here rather
    than in label_store so gameplay/positions.py can use it without a
    function-level `sys.path` insert to reach back into scripts/.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_rule_ids(rules_path: Path = RULES_PATH) -> set[str]:
    """Every rule id in the pinned CR, for validating citations.

    Six callers each had their own copy of this — two as functions, four as
    inline set comprehensions that would raise on a trailing newline.
    """
    return {r["rule_id"] for r in read_jsonl(rules_path, missing_ok=False)}


# --- "lead with the mistake" lint -------------------------------------------
# Lives here because it applies to BOTH kinds of hand-authored rubric: a board
# position (where common_errors is the blunder list) and a rules question. The
# defect it catches is a property of the wording, not of the record type, and
# the rubric server needs it without importing the gameplay package.

_STOP = {"a", "an", "the", "at", "to", "of", "on", "in", "with", "for", "and", "or",
         "it", "its", "is", "as", "by", "into", "then", "your", "their", "you",
         "instead", "rather", "than", "this", "that", "here", "which", "but"}


def _stem(word: str) -> str:
    """Crude suffix strip so 'casts' and 'cast' compare equal."""
    w = re.sub(r"[^a-z0-9/+-]", "", word.lower())
    for suffix in ("ing", "es", "ed", "s"):
        if len(w) > 4 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _content(text: str, limit: int | None = None) -> list[str]:
    words = [_stem(w) for w in text.split()]
    words = [w for w in words if w and w not in _STOP]
    return words[:limit] if limit else words


def lint_common_errors(pos: dict) -> list[str]:
    """Warn when a blunder's opening clause is also true of the correct line.

    Measured, not guessed (Section 16.12). Six of eight disputed judge calls on
    the seed set sat on two positions, and both had a `common_errors` line whose
    leading clause restated the correct play — the right line *is* to cast
    Lightning Strike, and the error read "Casts Lightning Strike at the
    opponent's face instead of...". A judge extracting claims can match the
    opening before it reaches the qualifier that makes the play wrong.

    Rewriting three such lines closed the two judges' blunder-rate gap from 28
    points to 6. It did not fix per-call disagreement, so this is a warning
    about a known bias, not a correctness check — the form shows it and still
    lets the position be saved.
    """
    # Deliberately CONSERVATIVE: it fires only when the error's opening words
    # appear as a contiguous run in the correct line, i.e. a verbatim restatement.
    #
    # A looser "do these words appear anywhere in the reference" version was
    # tried first and was wrong in both directions on the seed set — it missed
    # "Casts Lightning Strike at the opponent's face" (the case it was built
    # for, because the cap let non-matching words like "face" veto it) and fired
    # on "Plays Island" and "Attacks with Centaur Courser", neither of which the
    # judges ever disputed. A warning that is wrong both ways gets ignored, so
    # this one only claims the clearest form and stays silent otherwise.
    references = [_content(pos.get("answer", ""))]
    references += [_content(kp) for kp in (pos.get("key_points") or [])]
    references = [r for r in references if r]
    if not references:
        return []

    def restates(words: list[str]) -> bool:
        return any(
            any(ref[i:i + len(words)] == words for i in range(len(ref) - len(words) + 1))
            for ref in references
        )

    warnings = []
    for err in pos.get("common_errors") or []:
        lead = err.split(",")[0]
        words = _content(lead)
        raw = [w for w in lead.split() if _stem(w) in words]  # original spellings
        # Try the longest opening first, down to a two-word minimum — one word
        # ("blocks", "casts") is far too common to mean anything. The span must
        # also carry two real words: "takes 4" matched a correct line that
        # mentioned taking 4 damage, and that position drew no judge dispute.
        for n in range(min(4, len(words)), 1, -1):
            span = words[:n]
            if sum(1 for w in span if not w.isdigit()) < 2:
                continue
            if restates(span):
                shown = " ".join(raw[:n]) or " ".join(span)
                warnings.append(
                    f'"{shown}" restates the correct line, so a judge can match this '
                    f"error before reaching what makes the play wrong. Lead with the "
                    f"mistake instead — {err[:55]}...")
                break
    return warnings
