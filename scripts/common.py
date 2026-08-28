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

import hashlib
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

# The exact parse of that release. Pinned as a committed constant rather than
# written to a stamp file by ingest.py: a bad parse that regenerates the corpus
# AND its stamp together would pass a self-issued check and prove nothing. A
# constant fails for whoever ran it, and moving it is a one-line diff sitting
# beside CR_VERSION in review.
#
# `guard_shrink` already refuses a short parse and git already shows a
# committed corpus changing. This catches the two neither does:
#
#   * a parse that keeps the rule count and changes the text — a regex edit
#     that mangles content parses the same 3,162 rules, and would silently move
#     every citation-validity number in Section 9 without tripping anything;
#   * a local rebuild that diverges, is never committed, and is then evaluated
#     against.
#
# Re-pin deliberately with `python scripts/ingest.py --update-pin`. That should
# almost never be needed — the parse is reproducible, verified byte-for-byte.
CR_PIN = {
    "rules_sha256": "bbb5ef03db73e51b8092046e79412cf11ca0b8e4510df02ed823049a97a29ead",
    "glossary_sha256": "78cff7d85e6a26ae1ac89587fbf0c3f9c292f3608f7358ee24118419d598d75c",
    "n_rules": 3162,
    "n_glossary": 739,
}

# The same guarantee for the card corpora. Scryfall is a LIVE API: a re-fetch
# months later returns errata'd oracle text under the same card name and the
# same record count, which `guard_shrink` cannot see and git will not flag
# unless someone reads a 25MB diff.
#
# `card_chunks.jsonl` is pinned rather than `oracle_cards.jsonl` because it is
# what everything downstream actually reads — eleven call sites reach it through
# `CardIndex` — and it is the layer where a *chunker* change shows up too. Same
# reasoning as pinning `rules.jsonl` rather than the raw CR text.
#
# `ruling_chunks.jsonl` is pinned but currently READ BY NOTHING: it is ingested
# and never wired into retrieval. Pinned anyway, because the moment it is wired
# in is the moment nobody will think to pin it.
CARD_PIN = {
    "card_chunks_sha256": "407903e2d42442dbdb8c598d4269fe4f8c2349c07a2164c7d9b7cf438df240f7",
    "ruling_chunks_sha256": "1e3e0fd4b57d4e10796e6adaf7fd0fdbe3a3899c0fd10aec7b71745e2ee24c17",
    "n_card_chunks": 34933,
    "n_ruling_chunks": 19726,
    # WHICH snapshot, not just that it has not changed. The sha proves the bytes
    # are the ones every published number was computed against; it cannot say
    # what they are, and "our Oracle dump" is not an answer when Scryfall
    # rebuilds bulk data daily and errata cards in place (Section 21.110).
    #
    # `download_bulk` already RETURNS Scryfall's upstream `updated_at` and
    # `fetch_cards.py` already writes it into a per-file manifest — but the two
    # bulk files this corpus is actually built from were pulled before that code
    # existed, so no manifest was written for them and the upstream value is
    # unrecoverable now. Recovered instead from the files themselves:
    "oracle_bulk_downloaded_at": "2026-08-11T16:06Z",   # local mtime, not upstream
    "rulings_bulk_downloaded_at": "2026-08-12T08:23Z",  # local mtime, not upstream
    "oracle_bulk_updated_at": None,   # unknown; capture on the next re-pin
    # A lower bound anyone can re-derive: the newest `released_at` in the raw
    # bulk file. Future-dated because Scryfall carries spoiled sets before
    # release, so it bounds the snapshot from ABOVE in set coverage rather than
    # dating it — which is still the most useful single fact about the corpus.
    "newest_set_released_at": "2026-11-13",
    "n_raw_bulk_cards": 38626,        # before chunking drops non-playable rows
}

# The MTG Wiki gloss (scripts/fetch_wiki.py). Pinned for the same reason the
# others are, and more urgently: a wiki page is edited far more often than the
# CR is published, so a re-fetch silently changing text under a published number
# is likelier here than anywhere else in this repo.
#
# UNOFFICIAL. Community-edited prose, never a citation source — every record
# carries `authority: "unofficial"` and its page revision. CC BY-NC-SA 2.5.
WIKI_PIN = {
    "wiki_chunks_sha256": "c866bf0b7c39829559398ae2f3b1328d6c8bed9d2f3c577d9479f4addfe361da",
    "n_wiki_chunks": 141,
    "n_pages": 46,
    "source": "mtg.fandom.com Portal:Rules",
    "license": "CC BY-NC-SA 2.5",
}

# Errors any position can contain, regardless of what it is testing.
#
# `common_errors` on a position enumerate *strategy* blunders — playing the
# wrong card, blocking the wrong creature. Section 21.49 measured that 81% of
# adjudicated answers were bad for a reason none of them describes: the model
# does nothing, loops, plays something unavailable, or misstates the step. The
# judge cannot charge an error that is not listed, so it charged a neighbouring
# one — precision 3%, recall 100%, zero misses (21.65).
#
# These six close that gap. They are position-INDEPENDENT, so they are appended
# to every rubric rather than authored per board, and each one is **decidable by
# a parser**: when the judge charges one, a checker can confirm or refute it
# without a human and without a second judge. That is what makes them different
# in kind from every other rubric entry here, and it is the first per-error
# precision measurement this project can make on its own (Section 21.70).
#
# ORDER IS PART OF THE DATA. The judge returns error NUMBERS, so reordering or
# inserting renumbers every stored verdict. Append only.
# WORDED TO MATCH THE CHECKER, not to read naturally. The first draft said "the
# answer takes no action at all — it only passes", and the judge was shown
# `PHASE Declare Blockers / TAP Forest FOR {G}` x4 / `PASS`. It correctly
# declined to charge an answer with four visible actions in it. The checker
# meant "no PLAY", counting declarations as not-plays; the wording said
# "no action", and an answer full of taps contradicts that on its face.
#
# So a correct judge and a correct checker disagreed, and the number came out as
# 42% recall on the most obvious class in the list. Every entry now names the
# grammar it is talking about — plays versus TAP and PHASE lines — because the
# verbose grammar (21.60) is exactly what made "did nothing" look busy
# (Section 21.72).
PROTOCOL_ERRORS = (
    "The answer makes no play: it only passes, possibly after PHASE or TAP "
    "lines. Declaring a phase and tapping lands are not plays.",
    "The answer repeats the same PLAY over and over instead of playing a line. "
    "Repeated TAP lines are not this error — tapping several lands is normal.",
    "The answer names a play that is not available in this position. Charge this "
    "if ANY line is unavailable, even when the other lines are correct.",
    "The answer's PHASE line names a step other than the one the position is in.",
    "The answer's TAP lines do not add up to the cost of the spells it casts — "
    "it is short.",
    "The answer's TAP lines add up to MORE than the spells it casts require, "
    "floating mana for nothing. Only charge this when it actually casts something.",
    "The answer casts a permanent that is already on the battlefield rather than "
    "one in hand.",
    # 8-10 appended in Section 21.83, after each was measured on stored answers
    # and found in reviewer notes. Appended, never inserted: the judge returns
    # error NUMBERS, so 1-7 must keep their meaning or every verdict ever
    # collected silently changes what it says.
    "The answer casts a CREATURE spell with a TARGET. Creature spells do not "
    "target on cast. Do not charge this for a spell whose text says \"target\", "
    "nor for an Aura.",
    "The answer declares TAP lines and then casts nothing at all, so the mana is "
    "wasted. This is not entry 6, which is about over-paying for a spell that "
    "WAS cast; charge this only when no spell is cast.",
    "The answer makes a play without any PHASE line, so it never says which step "
    "it is acting in. Do not charge this when the only action is PASS.",
)

# Which of the above make a turn INVALID versus merely bad. A reviewer drew the
# line: under-tapping means the spell cannot be cast at all, while over-tapping
# is a legal play a real player makes and should still be discouraged. Folding
# both into one entry asked the judge to charge two different mistakes with one
# number, and asked the parser to confirm a charge that could be true for either
# reason.
#
# The distinction is the gate's, not the judge's: the target at this stage is a
# turn that is completely VALID, not a turn that is optimal. Gate 1 is about
# validity and these are its vocabulary; strategy errors are about optimality
# and stay where they are (Section 21.70).
#
# 8-10 follow the same line (Section 21.83):
#   8  a creature spell cast with a target is not a legal action at all -> INVALID
#   9  wasted mana is legal and merely bad, exactly like 6            -> not invalid
#  10  a missing PHASE line is an unverifiable claim, not an illegal one. Entry 4
#      (a WRONG phase) invalidates because the play provably happened at the
#      wrong time; silence proves nothing, and a turn cannot be called invalid
#      because the player failed to narrate it                        -> not invalid
PROTOCOL_INVALIDATING = (1, 2, 3, 4, 5, 7, 8)  # 6, 9 legal but wasteful; 10 unverifiable

# --- Rule-id patterns -------------------------------------------------------

# Find rule ids inside prose: "...as a state-based action (704.5g) and..."
RULE_ID_RE = re.compile(r"\b\d{3}\.\d+[a-z]?\b")
# Validate that a whole string is exactly one rule id, e.g. from a `rule_citations` list.
RULE_ID_EXACT_RE = re.compile(r"^\d{3}\.\d+[a-z]?$")

# --- Refusal-shaped answers -------------------------------------------------
#
# Section 21.6 measured 18% of the synthetic SFT set as refusals — the adapter
# was explicitly trained to decline one example in five, which is a sufficient
# explanation on its own for the fine-tuned arm scoring lowest on correctness in
# every run since Section 9.
#
# Here rather than in the builder because `audit_sft.py` reports the same number
# for a dataset the builder did not write, and two copies of this pattern would
# be the duplicated-helper trap on the statistic that condemned run 3.
REFUSAL_RE = re.compile(
    r"do(es)? not (contain|describe|provide|mention|specify)"
    r"|cannot (answer|determine|be answered)"
    r"|not (enough|sufficient) information"
    r"|unable to (answer|determine)",
    re.I,
)

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
WIKI_CHUNKS_PATH = REPO_ROOT / "data/processed/wiki_chunks.jsonl"
RULING_CHUNKS_PATH = REPO_ROOT / "data/cards/processed/ruling_chunks.jsonl"

RAW_RULINGS_PATH = REPO_ROOT / "data/cards/raw/rulings.jsonl"

GOLD_PATH = REPO_ROOT / "data/gold/gold_questions.jsonl"
POSITIONS_PATH = REPO_ROOT / "data/gold/positions.jsonl"
RULESGURU_SNAPSHOT = REPO_ROOT / "data/gold/rulesguru/questions.jsonl"
GOLD_CANDIDATES_PATH = REPO_ROOT / "data/gold/rulesguru/gold_candidates.jsonl"
# Definition-recall candidates derived from the CR glossary. A separate file
# because RulesGuru is a *scenario* database — "What does X mean?" never
# appears in it, so `definition recall` has zero candidates there — and because
# the RulesGuru snapshot is frozen and additive (re-fetching re-randomizes
# names and cards). Mixing a second source into it would blur that guarantee.
GLOSSARY_CANDIDATES_PATH = REPO_ROOT / "data/gold/glossary_candidates.jsonl"
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
# --- Closed vocabularies -----------------------------------------------------
#
# Every list of allowed values lives HERE and nowhere else. Four of them were
# previously defined in two or three modules apiece — `DIFFICULTIES` in three,
# `CATEGORIES` in two (as a list and as a set), and two more mirrored into this
# file so the deployed form could reach them. All four happened to agree when
# anyone last looked, which is the only reason none of them had bitten yet.
#
# `common.py` is the right home for a reason beyond tidiness: `rubric_server.py`
# ships with only this module, so anything the public form validates against had
# to be here regardless. Everything else re-exports from here (Section 21.97).
#
# These are DATA, so keeping them here costs the pure-stdlib rule nothing.

# The rules gold set. `label_store` drives stratified sampling from this, and an
# in-flight n~100 comparison depends on its composition — see the note in
# `gameplay.positions` on why the gameplay categories are deliberately separate.
CATEGORIES = (
    "definition recall", "turn-structure walkthrough", "priority reasoning",
    "interaction puzzle", "state-based actions", "zone transition",
    "layer-system question", "templating/keyword meaning",
)

DIFFICULTIES = ("basic", "intermediate", "advanced")

# Board positions. Kept apart from CATEGORIES above on purpose: mixing gameplay
# categories into the rules set would change what stratified sampling draws.
POSITION_CATEGORIES = (
    "mulligan", "land sequencing", "combat math", "blocking",
    "removal timing", "trigger ordering", "race vs stabilize",
    "payment", "closing the turn",
)

# --- Steps -----------------------------------------------------------------
#
# MIRRORS `mage.constants.PhaseStep`, read from a magefree/mage checkout rather
# than invented here. The previous vocabulary was ours, and it showed: fourteen
# entries for twelve steps, `main` as an ambiguous alias for one of two of them,
# no ordering (and a first four in turn order, so it READ as ordered), no first
# strike, and nine spellings across 32 positions needing a five-entry alias
# table to export at all (21.99). XMage is a working engine with a fifteen-year
# rules corpus behind it; where its shape and ours differ, its shape wins
# (Section 21.102).
#
# Each row is XMage's own four renderings of one step:
#     (enum name, index, text, stepText, stepShortText)
# `index` is XMage's, and `isBefore`/`isAfter` there are integer comparisons on
# it — which is what `phase_index` below reproduces.
PHASE_STEPS = (
    ("UNTAP", 0, "Untap", "untap step", "UN"),
    ("UPKEEP", 1, "Upkeep", "upkeep", "UP"),
    ("DRAW", 2, "Draw", "draw step", "DR"),
    ("PRECOMBAT_MAIN", 3, "Precombat Main", "precombat main step", "M1"),
    ("BEGIN_COMBAT", 4, "Begin Combat", "begin combat step", "BC"),
    ("DECLARE_ATTACKERS", 5, "Declare Attackers", "declare attackers step", "DA"),
    ("DECLARE_BLOCKERS", 6, "Declare Blockers", "declare blockers step", "DB"),
    ("FIRST_COMBAT_DAMAGE", 7, "First Combat Damage", "first combat damage", "FCD"),
    ("COMBAT_DAMAGE", 8, "Combat Damage", "combat damage step", "CD"),
    ("END_COMBAT", 9, "End Combat", "end of combat step", "EC"),
    ("POSTCOMBAT_MAIN", 10, "Postcombat Main", "postcombat main step", "M2"),
    ("END_TURN", 11, "End Turn", "end turn step", "ET"),
    ("CLEANUP", 12, "Cleanup", "cleanup step", "CL"),
)

# A mulligan happens BEFORE the turn begins, so XMage has no PhaseStep for it
# and neither does the CR. Kept as an explicit non-step: positions ask mulligan
# questions, and pretending it is a step would put it in the order, where every
# before/after comparison against it would silently answer.
PRE_TURN_PHASE = "opening hand"

# Spellings XMage does NOT use, mapped to the step they mean. Accepted, never
# canonical — `canonical_phase` converts them away.
#
# Two sources, and both have to be honoured. Four are ours, left over from the
# vocabulary XMage's replaced. The other three are the COMPREHENSIVE RULES'
# own wording: "end step" (513), "beginning of combat step" (507), "end of
# combat step" (511) — where XMage says "end turn", "begin combat" and "end
# combat". A model trained on the CR writes the CR's words, and refusing them
# would score a correctly-named step as narration.
#
# So XMage wins on what is STORED and rendered, and the CR is still readable.
# The two do not conflict: one is the canonical form, the other is an input.
LEGACY_PHASE_ALIASES = {
    "pre-combat main": "PRECOMBAT_MAIN",
    "post-combat main": "POSTCOMBAT_MAIN",
    "beginning of combat": "BEGIN_COMBAT",
    "end of combat": "END_COMBAT",
    "end step": "END_TURN",
    "ending phase": "END_TURN",
    "cleanup step": "CLEANUP",
}

# Words that NAME a step without identifying one. "main" is the whole list: a
# turn has two main phases, so `PHASE Main` says which kind of step the answer
# is in and not which of them.
#
# This is the distinction `PHASE_NAMES` and `phase_step` are separately for, and
# collapsing them broke a test the moment "main" left the vocabulary. They ask
# different questions:
#
#   PHASE_NAMES   is this a step name at all, or is it narration?   -> accept
#   phase_step    WHICH step is it?                                 -> None
#
# So an answer writing `PHASE Main` still declares a step — refusing it would
# turn a correct declaration into prose, which is 21.61's failure exactly — while
# a POSITION may not store "main", because a board sits in one specific step and
# `canonical_phase` has no way to pick. One is an input, the other is an
# identity (Section 21.102).
AMBIGUOUS_PHASE_WORDS = ("main",)

# The steps `PHASE` and `END PHASE` accept. A CLOSED vocabulary is what stops
# prose becoming a declaration — "Phase two of my plan" opens with the exact
# word and is narration (21.61).
#
# Both of XMage's prose renderings, because both are XMage's: `stepText`
# ("declare attackers step") is what a position stores, `text` ("Declare
# Attackers") is what its `toString` emits, and an answer writing either means
# the same step. Callers match by SUBSTRING, so the short codes are excluded —
# "UN" and "DA" lowercase into fragments of ordinary words and would turn most
# prose into a declaration.
#
# Longest first, so a caller taking the first hit gets the most specific one.
PHASE_NAMES = tuple(sorted(
    {row[3].lower() for row in PHASE_STEPS}
    | {row[2].lower() for row in PHASE_STEPS}
    | set(LEGACY_PHASE_ALIASES)
    | set(AMBIGUOUS_PHASE_WORDS)
    | {PRE_TURN_PHASE},
    key=lambda s: (-len(s), s),
))

# Enum names in XMage's index order. `PRE_TURN_PHASE` is deliberately absent —
# a caller asking "is X before Y?" about a mulligan must handle "not in the
# order" rather than receive an answer (Section 21.101).
PHASE_ORDER = tuple(row[0] for row in PHASE_STEPS)



def permanent_pt(perm: dict) -> tuple[int, int]:
    """A battlefield entry's power and toughness, however it stores them.

    A position stores `pt` as the STRING "1/1" and omits it for a land, because
    a rendered board must not claim a Mountain is a 0/0. A raw collector
    snapshot stores integer `power` and `toughness`. Both shapes reach the same
    consumers, and reading the wrong one silently yields 0/0.

    That cost twice in one session: `board_fidelity` reported 17 boards as
    mismatched because every permanent compared as 0/0 (21.123), and the token
    rebuild emitted a 1/1 Faerie as a **0/0**, which state-based actions would
    have killed on the spot. Second occurrence is why this lives here rather
    than in either caller (Section 21.124).
    """
    pt = perm.get("pt")
    if pt:
        try:
            a, _, b = str(pt).partition("/")
            return int(a), int(b)
        except ValueError:
            return 0, 0
    return int(perm.get("power") or 0), int(perm.get("toughness") or 0)

def phase_step(text: str) -> str | None:
    """Any spelling of a step -> its XMage enum name, or None.

    Accepts all four of XMage's renderings plus the enum name, and tolerates the
    possessives and turn ownership real data carries ("opponent's declare
    attackers step"). Longest match wins, so "precombat main step" is never
    resolved by the "main" inside it.
    """
    if not text:
        return None
    t = text.strip().lower()
    best: tuple[int, str] | None = None
    for name, _index, disp, step_text, short in PHASE_STEPS:
        for form in (step_text.lower(), disp.lower(), name.lower(),
                     name.lower().replace("_", " ")):
            if form in t and (best is None or len(form) > best[0]):
                best = (len(form), name)
        if t == short.lower():
            return name
    for form, name in LEGACY_PHASE_ALIASES.items():
        if form in t and (best is None or len(form) > best[0]):
            best = (len(form), name)
    return best[1] if best else None


def phase_index(text: str) -> int | None:
    """Where a step sits in the turn, or None if it is not one.

    None for `opening hand` and for anything unrecognised — both mean "no
    before/after answer exists", which is the honest return for a mulligan.
    """
    name = phase_step(text)
    if name is None:
        return None
    return next(row[1] for row in PHASE_STEPS if row[0] == name)


def canonical_phase(text: str) -> str | None:
    """Any spelling -> the one this project stores: XMage's `stepText`."""
    if text and text.strip().lower() == PRE_TURN_PHASE:
        return PRE_TURN_PHASE
    name = phase_step(text)
    if name is None:
        return PRE_TURN_PHASE if text and "opening hand" in text.lower() else None
    return next(row[3] for row in PHASE_STEPS if row[0] == name)

# Why a position is flagged for editing. A closed vocabulary, like
# `PHASE_VOCABULARY`, so the reason is countable rather than prose — the whole
# point is that a later author can see the PATTERN, and "needs work" in a free
# text field is not a pattern (Section 21.94).
#
# Deliberately NOT a filter. A flagged position is still generated for, still
# judged, and still counted; `eval_positions` reports how many of a run carry a
# flag so the caveat travels with the number instead of a quiet exclusion
# changing it. Editing a board is what invalidates stored answers, and
# `render_position` already carries that consequence (21.59).
POSITION_REVIEW_KINDS = {
    "shorten": "the reference does more than the board asks — trim it, often by "
               "moving the position's own phase to where the decision is",
    "split": "two decisions in one board — make two positions, or a scenario "
             "whose second step states what came between",
    "reword": "the board is right and the rubric is not — key_points or "
              "common_errors need work",
    "retire": "superseded or redundant; drop it from the set",
}

ACTION_GRAMMAR = (
    "PHASE <step>                      state the step you are acting in\n"
    "PLAY <card>                       play a land\n"
    "CAST <card>                       cast a spell that has no targets\n"
    "CAST <card> TARGET <a>, <b>       cast a spell, naming its targets\n"
    "ACTIVATE <permanent>: <ability>   activate an ability\n"
    "ATTACK <c>, <c> -> <defender>     declare attackers and who they attack\n"
    "BLOCK <blocker> -> <attacker>     one assignment per line\n"
    "ORDER TRIGGERS <a>, <b>           in the order they should resolve\n"
    "MULLIGAN                          ship the opening hand\n"
    "KEEP                              keep the opening hand as it is\n"
    "KEEP BOTTOM <a>, <b>              keep, naming the cards put on the bottom\n"
    "TAP <permanent> FOR <mana>        tap for mana, one permanent per line\n"
    "END PHASE <step>                  leave this step for the next one\n"
    "PASS                              take no further action"
)

GAMEPLAY_SYSTEM_PROMPT = (
    "You are piloting a Magic: The Gathering deck. You are given a board "
    "position and must decide what to do.\n\n"
    "Answer with the actions you take, ONE PER LINE, using exactly this "
    f"grammar:\n\n{ACTION_GRAMMAR}\n\n"
    "Write card names exactly as the position shows them, with no brackets or "
    "quotation marks around them. Give your reasoning first if you want to, "
    "then the actions. End with a single PASS.\n\n"
    # PASS and END PHASE are different plays and the grammar had only one of
    # them, so a full turn was unexpressible in the notation it was to be
    # scored in (Section 21.87).
    "PASS and END PHASE are different. PASS offers each opponent the chance to "
    "respond to the play you just made; it does not move the turn on. END PHASE "
    "leaves the step you declared and goes to the next one. A line of play "
    "within a single step ends with PASS; a turn that moves through several "
    "steps declares each with PHASE, ends each with END PHASE, and finishes "
    "with a single PASS.\n\n"
    "A creature may attack a player or a planeswalker that player controls, and "
    "nothing else. Name the defender after the arrow, one ATTACK line per "
    "defender: `ATTACK Bear, Elk -> Opponent` sends both at the player, and a "
    "second line may send another creature somewhere else.\n\n"
    "Be explicit rather than brief. Open with PHASE, naming the step the "
    "position is in. Before casting anything, TAP the lands that "
    "pay for it, one line each, naming the mana each one produces — a land that "
    "can add more than one colour produces only what you name, and some can add "
    "a colour only if a condition is met. Do not leave mana implicit."
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


# The best-calibrated judge measured. All three positive-control trip-wires,
# scored under matched `points_only` arithmetic (Section 21.34):
#
#                        range   ordering   false err   coverage   oracle/real
#   Qwen2.5-7B           +3.22      93%        40%        98/99       86%/43%
#   Llama-3.1-8B         +2.66      94%         4%        95/99       81%/63%
#   Qwen2.5-32B          +3.60     100%         0%        99/99       90%/ 8%
#
# The last column is the one that decides it: the fraction of rubric points the
# judge credits on the ORACLE — the reference answer itself — against real model
# answers. A judge that scores the oracle at 90% understands the rubric, so
# crediting real answers at 8% is discrimination and not verbatim-matching.
# Llama credits real answers at 63% against 81% on the oracle: it barely
# separates the two, and gives FULL marks to 40% of model answers, including
# ones that contradict the reference outright.
#
# ~4.5x slower per judge call than the 7-8B judges, and worth it: `--rescore-from`
# skips generation, which is the expensive half.
#
# Lives here rather than in eval.py because eval_positions.py needs it while
# building its argument parser, and it defers `import eval` until main() to keep
# the mlx import out of `--help`.
#
# Overridable with --judge-model and recorded in every report, per the stale
# ADAPTER_PATH lesson.
CALIBRATED_JUDGE_ID = "mlx-community/Qwen2.5-32B-Instruct-4bit"

PROMPT_STAMP_FILE = "prompt_fingerprint.json"


def gameplay_fingerprint() -> dict:
    """Identify the prompt shape a POSITION run was generated under.

    `prompt_fingerprint` guards the rules track and does not cover
    `GAMEPLAY_SYSTEM_PROMPT` at all, so until this existed a change to the
    gameplay prompt — the grammar the model is told to answer in — was
    completely invisible. That is Section 8.7's mechanism with no guard on it,
    on the one prompt whose *text is the output format*: edit the grammar block
    and every stored answer was produced under a different contract, while
    nothing in any run file records which.

    Deliberately SEPARATE from `prompt_fingerprint` rather than a fifth part of
    it. The two tracks share no prompt, and folding them together would make a
    gameplay-grammar edit invalidate a rules adapter that never saw it —
    trading a missing guard for a false one.

    Hashes the system prompt AND the assembled message shape, for the reason
    8.7 gives: the failure there was a shape change with every string intact.
    """
    # A fixed minimal board, so the digest tracks the FORMAT and not whichever
    # position happened to be passed in — the same reason prompt_fingerprint
    # renders with placeholder text.
    stub = {"turn": 1, "phase": "precombat main", "active_player": "you",
            "priority": "you", "battlefield": [],
            "players": {"you": {"life": 20, "hand": ["<CARD>"], "library_count": 0},
                        "opp": {"life": 20, "hand_count": 0, "library_count": 0}},
            "legal_actions": ["<ACTION>"]}
    shapes = [
        "|".join(m["role"] + ":" + m["content"] for m in build_position_messages(stub)),
        "|".join(m["role"] + ":" + m["content"]
                 for m in build_position_messages(stub, closed=True)),
    ]
    parts = {"GAMEPLAY_SYSTEM_PROMPT": GAMEPLAY_SYSTEM_PROMPT,
             "message_shapes": "\n----\n".join(shapes)}
    digests = {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in parts.items()}
    combined = hashlib.sha256(
        "\n".join(f"{k}={digests[k]}" for k in sorted(digests)).encode()
    ).hexdigest()
    return {"gameplay_fingerprint": combined, "parts": digests}


def prompt_fingerprint() -> dict:
    """Identify the prompt shape an adapter was trained under.

    An adapter is only valid for the format it saw. A prompt edit here silently
    invalidates every adapter on disk, and the failure looks like a capability
    result: the model appears to have got worse. That is the project's #1
    documented failure mode (Section 8.7) and it cost a full re-run.

    Hashes the three system prompts AND the assembled user-message shape, since
    the Section 8.7 failure was a shape change — bare question at training,
    "Rules text: ...\\n\\nQuestion: ..." at inference — with all three system
    prompts untouched. Hashing only the system strings would have missed it,
    which is the whole reason it is worth hashing anything.

    Rendered with fixed placeholder text so the digest tracks the FORMAT and not
    whatever question happened to be passed in.
    """
    shapes = [
        "|".join(m["role"] + ":" + m["content"] for m in build_rag_messages("<Q>")),
        "|".join(m["role"] + ":" + m["content"] for m in build_rag_messages("<Q>", "<CTX>")),
        "|".join(m["role"] + ":" + m["content"]
                 for m in build_rag_messages("<Q>", "Cards referenced:\n<CTX>", preformatted=True)),
    ]
    parts = {
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "RAG_SYSTEM_PROMPT": RAG_SYSTEM_PROMPT,
        "CARDS_RAG_SYSTEM_PROMPT": CARDS_RAG_SYSTEM_PROMPT,
        "message_shapes": "\n----\n".join(shapes),
    }
    digests = {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in parts.items()}
    combined = hashlib.sha256(
        "\n".join(f"{k}={digests[k]}" for k in sorted(digests)).encode()
    ).hexdigest()
    return {"prompt_fingerprint": combined, "parts": digests}


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


# --- Authoring a position from a flat form ---------------------------------
#
# Lives here, not in gameplay/positions.py, because `rubric_server.py` needs it
# and the server's ONLY project import is this module — pulling in
# gameplay/positions.py would put the card index, the gold set and the run files
# on the import path of a public service. Both functions are pure string
# parsing, so nothing followed them across.

_COUNT_RE = re.compile(r"^x(\d+)$", re.I)

_PT_RE = re.compile(r"^\d+/\d+$")

# "untapped" and "blocking" are recognised and mean the DEFAULT, not a flag.
# Without them an author writing the natural thing — "Island untapped" — got a
# permanent whose card name was "Island untapped", which the renderer then
# printed as `Island untapped — untapped`. Card-name validation catches it at
# ingest, but only after the position is written, and the renderer showed the
# corrupted name back to the author as if it were correct.
_PERM_FLAGS = {"tapped": "tapped", "attacking": "attacking",
               "sick": "summoning_sick", "summoning-sick": "summoning_sick",
               "summoning": "summoning_sick"}
_PERM_NOOP_WORDS = {"untapped", "unblocked", "blocking"}


def parse_permanent_line(line: str, controller: str) -> list[dict]:
    """Parse one battlefield line into permanents. Returns [] for a blank line."""
    tokens = line.replace(",", " ").split()
    if not tokens:
        return []
    count, pt, flags, name_parts = 1, None, {}, []
    for tok in tokens:
        low = tok.lower()
        if (m := _COUNT_RE.match(tok)):
            count = max(1, min(int(m.group(1)), 40))
        elif _PT_RE.match(tok):
            pt = tok
        elif low in _PERM_FLAGS:
            flags[_PERM_FLAGS[low]] = True
        elif low in _PERM_NOOP_WORDS:
            continue          # states the default; never part of the card name
        else:
            name_parts.append(tok)
    name = " ".join(name_parts).strip()
    if not name:
        return []
    out = []
    for _ in range(count):
        p = {"controller": controller, "card": name, "tapped": flags.get("tapped", False)}
        if pt:
            p["pt"] = pt
        for key in ("attacking", "summoning_sick"):
            if flags.get(key):
                p[key] = True
        out.append(p)
    return out


def position_from_form(f: dict) -> dict:
    """Turn the authoring form's flat fields into a position record.

    Kept server-side so the browser never constructs the schema: the form can
    change shape without the stored records changing shape, and the same
    function backs both the live preview and the save.
    """
    def rows(key):
        return [s.strip() for s in (f.get(key) or "").splitlines() if s.strip()]

    def num(key, default=0):
        try:
            return int(str(f.get(key, "")).strip() or default)
        except ValueError:
            return default

    battlefield = []
    for line in rows("you_battlefield"):
        battlefield.extend(parse_permanent_line(line, "you"))
    for line in rows("opp_battlefield"):
        battlefield.extend(parse_permanent_line(line, "opp"))

    pos = {
        "id": (f.get("id") or "").strip(),
        "format": (f.get("format") or "standard").strip(),
        "turn": num("turn"),
        "phase": (f.get("phase") or "").strip(),
        "active_player": f.get("active_player") or "you",
        "priority": f.get("priority") or "you",
        "players": {
            "you": {"life": num("you_life", 20), "hand": rows("you_hand"),
                    "library_count": num("you_library", 0)},
            "opp": {"life": num("opp_life", 20), "hand_count": num("opp_hand_count", 0),
                    "library_count": num("opp_library", 0)},
        },
        "battlefield": battlefield,
        "graveyards": {"you": rows("you_graveyard"), "opp": rows("opp_graveyard")},
        "stack": rows("stack"),
        "known_information": rows("known_information"),
        "legal_actions": rows("legal_actions"),
        "answer": (f.get("answer") or "").strip(),
        "key_points": rows("key_points"),
        "common_errors": rows("common_errors"),
        "rule_citations": [s for s in (f.get("rule_citations") or "").replace(",", " ").split() if s],
        "category": f.get("category") or "",
        "difficulty": f.get("difficulty") or "",
        "source": (f.get("source") or "").strip(),
        "cr_version": CR_VERSION,
    }
    if f.get("mana_available"):
        pos["mana_available"] = f["mana_available"].strip()
    if f.get("deck_note"):
        pos["deck_note"] = f["deck_note"].strip()
    return pos


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


# Measured on the n=22 run: `base_open`, the BEST arm, answered every position
# in a mean of 39 characters with zero lines of reasoning on 22 of 22. It is not
# deliberating about these boards at all, it is emitting a plausible action.
#
# GAMEPLAY_SYSTEM_PROMPT ends with "Give your reasoning first if you want to",
# which is permission, and no arm takes it. This makes it an instruction and
# gives it a shape. Appended rather than edited in, so the existing prompt is
# untouched and the arms already measured stay reproducible.
#
# The ACTIONS: marker is load-bearing, not decoration. Prose about a play parses
# AS that play ("Play Mountain first would strand Shock" -> PLAY with a garbage
# operand), so without a boundary this arm would have its legality destroyed by
# its own explanation. See actions.parse_output.
GAMEPLAY_DELIBERATE_INSTRUCTION = (
    "\n\nWork the position out in writing before you choose. This is required, "
    "not optional. Cover, in this order:\n"
    "1. What happens if you do nothing — what does the opponent attack with "
    "next turn, and does it kill you?\n"
    "2. Each play available to you, and what it costs you.\n"
    "3. Which play is best, and why each of the others is worse.\n\n"
    "Then write a line containing only ACTIONS: and give your actions after it, "
    "one per line, in the grammar above. Nothing before that line is read as a "
    "play, so put every action after it."
)


def build_position_messages(pos: dict, context: str | None = None,
                            closed: bool = False,
                            deliberate: bool = False) -> list[dict]:
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
        listing = ("  (no legal plays are available; PASS is the only response)"
                   if not legal else "\n".join(f"  {a}" for a in legal))
        parts.append(
            "These are the only legal plays available to you. Choose from "
            f"this list and reply with the ones you take:\n{listing}"
        )
        parts.append("Which do you take?")
    else:
        parts.append("What do you do?")

    system = GAMEPLAY_SYSTEM_PROMPT
    if deliberate:
        system += GAMEPLAY_DELIBERATE_INSTRUCTION
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


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Chance-corrected agreement on a binary call. NaN when undefined.

    Raw percent agreement flatters a skewed call: if both sides say "blundered"
    80% of the time they agree ~68% by chance alone. Kappa subtracts that.

    One definition, for the same reason `pearson_r` is one: this is the number
    the gameplay track's headline agreement is stated in (Section 21.47's
    +0.47), it was computed inline in `eval_positions.compare_judges`, and
    `adjudicate.score_run` — the judge-versus-HUMAN comparison, where chance
    correction matters most because the human's blunder calls are skewed —
    reported raw agreement with no correction at all (Section 21.57).
    """
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(1 for x, y in pairs if x == y) / n
    pa1 = sum(1 for x, _ in pairs if x) / n
    pb1 = sum(1 for _, y in pairs if y) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


# --- Card slots -------------------------------------------------------------
#
# RulesGuru returns the same ruling instantiated on different cards each time an
# id is fetched. A rubric that names its cards is therefore tied to one
# instantiation, when the ruling it encodes is not. Storing card names as slots
# lets one rubric score every version of its question — which turns "does the
# model know this rule" into a question that can actually be asked, separately
# from "has the model memorised this card".
#
# NOT braces. Magic writes mana costs as {b}, {2}, {G}, and a gold rubric
# already reads "The payment of {b} is made as ...". Braces would collide, and
# `str.format` on such a string raises KeyError('b').
SLOT_RE = re.compile(r"\[\[(card\d+)\]\]")


def templatize(text: str, slots: dict[str, str]) -> str:
    """Card names -> [[cardN]]. Longest name first, so a card whose name
    contains another card's name cannot be half-replaced."""
    if not text or not slots:
        return text
    for slot, name in sorted(slots.items(), key=lambda kv: len(kv[1]), reverse=True):
        text = re.sub(r"\b" + re.escape(name) + r"\b", f"[[{slot}]]", text)
    return text


def untemplatize(text: str, slots: dict[str, str]) -> str:
    """[[cardN]] -> card names. An unknown slot is left as-is rather than
    blanked, so a typo stays visible instead of silently deleting a claim."""
    if not text:
        return text
    return SLOT_RE.sub(lambda m: (slots or {}).get(m.group(1), m.group(0)), text)


HAND_AUTHORED_PREFIX = "hand-authored"


# Verbs a PLAYER does. A capitalized word immediately in front of one of these
# is a player; a capitalized word anywhere else is Magic vocabulary — a card,
# a creature type, a keyword — and none of our business.
#
# Only verbs that REQUIRE a player subject. An early draft also carried
# "is/are/do/does/will/would" and rewrote "What are the characteristics of..."
# into "Player B are the characteristics of..." — a generic copula is not
# evidence of a player.
#
# ONE list, because there were two. `migrate_gold_naming` kept its own copy
# with five extra verbs (search, pass, concede, attempt, try), so the migration
# and `stray_names` disagreed about what a player is — the duplicated-helper
# failure this repo keeps finding, live. The union is used everywhere.
PLAYER_VERBS = (
    "controls control casts cast has have had plays play played attacks attack "
    "blocks block targets target draws draw discards discard sacrifices sacrifice "
    "activates activate taps tap owns own gains gain loses lose wants want "
    "responds respond chooses choose declares declare puts put moves move "
    "exiles exile destroys destroy counters counter reveals reveal wins win "
    "searches search passes pass concedes concede attempts attempt tries try "
    # Added after "Nikolai uses Processor Assault" survived a migration. Measured
    # across all 1,202 candidates: these three verbs find 17 more records and 15
    # more names, every one a person. "gets", "returns" and "makes" were tried
    # alongside and rejected despite finding 3 records more — creatures get
    # bonuses and spells return permanents, so those would misfire on a corpus
    # this one happens not to contain.
    "uses use used names name named takes take took"
).split()
_PLAYER_VERBS = PLAYER_VERBS  # back-compat for readers of the private name
# An adverb may sit between the name and the verb — "Arturo then casts", "Avery
# also sacrifices". Optional, and drawn from a closed list rather than allowing
# any word, which would turn the pattern into "capitalized word near a verb" and
# lose the precision that makes it trustworthy. 5 records name a player only
# this way, all five genuine.
# "already", "first" and "again" are deliberately absent. They read naturally
# after a noun that is not a player — "Lands already played this turn" is a real
# rubric line, and it made "Lands" a player name. Measured: dropping all three
# costs zero names, because every player they would have found is discoverable
# somewhere else in the same record.
_PLAYER_ADVERBS = ("then also now later instead immediately next "
                   "subsequently").split()
_PLAYER_NAME = re.compile(
    r"\b([A-Z][a-z]{2,})\s+(?:(?:" + "|".join(_PLAYER_ADVERBS) + r")\s+)?(?:"
    + "|".join(PLAYER_VERBS) + r")\b")
PLAYER_LABELS = [f"Player {c}" for c in "ABCDEFGH"]
_EXISTING_LABEL = re.compile(r"\bPlayer [A-H]\b")

# Capitalized, in front of a player verb, and NOT a name. A sentence-initial
# pronoun satisfies the grammar test perfectly: "Ally controls a token. They
# cast Cloudshift" made "They" a second player, so normalization rewrote it as
# "Player B cast Cloudshift" — inventing an opponent and handing them the
# spell. A one-player scenario silently became a two-player one, and the answer
# it is scored against still describes the original.
#
# Measured over all 1,202 candidates before adding this: exactly four words in
# this list ever match — They (121), Who (7), Then (1), The (1) — and every one
# of the other 500 detections is a person's name. The rest of the list is a
# closed class, so it costs nothing and covers a future fetch.
_NOT_A_NAME = {
    "They", "There", "Their", "Them", "This", "That", "These", "Those",
    "Who", "Whom", "Which", "What", "When", "Where", "Whether", "While",
    "Both", "Each", "Either", "Neither", "Every", "Some", "Any", "All",
    "One", "Two", "Three", "Four", "Five", "Nobody", "Someone", "Everyone",
    "The", "And", "But", "For", "Nor", "Yet", "Then", "Than", "Thus",
    "If", "After", "Before", "Since", "Because", "However", "Instead",
    "Also", "Now", "Once", "Only", "Both", "Player", "Players",
    "Opponent", "Opponents", "Creature", "Creatures", "Permanent",
    "Permanents", "Card", "Cards", "Token", "Tokens",
}


def stray_names(question: str, answer: str, lines: list[str],
                cards: list[str] | None = None) -> list[str]:
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
    # Card words are not player names. Without `cards` this warned on "Reef"
    # (Shivan Reef), "Spellbomb", "Devilboon", "Moon" and "Storm" — every one a
    # card the rubric legitimately names, and every one in front of a player
    # verb by coincidence of phrasing. A check that is wrong in both directions
    # gets ignored, which is worse than not having it.
    known = (set(_PLAYER_NAME.findall(f"{question or ''} {answer or ''}"))
             | _NOT_A_NAME | _card_words(cards or []))
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


def _card_words(cards: list[str]) -> set[str]:
    """Every capitalized token inside a card name.

    "Alesha, Who Smiles at Death" would otherwise have "Alesha" read as a
    player. Card names are known exactly, so exclude their tokens rather than
    guessing from capitalization.
    """
    out: set[str] = set()
    for name in cards or []:
        for tok in re.split(r"[^A-Za-z]+", name):
            if tok and tok[0].isupper():
                out.add(tok)
    return out


# A player named only in the possessive ("Nyla's hand") or as the object of a
# preposition ("from Braylen using...") never stands in front of a verb, so the
# grammar test alone never finds them. Seven gold records ended up with a raw
# name sitting next to Player A/B for exactly this reason.
_PLAYER_POSSESSIVE = re.compile(r"\b([A-Z][a-z]{2,})'s\b")
# RETIRED, and kept only so the decision is legible. A name as the object of a
# preposition — "from Braylen using..." — is a real way to name a player, and
# this found 18 of the 1,202 candidates that no other pattern reaches.
#
# It is also the only pattern that can change what a question MEANS. Every
# other one requires a grammatical role a game term cannot occupy; this one
# matches any capitalized word after a common preposition, so "refers to Sand
# Warriors" makes "Sand" a player and rewrites the sentence into nonsense. The
# glossary gate caught "to Devour" and "with Cascade" but not "Sand", which is
# a creature type rather than a defined term — and there is no enumerable list
# of everything it could hit.
#
# The standing rule is that the normalizer should MISS a name rather than risk
# the meaning of a question, because a missed name is cosmetic and a corrupted
# question is a broken measurement. 18 records keep a raw name; none of them
# gets rewritten into something it does not say.
_PLAYER_PREP = re.compile(r"\b(?:from|to|by|with|against)\s+([A-Z][a-z]{2,})\b")
# "Adonis is attacking with Iron Tusk Elephant." The copula was pulled out of
# PLAYER_VERBS because a bare "is/are" rewrote "What are the characteristics
# of..." into "Player B are the characteristics of...". The progressive form is
# safe where the copula is not: it requires a following -ing word, which no
# question phrasing supplies. 17 records name a player only this way.
_PLAYER_PROGRESSIVE = re.compile(r"\b([A-Z][a-z]{2,})\s+(?:is|was|are|were)\s+\w+ing\b")


def load_glossary_terms(path: Path | None = None) -> frozenset[str]:
    """Capitalized tokens of every Comprehensive Rules glossary term.

    The vocabulary gate for the two patterns above, and the reason they are
    safe. Both are far looser than the verb test: "to Devour", "with Cascade"
    and "refers to Sand Warriors" all look exactly like "from Braylen".

    The glossary is the right source because it is precisely the list of words
    Magic has defined — 739 of them. The obvious alternative, card-name tokens
    from the Oracle pool, was measured and is unusable: 20,967 tokens that
    include ordinary English, so it suppressed 26 real player names (Alex,
    Nico, Nyla, Blake, Autumn, Clay...) to catch 5 game terms. Wrong trade.

    Not loaded at import. `common.py` is copied into the rubric-form image
    without any data files, and nothing on that path calls `find_players`.
    """
    terms: set[str] = set()
    for entry in read_jsonl(path or GLOSSARY_PATH, missing_ok=True):
        for tok in re.split(r"[^A-Za-z]+", entry.get("term", "")):
            if tok:
                terms.add(tok.capitalize())
    return frozenset(terms)


def find_players(question: str, answer: str, cards: list[str],
                 magic_terms: frozenset[str] | None = None) -> list[str]:
    """Player names, in order of first appearance in question then answer.

    The possessive and prepositional patterns run **only** when `magic_terms`
    is supplied, so a caller that cannot load the glossary gets the narrow,
    proven grammar test rather than a loose one with its guard missing.
    """
    banned = _card_words(cards) | _NOT_A_NAME
    order: list[str] = []

    def is_term(word: str) -> bool:
        # Plural too, against both vocabularies: the glossary defines "Aura"
        # while the text says "to Auras", and a record naming "Resolute
        # Survivors" bans "Resolute" while the text says "Resolutes".
        singular = word[:-1] if word.endswith("s") else word
        return (word.capitalize() in magic_terms
                or singular.capitalize() in magic_terms
                or singular in banned)

    patterns = [_PLAYER_NAME]
    if magic_terms is not None:
        # _PLAYER_PREP is deliberately NOT here. See its definition.
        patterns += [_PLAYER_POSSESSIVE, _PLAYER_PROGRESSIVE]

    for text in (question or "", answer or ""):
        # Sorted by position so first-appearance order holds across patterns.
        found = sorted((m.start(1), m.group(1), p is not _PLAYER_NAME)
                       for p in patterns for m in p.finditer(text))
        for _, name, widened in found:
            if name in banned or name in order:
                continue
            if widened and is_term(name):
                continue
            order.append(name)
    return order


def _rename(text: str, mapping: dict[str, str]) -> str:
    if not text or not mapping:
        return text
    # Longest first so a name that is a prefix of another cannot half-match.
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in
                                           sorted(mapping, key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(1)], text)


def normalize_record(rec: dict, templatize_rubric: bool | None = None,
                     magic_terms: frozenset[str] | None = None) -> tuple[dict, list[str]]:
    """Player names to `Player A/B/...`, and `card_slots` derived from `cards`.

    Returns `(new record, notes)` and never mutates the input.

    RulesGuru re-randomizes player names *and* cards on every fetch, so both
    are arbitrary labels on a ruling that is not. Normalizing players makes a
    rubric able to name one directly; storing cards as slots lets one rubric
    score every instantiation of its question, which is what separates "does
    the model know this rule" from "has it memorized this card".

    Lives here, and not in `migrate_gold_naming.py` where it started, because
    three callers need the *same* answer: the one-time migration, the task
    export a contributor reads, and the promotion that writes a candidate into
    the gold set. When only the migration had it, records promoted afterwards
    entered the gold set un-normalized and slot-less while the first 39 were
    fine — two conventions inside one measurement, arriving silently.

    Templates never enter `question` or `answer`. Those are what the model
    reads, and meta-syntax in a prompt gets copied: `ACTION_GRAMMAR` once wrote
    optional operands as `[TARGET <x>]`, the model reproduced the brackets, and
    correct plays scored as illegal.

    `templatize_rubric` defaults to "only if a person wrote this rubric" — a
    machine draft is going to be rewritten from scratch, and templating it just
    makes the text the author is meant to replace harder to read.
    """
    out = dict(rec)
    notes: list[str] = []
    cards = rec.get("cards") or []

    players = find_players(rec.get("question", ""), rec.get("answer", ""), cards,
                           magic_terms)
    # Labels already in the text are taken. This function is not only run on
    # raw RulesGuru text: it also re-runs over records that are already
    # partly normalized, where "Player A" is a literal string and therefore
    # invisible to name discovery. Allocating from the top of the list again
    # rewrote "control of a Resolute Survivors from Noemi" into "...from
    # Player A" — the player who took control — and turned the key point
    # "untaps during Noemi's untap step" into "during Player A's untap step",
    # which is the opposite of the ruling. Allocate only unused labels.
    taken = set(_EXISTING_LABEL.findall(f"{rec.get('question','')} {rec.get('answer','')}"))
    free = [lab for lab in PLAYER_LABELS if lab not in taken]
    if len(players) > len(free):
        notes.append(f"!! {len(players)} players found, only {len(free)} labels free — skipped")
        return dict(rec), notes
    mapping = dict(zip(players, free))
    if mapping:
        out["question"] = _rename(rec.get("question", ""), mapping)
        out["answer"] = _rename(rec.get("answer", ""), mapping)
        out["paraphrases"] = [_rename(p, mapping) for p in rec.get("paraphrases") or []]
        # The rubric names players too. Renaming only question and answer left a
        # key point reading "At the moment [[card2]] enters Alex controls no
        # Swamps" against a question that no longer mentions Alex.
        for field in ("key_points", "common_errors"):
            out[field] = [_rename(x, mapping) for x in rec.get(field) or []]
        notes.append("players: " + ", ".join(f"{k} -> {v}" for k, v in mapping.items()))
    else:
        notes.append("players: none found")

    if cards:
        slots = {f"card{i}": name for i, name in enumerate(cards, 1)}
        out["card_slots"] = slots
        notes.append("slots: " + ", ".join(f"{k}={v}" for k, v in slots.items()))
        want = is_hand_authored(rec) if templatize_rubric is None else templatize_rubric
        if want:
            for field in ("key_points", "common_errors"):
                before = out.get(field) or []          # already player-renamed
                after = [templatize(x, slots) for x in before]
                out[field] = after
                for b, a in zip(before, after):
                    if b != a:
                        notes.append(f"  {field}: {a}")
        else:
            notes.append("  (machine draft — rubric left alone, it gets rewritten)")
    return out, notes


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

    Use `iter_jsonl` instead when the file is large — `oracle_cards.jsonl` is
    193MB and several readers stream it on purpose. Both share this guard, so
    "stream it" never means "reimplement it".
    """
    return list(iter_jsonl(path, missing_ok=missing_ok))


def iter_jsonl(path: Path, missing_ok: bool = True):
    """Streaming `read_jsonl`, for files too large to hold in memory.

    Exists so that needing to stream is not a reason to hand-roll the reader.
    The list-building version above was added to kill five divergent copies,
    but ten inline `for line in f: json.loads(line)` loops survived it —
    including over `oracle_cards.jsonl` and `rulings.jsonl`, where loading the
    whole file was not an option and so the shared helper did not fit. That gap
    is why the guard was still missing in eight files after being "fixed".
    """
    if missing_ok and not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def guard_shrink(path: Path, new_count: int, force: bool = False,
                 min_ratio: float = 0.5, what: str = "records",
                 hint: str = "") -> None:
    """Refuse to overwrite a corpus with a much smaller one. Raises SystemExit.

    The README's own quick start once overwrote a 34,933-chunk card corpus with
    a 4% subset, and nothing downstream complained — card resolution just got
    quietly worse. `chunk_cards.py` grew a guard afterwards; `ingest.py` and
    `chunk.py`, which are the same kind of destructive rebuild, did not. One
    definition now, so a new rebuild script inherits it instead of
    re-deciding.

    `min_ratio` is the fraction of the existing corpus the new one must reach.
    It is a parameter rather than a constant because the two failure modes are
    genuinely different:

      * **0.5 for derived corpora.** The known failure is writing a format
        subset over the full pool, which is an order-of-magnitude shrink. A
        tighter bound would fire on ordinary churn.
      * **1.0 for `rules.jsonl`.** The Comprehensive Rules are pinned in this
        file (`CR_VERSION`), so the rule count is a constant until someone
        deliberately changes the pin. Any shrink at all is a parse regression,
        and it is the expensive one: nothing downstream checksums
        `rules.jsonl`, and `validate_gold` resolves gold citations against it —
        so a short corpus reports *correct* hand-authored citations as
        unresolvable, which invites "fixing" the data to match a broken parse.
    """
    if force or not path.exists():
        return
    existing = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if new_count >= existing * min_ratio:
        return
    detail = (f"refusing to shrink {path} from {existing} to {new_count} {what}.\n"
              f"  (the new corpus must be at least {min_ratio:.0%} of the existing one)")
    if hint:
        detail += f"\n  {hint}"
    raise SystemExit(detail + "\n  Pass --force if the shrink is intended.")


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


def file_sha256(path: Path) -> str:
    """Content digest of a file. ~0.8ms on the 1.4MB rules.jsonl."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_cr_pin(rules_path: Path = RULES_PATH,
                  glossary_path: Path | None = None) -> None:
    """Check a corpus against `CR_PIN`. Raises SystemExit on mismatch.

    **Only the canonical paths are checked.** Passing `--rules somewhere_else`
    is a deliberate act — building a test corpus, comparing two parses — and
    pinning it would break exactly the workflow that needs to point elsewhere.
    A local rebuild is still covered, because ingest.py writes to the canonical
    path by default and that is the case this exists for.

    Called from `load_rule_ids`, which is the chokepoint: seven of the nine
    readers of rules.jsonl reach it, so one check covers them. `chunk.py` and
    `chunk_cards.py` read structurally rather than for ids, so they call this
    directly.
    """
    checks = [("rules", rules_path, RULES_PATH, CR_PIN["rules_sha256"], CR_PIN["n_rules"])]
    if glossary_path is not None:
        checks.append(("glossary", glossary_path, GLOSSARY_PATH,
                       CR_PIN["glossary_sha256"], CR_PIN["n_glossary"]))
    _verify_pin(checks, f"the pinned CR {CR_VERSION} parse",
                "python scripts/ingest.py --update-pin")


def _verify_pin(checks: list[tuple], what: str, repin_cmd: str) -> None:
    """Shared body of every content pin. Raises SystemExit on mismatch.

    One implementation, because a second copy would be the duplicated-helper
    trap on the guard that exists to catch silent corpus drift — and the copy
    that got the "only check the canonical path" rule wrong would either nag on
    every deliberate `--cards somewhere_else` or check nothing at all.
    """
    for label, path, canonical, expected, n_expected in checks:
        if not expected or path.resolve() != canonical.resolve() or not path.exists():
            continue
        actual = file_sha256(path)
        if actual == expected:
            continue
        n_actual = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        same_count = " — SAME record count, so only the TEXT changed" if n_actual == n_expected else ""
        raise SystemExit(
            f"{path} does not match {what}.\n"
            f"  expected sha256 {expected[:16]}...  ({n_expected} {label})\n"
            f"  actual   sha256 {actual[:16]}...  ({n_actual} {label}){same_count}\n"
            f"  Every published number was computed against the pinned corpus, so this\n"
            f"  corpus and those numbers do not describe the same thing. Either restore\n"
            f"  it (`git checkout {path}`) or, if the new parse is intended, re-pin with\n"
            f"  `{repin_cmd}` and re-run what depends on it."
        )


def verify_wiki_pin(wiki_chunks_path: Path = WIKI_CHUNKS_PATH) -> None:
    """Check the wiki gloss against `WIKI_PIN`.

    Goes through `_verify_pin` like the other two — a third copy of that body
    would be the duplicated-helper trap on the guard that exists to catch silent
    corpus drift.

    Not called from a chokepoint yet, because nothing reads this corpus: whether
    wiki prose competes with CR text for retrieval slots is unmeasured, so it is
    wired into no pipeline. Any future reader should call this the way
    `CardIndex.__init__` calls `verify_card_pin`.
    """
    _verify_pin([("wiki chunks", wiki_chunks_path, WIKI_CHUNKS_PATH,
                  WIKI_PIN["wiki_chunks_sha256"], WIKI_PIN["n_wiki_chunks"])],
                "the pinned MTG Wiki snapshot",
                "python scripts/fetch_wiki.py --refresh && "
                "python scripts/chunk_wiki.py --update-pin")


def verify_card_pin(card_chunks_path: Path = CARD_CHUNKS_PATH,
                    ruling_chunks_path: Path | None = None) -> None:
    """Check the card corpora against `CARD_PIN`.

    Called from `CardIndex.__init__`, which is the chokepoint: eleven call
    sites reach the card corpus through it. Costs ~12ms against the 25MB read
    the constructor was already doing.

    Scryfall is live, so this catches what `guard_shrink` and git cannot — a
    re-fetch that returns errata'd oracle text at an unchanged record count.
    """
    checks = [("card chunks", card_chunks_path, CARD_CHUNKS_PATH,
               CARD_PIN["card_chunks_sha256"], CARD_PIN["n_card_chunks"])]
    if ruling_chunks_path is not None:
        checks.append(("ruling chunks", ruling_chunks_path, RULING_CHUNKS_PATH,
                       CARD_PIN["ruling_chunks_sha256"], CARD_PIN["n_ruling_chunks"]))
    _verify_pin(checks, "the pinned Scryfall snapshot",
                "python scripts/chunk_cards.py --update-pin")


def load_rule_ids(rules_path: Path = RULES_PATH) -> set[str]:
    """Every rule id in the pinned CR, for validating citations.

    Six callers each had their own copy of this — two as functions, four as
    inline set comprehensions that would raise on a trailing newline.

    Also the pin chokepoint: `verify_cr_pin` runs here because seven of the
    nine readers of rules.jsonl go through this function, so the corpus cannot
    be silently swapped underneath a validation or an eval.
    """
    verify_cr_pin(rules_path)
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


def verdict_is_current(verdict: dict, answer_sha: str, n_entries: int) -> bool:
    """Is this verdict about the CURRENT text AND the CURRENT rubric?

    Two things can go stale under a verdict, and checking one caught only half.

    21.62 fixed the text: a verdict is keyed `record_id::arm`, the arms get
    regenerated, and the same key names different words — so `answer_sha` rides
    on every verdict and a mismatch means "not done".

    The rubric goes stale independently. `PROTOCOL_ERRORS` was added to the
    judge's list and not to the form, so 27 verdicts were given against four
    entries where the judge was asked about eleven (21.75). The answers did not
    change, so the digest still matches, so every one of those tasks would show
    **done** — and the reviewer would skip exactly the tasks that most need
    redoing. That is 21.62's failure one level up: the identifier survives and
    what it identifies has changed underneath.

    Compared on the ENTRY COUNT the verdict was offered rather than on
    `form_version`, so a version bump for an unrelated reason does not discard
    work, and a rubric that grows always does. A verdict with no `n_shown`
    predates the field and is treated as not current — the same asymmetry
    `api_tasks` already applies to a missing digest, and for the same reason:
    asking for one duplicate verdict is visible and cheap, while silently
    skipping one is neither.
    """
    if verdict.get("answer_sha") != answer_sha:
        return False
    return verdict.get("n_shown") == n_entries


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
        # The restatement check below only makes sense for the BEHAVIOUR form
        # (Section 21.35). It was built for "Casts Lightning Strike at the
        # opponent's face instead of ...", where the opening clause restates the
        # correct play and a judge can match it before reaching the qualifier
        # that makes the play wrong.
        #
        # A claim has no qualifier — the whole sentence is the assertion, and its
        # discriminating content is the PREDICATE, which is always present.
        # "Lightning Strike should be aimed at the opponent" necessarily opens
        # with the same card as the correct line, because both are about that
        # card; what differs is what is said about it. Running the old check
        # over the rewritten corpus produced 50 warnings on 24 positions, none
        # of them the failure it was written to catch.
        if not looks_like_behaviour(err):
            continue
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

        # The behaviour-vs-claim form (Section 21.35). The judge is asked which
        # of these the candidate ASSERTED, so an entry has to be assertable: a
        # sentence a wrong answer could contain. "Adds Centaur Courser to the
        # block" is a description of what a player does, and answering "did this
        # text assert that?" is a different, vaguer question than the one
        # key_points ask — which is the leading explanation for the 40%
        # false-positive rate measured against the reference answer itself.
        if looks_like_behaviour(err):
            warnings.append(
                f'"{err.split()[0]}" opens a description of what a player DOES, not a '
                "claim an answer could make. Write the false claim itself so the judge "
                "can ask the same question it asks of key_points — see SCHEMA.md rule 5 "
                f"— {err[:55]}...")
    return warnings


# Capitalised words ending in -s that are NOT verbs. The first version of this
# check was a bare regex and fired on "Triggers resolve in the order..." and
# "This hand should be mulliganed" — both perfectly good claims, because
# "Triggers" and "This" end in s. A warning wrong in both directions gets
# ignored, which is the lesson the restatement check above already learned.
_NOT_A_VERB = {
    "this", "these", "those", "its", "his", "hers", "theirs", "as", "yes",
    "plus", "minus", "less", "unless", "always", "perhaps",
}

# Words that mark the PRECEDING word as a subject rather than a verb. If the
# second token is one of these, the sentence reads "<subject> <verb> ..." and is
# a claim; a behaviour description reads "<verb> <object> ..." instead.
_VERB_AFTER_SUBJECT = {
    "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "can", "could", "should", "would", "will", "may", "might", "must",
    "do", "does", "did", "resolve", "resolves", "apply", "applies",
    "enter", "enters", "deal", "deals", "get", "gets", "stay", "stays",
    "count", "counts", "work", "works", "happen", "happens", "remain",
    "remains", "become", "becomes", "cost", "costs", "need", "needs",
    "go", "goes", "come", "comes", "make", "makes", "let", "lets",
}

_THIRD_PERSON_VERB_RE = re.compile(r"^[A-Z][a-z]{2,}(?:s|es)$")


def looks_like_behaviour(text: str) -> bool:
    """True when a `common_errors` line describes what a player DOES.

    The judge is asked which of these the candidate ASSERTED, so an entry has to
    be a sentence a wrong answer could contain. "Adds Centaur Courser to the
    block" is not — see SCHEMA.md rule 5 and Section 21.35.

    Narrow on purpose: 89% of the pre-21.35 corpus opened with a capitalised
    third-person verb, and this catches that form without firing on a claim.
    """
    words = (text or "").strip().split()
    if len(words) < 2:
        return False
    first, second = words[0], words[1].lower().strip(".,")
    if first.lower() in _NOT_A_VERB or second in _VERB_AFTER_SUBJECT:
        return False
    return bool(_THIRD_PERSON_VERB_RE.match(first))
