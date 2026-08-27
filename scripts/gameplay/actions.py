"""The action grammar: what a model is allowed to say when it takes a turn.

Rules answers are prose and get judged as prose. A *play* has to be compared
mechanically — was it legal, was it the same play the rubric describes — so
the model's output needs a shape a machine can read.

The grammar:

    PLAY <card>                       a land drop
    CAST <card> [TARGET <a>, <b>]     a spell, with targets if it has them
    ACTIVATE <permanent>: <ability>   an activated ability
    ATTACK <c>[, <c>] -> <defender>   declare attackers and their direction
    END PHASE [<step>]                leave this step for the next one
    BLOCK <blocker> -> <attacker>     one assignment per line
    ORDER TRIGGERS <a>, <b>           in the order they should RESOLVE
    MULLIGAN                          ship the opening hand
    KEEP [BOTTOM <a>, <b>]            keep it, naming any cards put on the bottom
    PASS                              yield priority / end the turn

Two design rules earn their keep here:

**Liberal on input.** Real model output arrives wrapped in markdown bullets,
numbered lists, bold, code fences, and a sentence of throat-clearing. Rejecting
that would measure formatting compliance, not play quality. So scaffolding is
stripped before parsing.

**Never silently drop a line.** Every line lands in exactly one of three
buckets: an `Action`, a `ParseFailure`, or `ignored` prose. The distinction
between the last two is deliberate — a line that OPENS WITH A KNOWN VERB and
then fails to parse is a real failure (the model meant to act and malformed
it), while a line that never claimed to be an action is just narration. Folding
those together would either punish models for explaining themselves or hide
malformed actions inside the "prose" bucket, and both corrupt Gate 1.

Syntax only. Card names are not checked against Oracle here — that is the
position validator's job (see build_position.py), and keeping the split clean
means a parse failure always means "malformed", never "unknown card".
"""

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-exported so `from actions import PHASE_NAMES` keeps working while the list
# itself has one home (Section 21.97). Imported at the TOP because the old
# definition sat below its own uses — legal at module scope, and a trap for
# anyone moving it.
from common import PHASE_NAMES  # noqa: E402,F401

VERBS = ("END PHASE", "PLAY", "CAST", "ACTIVATE", "ATTACK", "BLOCK", "ORDER TRIGGERS",
         "MULLIGAN", "KEEP", "PASS", "TAP", "PHASE")

# Markdown/list scaffolding the model wraps its answer in. Stripped, not failed.
_SCAFFOLD_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s*")
_BOLD_RE = re.compile(r"\*\*|__|`")
# "Action: CAST ..." / "Play: ATTACK ..." — a label the model adds to its line.
_LABEL_RE = re.compile(r"^\s*(?:action|play|step|move)\s*\d*\s*:\s*", re.I)
_FENCE_RE = re.compile(r"^\s*```")
# A line that is nothing but "ACTIONS:" — the boundary between reasoning and
# plays. Tolerates the markdown and punctuation models decorate headers with.
_ACTIONS_HEADER_RE = re.compile(
    r"^\s*[#>\-*\s]*(?:\*\*|__)?\s*ACTIONS?\s*:?\s*(?:\*\*|__)?\s*:?\s*$", re.I)


@dataclass(frozen=True)
class Action:
    """One parsed play. `key()` is the canonical form used for comparison."""

    verb: str
    args: tuple = ()
    raw: str = ""
    # ATTACK only: who the attackers are being sent at. A creature attacks a
    # player or a planeswalker that player controls (506.2), and those are the
    # only legal directions — so this is a small closed idea rather than a free
    # operand. Empty for every other verb, and empty for an ATTACK that names no
    # defender, which stays legal because 32 stored positions enumerate
    # `ATTACK <creature>` with no direction (Section 21.87).
    target: str = ""

    def key(self) -> str:
        if not self.args:
            return self.verb  # PASS, MULLIGAN, KEEP-at-seven
        if self.verb == "KEEP":
            return "KEEP BOTTOM " + ", ".join(sorted(self.args))
        if self.verb == "ATTACK":
            # Attacker order is not meaningful — sort so two orderings of the
            # same attack compare equal. The DEFENDER is meaningful: sending two
            # creatures at a player and at a planeswalker are different attacks.
            body = "ATTACK " + ", ".join(sorted(self.args))
            return f"{body} -> {self.target}" if self.target else body
        if self.verb == "BLOCK":
            return f"BLOCK {self.args[0]} -> {self.args[1]}"
        if self.verb == "ACTIVATE":
            return f"ACTIVATE {self.args[0]}: {self.args[1]}"
        if self.verb == "CAST" and len(self.args) > 1:
            return f"CAST {self.args[0]} TARGET " + ", ".join(self.args[1:])
        # ORDER TRIGGERS keeps its order — that is the whole content of the play.
        return f"{self.verb} " + ", ".join(self.args)

    def card_names(self) -> list[str]:
        """Names this action refers to, for validation against Oracle."""
        if self.verb == "ACTIVATE":
            return list(self.args[:1])  # the ability text is not a card name
        if self.verb in ("PLAY", "CAST", "ATTACK", "BLOCK", "KEEP"):
            return list(self.args)
        return []


@dataclass(frozen=True)
class ParseFailure:
    raw: str
    reason: str


@dataclass
class ParsedOutput:
    actions: list[Action] = field(default_factory=list)
    failures: list[ParseFailure] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    # How many consecutive duplicates were collapsed. A model that emits PASS
    # 190 times has taken one action and then degenerated; counting that as 190
    # actions makes "actions per answer" a measure of looping rather than of
    # play. The count is kept rather than discarded because the looping itself
    # is a real finding — the v2 adapter does exactly this on board states.
    repeats_collapsed: int = 0
    # The longest run of identical consecutive actions in the OUTPUT, counted
    # before ONCE_PER_TURN exemptions are applied. `repeats_collapsed` cannot
    # answer "did this loop" on its own — see `degenerate`.
    max_repeat: int = 0
    # The same, over PLAYS only. Section 21.58 widened `degenerate` to count
    # every repetition, which was right for `PLAY Plains` x133. The verbose
    # grammar then made it wrong: `TAP Forest FOR {G}` five times is what
    # tapping five Forests LOOKS like, and 13 of 18 degenerate flags on the
    # first protocol run were legitimate mana payments. Repeating a PLAY is a
    # loop; repeating a TAP is usually arithmetic, and a tap you cannot afford
    # is already `tap_problems`' finding (Section 21.72).
    max_play_repeat: int = 0

    @property
    def plays(self) -> list["Action"]:
        """The actions that change the game — declarations excluded.

        `PHASE` states a belief and `TAP` pays for something; neither is a line
        of play, and counting them would inflate `actions/answer` and let a
        model look busier by being more verbose. Gate 1 and `only_pass` read
        this, so asking for verbosity cannot move either (Section 21.61).
        """
        return [a for a in self.actions if a.verb not in DECLARATIONS]

    @property
    def ok(self) -> bool:
        """Gate 1's per-output criterion: at least one action, nothing malformed."""
        return bool(self.actions) and not self.failures

    @property
    def degenerate(self) -> bool:
        """True when the output was mostly repetition rather than a line of play.

        Counts BOTH kinds of repetition. `repeats_collapsed` deliberately skips
        ONCE_PER_TURN verbs, because collapsing `PLAY Swamp / PLAY Swamp` hid a
        second land drop from the legality check — the right call for the action
        list, and the wrong one here. Deriving "did this output loop" from that
        field alone inherited half its meaning: `PLAY Plains` **133 times**
        reported `degenerate=False`, and the report's Degenerate column read 0%
        (Section 21.58).

        The split the comment above ONCE_PER_TURN describes goes one level
        further, then: `repeats_collapsed` is about the action LIST, and this is
        about the OUTPUT. A loop is a loop whichever verb it loops on.

        THREE counts, because the first two only see ADJACENT repetition.
        `max_play_repeat` compares each action to the one before it, so it is
        blind to a loop that cycles — and a decode loop cycles through a
        PATTERN, not a single line. Measured: one stored answer ran to 755
        actions with `CAST Shock` **107 times** and 217 `PASS`es, interleaved as
        `PHASE / PASS / CAST Shock / PASS`, and scored `max_play_repeat = 3` and
        `degenerate = False` (Section 21.107). `PLAY Plains` x133, the case that
        motivated 21.58, happened to be consecutive, and the fix generalised
        from that one shape.
        """
        return (self.repeats_collapsed >= 5 or self.max_play_repeat >= 5
                or self.max_play_total >= 5)

    @property
    def max_play_total(self) -> int:
        """How often the single most-repeated play appears ANYWHERE in the output.

        `PASS` is excluded and everything else is not. PASS is the protocol
        terminator — the prompt mandates one after each play — so a long, correct
        turn repeats it legitimately and counting it would flag good answers.
        Nothing else has that excuse: five identical casts in one turn is a loop,
        not a line, and the threshold matches the adjacent one so the two
        cannot drift apart.
        """
        counts = Counter(a.key() for a in self.plays
                         if a.key().casefold() != "pass")
        return max(counts.values(), default=0)

    @property
    def only_pass(self) -> bool:
        """The answer parsed, is legal, and does nothing.

        `PASS` always matches `legal_actions` (Section 16.13 — it is the
        protocol terminator every answer must end with), so an answer whose only
        action is `PASS` scores `all_legal=True`. Measured across the three n=24
        judgings: **69 of 69 (100%)** such answers passed the legality check, and
        the judge called them clean on **37 of 59** graded (63%) — because doing
        nothing commits no *listed strategy error*, and `common_errors` enumerate
        strategies (Section 21.49).

        So the highest-scoring thing a model can do on this instrument is
        decline to play. Both gates are blind to it: Gate 1 asks whether the
        protocol worked, and it did; Gate 3 asks which enumerated blunder was
        committed, and none was.

        Not over-firing on this set is checked, not assumed: **no** position in
        `positions.jsonl` lists `PASS` among its `legal_actions`, and **no** key
        point opens with pass/hold/decline. A position where doing nothing is
        correct would need this reconsidered, and there are none.
        """
        return bool(self.plays) and all(a.verb == "PASS" for a in self.plays)

    def keys(self) -> list[str]:
        return [a.key() for a in self.actions]


def _norm(s: str) -> str:
    """Collapse whitespace and drop trailing punctuation from a name."""
    return re.sub(r"\s+", " ", s).strip().strip(".,;")


def _split_list(s: str) -> tuple[str, ...]:
    """Split 'a, b and c' into names, tolerating 'and' as a separator."""
    parts = re.split(r",|\band\b", s)
    return tuple(p for p in (_norm(x) for x in parts) if p)


def parse_line(line: str) -> Action | ParseFailure | None:
    """Parse one line. Returns None for prose that never claimed to be an action."""
    text = _BOLD_RE.sub("", line)
    text = _SCAFFOLD_RE.sub("", text)
    text = _LABEL_RE.sub("", text)
    # Square brackets are meta-syntax, never part of a card name, and models
    # copy them out of a grammar that uses them to mark optional operands
    # ("CAST Lightning Strike [ TARGET Grizzly Bears]"). The grammar no longer
    # uses brackets (see common.ACTION_GRAMMAR), but stripping them here keeps
    # a correct play from being scored illegal on phrasing alone.
    text = text.replace("[", " ").replace("]", " ")
    # An em dash, en dash or minus sign in front of `>` is the same arrow. A
    # reviewer typing `->` on macOS gets `—>` from autocorrect, and one line of
    # a submitted reference had it while the line above it did not — same
    # intent, different bytes, and the parser saw a BLOCK with no arrow at all.
    # `→` is normalised for the same reason. Costless: no stored model answer
    # uses any of them (0 of 418), so this only ever rescues input (21.89).
    text = re.sub(r"[—–−]\s*>", "->", text).replace("→", "->")
    text = text.strip()
    if not text:
        return None

    upper = text.upper()
    # The verb must be a whole word. A bare `startswith` matches the present
    # participles models narrate with — "Attacking with Swiftspear is correct"
    # parsed as ATTACK("ing with Swiftspear is correct"), and "Blocking the
    # Bears with Elves" parsed as a BLOCK with its operands inverted. Those are
    # phantom actions manufactured from prose: they inflate the action count and
    # sink the legality rate, which would have shown up as a capability finding
    # rather than the parser bug it is.
    verb = next(
        (v for v in VERBS
         if upper.startswith(v) and (len(text) == len(v) or not text[len(v)].isalpha())),
        None,
    )
    if verb is None:
        return None  # narration, not a malformed action

    body = _norm(text[len(verb):])

    if verb == "PASS":
        # "PASS" and "PASS priority" / "PASS the turn" are the same play.
        return Action("PASS", (), line)

    if verb == "PHASE":
        # `PHASE <name>` — the model saying when it thinks it is (Section 21.61).
        # A declaration, not a play: it changes nothing on the board and is
        # excluded from the action count and from legality, so adding it cannot
        # move Gate 1. What it does is make a belief checkable that was
        # previously only inferable from whether an action happened to be legal.
        #
        # The body must name a real step. "Phase two of my plan is to attack"
        # opens with the exact word and is prose — the whole-word verb test
        # cannot separate them, because unlike "Attacking" there is no suffix to
        # notice. A closed vocabulary can: MTG steps are enumerated in the CR,
        # so anything else is narration rather than a malformed declaration.
        if not body:
            return ParseFailure(line, "PHASE needs a phase name")
        if not any(step in body.lower() for step in PHASE_NAMES):
            return None
        return Action("PHASE", (body,), line)

    if verb == "END PHASE":
        # Advancing the turn, which `PASS` does NOT mean. Passing priority
        # offers each opponent a window to respond to the play just made; ending
        # a phase leaves the step entirely. Conflating them left no way to say
        # "I am done here, move to combat", which is most of what a FULL TURN
        # is — so stage 6 was unexpressible in the grammar it was to be scored
        # in. The one step the non-active player ends is declare blockers, which
        # is why this is the active player's declaration (Section 21.87).
        #
        # A declaration, not a play: it changes no board state that
        # `legal_actions` enumerates, so it cannot move Gate 1.
        #
        # The step is optional. Naming it is checkable and better; requiring it
        # would fail an answer that ended the phase it had just declared, which
        # is unambiguous.
        if body and not any(step in body.lower() for step in PHASE_NAMES):
            return None  # narration — "End phase two of the plan"
        return Action("END PHASE", (body,) if body else (), line)

    if verb == "TAP":
        # `TAP <permanent> FOR <mana>` — the mana declaration (Section 21.60).
        # FOR is required: `TAP Forest` alone says a land was tapped but not what
        # it produced, and on a land with more than one mana ability those are
        # different statements. A land that taps for one thing makes them look
        # interchangeable, which is exactly when an under-specified grammar
        # looks fine and later is not.
        m = re.match(r"^(.*?)\s+FOR\s+(.+)$", body, re.I)
        if not m:
            return ParseFailure(line, "TAP needs `FOR <mana>` — say what it taps for")
        perm, mana = _norm(m.group(1)), _norm(m.group(2))
        if not perm or not mana:
            return ParseFailure(line, "TAP needs a permanent and the mana it produces")
        return Action("TAP", (perm, mana), line)

    if verb == "MULLIGAN":
        return Action("MULLIGAN", (), line)

    if verb == "KEEP":
        # London mulligan: keeping at fewer than seven means naming what goes
        # back, so the bottomed cards are part of the play and not a footnote.
        m = re.split(r"\bBOTTOM(?:ING)?\b", body, maxsplit=1, flags=re.I)
        return Action("KEEP", _split_list(m[1]) if len(m) > 1 else (), line)

    if not body:
        return ParseFailure(line, f"{verb} with no argument")

    if verb == "PLAY":
        return Action("PLAY", (body,), line)

    if verb == "CAST":
        # TARGET / TARGETING / "targeting" all appear in real output.
        # "TARGET x", "targeting x", "Target: x" all occur in real output.
        m = re.split(r"\bTARGET(?:ING|S)?\b\s*:?", body, maxsplit=1, flags=re.I)
        name = _norm(m[0])
        if not name:
            return ParseFailure(line, "CAST with no card name")
        targets = _split_list(m[1]) if len(m) > 1 else ()
        if len(m) > 1 and not targets:
            return ParseFailure(line, "CAST ... TARGET with no target named")
        return Action("CAST", (name, *targets), line)

    if verb == "ACTIVATE":
        if ":" not in body:
            return ParseFailure(line, "ACTIVATE needs '<permanent>: <ability>'")
        perm, ability = body.split(":", 1)
        perm, ability = _norm(perm), _norm(ability)
        if not perm or not ability:
            return ParseFailure(line, "ACTIVATE needs both a permanent and an ability")
        return Action("ACTIVATE", (perm, ability), line)

    if verb == "ATTACK":
        body = re.sub(r"^\s*WITH\b\s*", "", body, flags=re.I)  # "ATTACK WITH x"
        # `ATTACK <c>, <c> -> <defender>`. A creature attacks a player or a
        # planeswalker its controller does not control (506.2); nothing else is
        # a legal direction, so the arrow means the same thing it means for
        # BLOCK — assign the left to the right — rather than a second syntax.
        #
        # 6 of 418 stored answers already wrote the arrow before the grammar
        # allowed it, and so did the first human-authored reference line: the
        # arrow was doing work in BLOCK and nothing in ATTACK, which is the kind
        # of near-miss a model and a person make identically (Section 21.86).
        target = ""
        if "->" in body:
            body, _, target = body.partition("->")
            target = _norm(target)
            if not target:
                return ParseFailure(line, "ATTACK -> with no defender named")
        names = _split_list(body)
        if not names:
            return ParseFailure(line, "ATTACK with no creature named")
        return Action("ATTACK", names, line, target)

    if verb == "BLOCK":
        # Both orderings occur in the wild and they mean OPPOSITE things:
        #   "BLOCK <blocker> -> <attacker>"   (canonical)
        #   "BLOCK <attacker> WITH <blocker>" (natural English)
        # Silently getting this backwards would invert every blocking position
        # in the eval, so each form is matched explicitly and neither is a
        # fallback for the other.
        if "->" in body:
            blocker, attacker = body.split("->", 1)
        elif re.search(r"\bWITH\b", body, re.I):
            attacker, blocker = re.split(r"\bWITH\b", body, maxsplit=1, flags=re.I)
        else:
            return ParseFailure(line, "BLOCK needs '<blocker> -> <attacker>' or "
                                      "'<attacker> with <blocker>'")
        blocker, attacker = _norm(blocker), _norm(attacker)
        if not blocker or not attacker:
            return ParseFailure(line, "BLOCK needs both a blocker and an attacker")
        return Action("BLOCK", (blocker, attacker), line)

    if verb == "ORDER TRIGGERS":
        # The operand order is RESOLUTION order, which is what
        # `common.ACTION_GRAMMAR` promises the model. It is deliberately not
        # the order the triggers go on the stack — the two are reverses of
        # each other (405.2), so a position authored against one convention
        # and scored against the other marks the correct play wrong. Stated
        # here as well as in the grammar because POSITIONS.md sends authors of
        # `legal_actions` to this file.
        names = _split_list(body)
        if len(names) < 2:
            return ParseFailure(line, "ORDER TRIGGERS needs at least two triggers")
        return Action("ORDER TRIGGERS", names, line)

    return ParseFailure(line, f"unhandled verb {verb}")  # unreachable



def visible_answer(text: str) -> str:
    """The part of a model's output that is its ANSWER, not its scratchpad.

    A reasoning model's <think> block is deliberation, and deliberation contains
    every play it considered and REJECTED. Feeding that to a rubric judge asks
    "which claims did this candidate make?" about text where the candidate
    argued against half of them — an error the model reasoned its way out of
    gets scored as an error it committed.

    Measured on the first Qwen3-14B run: one answer carried 6,741 characters of
    reasoning in front of 100 characters of actual play. The judge was given all
    of it, and failed outright on 13 of 22 positions because three such
    candidates in one batched call is ~16,000 characters of prompt.

    Outputs with no <think> block are returned unchanged, so this is a no-op for
    every non-reasoning model — verified: 66/66 Qwen3 answers carry the block,
    0/66 Qwen2.5 answers do, so no existing baseline moves.
    """
    text = text or ""
    if "</think>" in text:
        return text.rpartition("</think>")[2].strip()
    if "<think>" in text:
        # Never closed: the model was still reasoning when it ran out of tokens,
        # so there is no answer to score.
        return ""
    return text

def parse_output(text: str) -> ParsedOutput:
    """Parse a full model response into actions, failures, and ignored prose.

    If the output contains an `ACTIONS:` line, everything BEFORE it is prose and
    only what follows is parsed. Without that marker the whole output is parsed,
    exactly as before, so existing runs are unaffected.

    The marker exists because prose about a play parses AS that play. Measured
    on reasoning-style text before this was added:

        "Play Mountain first would strand Shock in hand"
            -> PLAY Mountain first would strand Shock in hand
        "Block Grizzly Bears with Wall of Omens is the safe assignment"
            -> BLOCK Wall of Omens is the safe assignment -> Grizzly Bears

    Both parsed silently, with no ParseFailure, as actions with garbage
    operands. Any arm asked to reason in prose would have had its action count
    inflated and its legality destroyed by its own explanation — a harness
    artifact that would have read as "reasoning makes the model play worse".
    """
    out = ParsedOutput()
    in_fence = False
    text = text or ""
    # A reasoning model emits its scratchpad in <think>...</think> before the
    # answer. That is prose, and prose about a play parses AS that play, so
    # without this a thinking model's deliberation becomes its move list.
    #
    # Measured on Qwen3-14B: the think block alone ran past 900 tokens, and the
    # two actions the parser recovered came out of the middle of the reasoning
    # rather than from a conclusion — right answer, wrong reason to trust it.
    #
    # An UNCLOSED block means the model was still thinking when it hit the token
    # limit. Everything is then reasoning and nothing is an action, which is the
    # honest reading: it never answered. That shows up as a Gate 1 parse
    # failure rather than as a silently invented play.
    if "</think>" in text:
        head, _, text = text.rpartition("</think>")
        out.ignored.extend(ln.strip() for ln in head.splitlines() if ln.strip())
    elif "<think>" in text:
        out.ignored.extend(ln.strip() for ln in text.splitlines() if ln.strip())
        text = ""
    lines = text.splitlines()
    marker = next((i for i, ln in enumerate(lines) if _ACTIONS_HEADER_RE.match(ln)), None)
    if marker is not None:
        out.ignored.extend(ln.strip() for ln in lines[:marker] if ln.strip())
        lines = lines[marker + 1:]
    run = 1
    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not line.strip():
            continue
        result = parse_line(line)
        if result is None:
            out.ignored.append(line.strip())
        elif isinstance(result, Action):
            # Collapse a run of the same action: repeating it is not doing it
            # more than once, and unbounded repetition is a decode failure.
            #
            # ONCE_PER_TURN verbs are exempt, because for them repeating the
            # line IS doing it more than once and that is an illegal play, not
            # a decode loop. Collapsing "PLAY Swamp / PLAY Swamp" hid a second
            # land drop from the legality check entirely — see rule_illegalities.
            if out.actions and out.actions[-1].key() == result.key():
                run += 1
                out.max_repeat = max(out.max_repeat, run)
                if result.verb not in DECLARATIONS:
                    out.max_play_repeat = max(out.max_play_repeat, run)
            else:
                run = 1
            if (out.actions and out.actions[-1].key() == result.key()
                    and result.verb not in REPEAT_IS_MEANINGFUL):
                out.repeats_collapsed += 1
            else:
                out.actions.append(result)
        else:
            out.failures.append(result)
    return out


def match_to_legal(action: Action, legal_actions: list[str]) -> str | None:
    """Return the legal action this one corresponds to, or None.

    Both sides go through the same parser, so `ATTACK a, b` matches
    `ATTACK b, a` and `BLOCK x -> y` matches `BLOCK y with x`. Comparing raw
    strings instead would make the closed arm measure phrasing.

    PASS always matches, whether or not the position enumerates it — see
    Section 16.13. `GAMEPLAY_SYSTEM_PROMPT` ends with "End with a single PASS",
    so every well-formed answer contains one; scoring it against
    `legal_actions` made the protocol terminator compete with the game action
    of the same name. It cost Gate 1: `pos-seed-0006` is a mulligan decision,
    which happens before any player has priority (103.4), so its
    `legal_actions` are exactly ["MULLIGAN", "KEEP"] — correctly, since there
    is no priority to pass. The model answered `KEEP` / `PASS`, doing precisely
    what it was told, and the mandated terminator scored illegal.

    Declining to act is also always available in the game sense, so this is not
    merely a protocol escape hatch. An all-PASS dodge is still caught, by the
    gate that matters: it hits no key points and scores as a blunder.
    """
    want = action.key().casefold()
    if want == "pass":
        return "PASS"
    for legal in legal_actions:
        parsed = parse_line(legal)
        if not isinstance(parsed, Action):
            continue
        if parsed.key().casefold() == want:
            return legal
        # `ATTACK <c>` in a position means "attacking with <c> is available" and
        # says nothing about the direction, so an answer that names one is not
        # illegal for being more specific. When the position DOES name a
        # defender the answer must match it — otherwise adding the direction to
        # the grammar would have scored 32 positions' worth of correct attacks
        # illegal, arriving as "the model got worse at combat" (21.61's shape).
        if (parsed.verb == "ATTACK" and action.verb == "ATTACK"
                and not parsed.target and action.target
                and sorted(parsed.args) == sorted(action.args)):
            return legal
    return None


# Actions the rules allow at most once per turn. Membership in `legal_actions`
# is a SET test — it answers "may this be done?" and has no notion of "how many
# times", so two land drops both match and both score legal.
#
# Measured on the n=22 run before this existed: 10 of 88 answers named more than
# one PLAY, and none were scored illegal for it. Two different routes through:
#
#   PLAY Swamp / PLAY Swamp    consecutive duplicates are COLLAPSED, so the
#                              harness sees one land drop and never knows
#   PLAY Swamp / PLAY Forest   not duplicates, both in legal_actions, so
#                              all_legal comes back True
#
# The collapse is right for its own purpose — a model emitting PASS 190 times
# has looped, not acted — but `repeats_collapsed` was carrying two meanings at
# once: "degenerated into a loop" and "took a once-per-turn action twice". Those
# want opposite treatment, which is the trap this repo keeps rediscovering.
#
# Only PLAY is listed. It is unambiguous (305.2: one land per turn). ATTACK is
# also declared once, but the grammar asks for all attackers on ONE line, and a
# model splitting one declaration across two lines would be punished for
# formatting rather than for play — so it is deliberately left out.
ONCE_PER_TURN = ("PLAY",)

# Verbs where repeating the LINE means doing it again, so the collapser must
# leave it alone. A superset of ONCE_PER_TURN and deliberately a separate name:
# ONCE_PER_TURN is a claim about the RULES (305.2, one land per turn) and is
# what `rule_illegalities` reads; this is a claim about the PARSER. `TAP` is not
# once per turn — you may tap many lands — but `TAP Forest FOR {G}` twice means
# two Forests, and collapsing it reported "ok" for an answer tapping three
# copies of a land it had two of. Folding TAP into ONCE_PER_TURN instead would
# have made the legality rule assert a land-drop restriction on tapping
# (Section 21.60).
REPEAT_IS_MEANINGFUL = ONCE_PER_TURN + ("TAP",)

# The steps and phases a `PHASE` declaration may name (CR 500-514), plus the
# pre-game state positions use. A closed vocabulary, because "Phase two of my
# plan" opens with the verb and is prose; see parse_line.
DECLARATIONS = ("PHASE", "TAP", "END PHASE")

# PHASE_NAMES is re-exported from common (Section 21.97).


def rule_illegalities(parsed: ParsedOutput) -> list[str]:
    """Illegal plays visible from the output alone, independent of legal_actions.

    Deliberately narrow: this is not a rules engine. It catches the one class of
    illegality that the enumerated-set check structurally cannot see, which is
    doing a once-per-turn thing more than once.
    """
    out: list[str] = []
    for verb in ONCE_PER_TURN:
        n = sum(1 for a in parsed.actions if a.verb == verb)
        if n > 1:
            out.append(f"{verb} taken {n} times; it is allowed once per turn")

    # One creature blocks at most one attacker (509.1a).
    #
    # This is the SECOND class the enumerated-set check structurally cannot
    # see, and it is worse than the first because both halves look legal.
    # `legal_actions` lists ALTERNATIVES — on a blocking board it offers every
    # block that would be legal on its own — so an answer choosing two of them
    # with the same blocker matches the list twice and is scored all_legal.
    #
    # Found by human adjudication, not by the harness: a reviewer wrote "it
    # tries to use fog bank to block twice" about an answer the run had marked
    # legal. Across the stored position runs, 59 answers marked all_legal=True
    # do this. It inflates Gate 1's legality, and Gate 1 already fails — so
    # correcting it makes that gate fail harder rather than rescuing it.
    #
    # No menace, banding or "can block an additional creature" handling here:
    # this is not a rules engine, and no position in the set grants one. Adding
    # such a card means this check needs the exception.
    blockers: dict[str, int] = {}
    for a in parsed.actions:
        if a.verb == "BLOCK" and a.args:
            blockers[a.args[0]] = blockers.get(a.args[0], 0) + 1
    for blocker, n in blockers.items():
        if n > 1:
            out.append(f"{blocker} blocks {n} attackers; a creature blocks at most one (509.1a)")
    return out


def legality(parsed: ParsedOutput, legal_actions: list[str]) -> dict:
    """How much of an output names an action from the enumerated legal set.

    Scored over `plays`, never `actions`. `legal_actions` enumerates *plays*, so
    a `PHASE` or `TAP` line can never appear in it — evaluating declarations
    against that list made a correct verbose answer score `all_legal=False` on
    every declaration it contained. The prompt asks for those lines
    (Section 21.60), so the effect would have been an arm's legality collapsing
    the moment verbosity was required, arriving in the shape of a result:
    *"asking the model to show its working makes it play worse."*

    This repo has had that exact shape twice — a grammar whose optional-operand
    brackets the model copied, and prose about a play parsing AS that play. Both
    times the harness punished the model for doing what it was told.
    """
    plays = parsed.plays
    matched = [a for a in plays if match_to_legal(a, legal_actions)]
    rule_bad = rule_illegalities(parsed)
    return {
        "n_actions": len(plays),
        "n_legal": len(matched),
        "n_failures": len(parsed.failures),
        "all_legal": (bool(plays) and len(matched) == len(plays) and not rule_bad),
        "illegal": [a.key() for a in plays if not match_to_legal(a, legal_actions)]
                   + rule_bad,
    }
