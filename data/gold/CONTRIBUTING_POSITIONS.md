# Writing position rubrics — a guide for contributors

You were sent a link for **board positions**, not rules questions. If you've
seen [CONTRIBUTING.md](CONTRIBUTING.md) already, most of it still applies —
this page covers only what's different, and it's short.

**You will get a link.** Open it, type your name in the box at the top, pick a
board from the list. Your name is how your work gets credited — nothing else
about it matters, and there is no password.

---

## The one thing that's different: there is no verified answer

A rules question comes with a trusted answer for you to decompose. **A board
does not.** These 71 boards were pulled from a real recorded game and haven't
been given a correct line yet. If the page shows you an empty "answer" area,
that's expected — you are not missing anything, and you are not being asked to
judge whether an absent answer is right.

Instead, you're deciding **what the correct play is**, and writing that down.

**Key points** — the correct *line of play* for this board. What should
happen, and why. Same idea as a rules question's key points, just describing a
decision instead of a rule.

**Common errors** — mistakes a player could make on *this specific board*.
Not protocol mistakes (wrong phase, illegal taps, casting something already in
play) — those are already covered and shown to you read-only on the page. If
you write one of those, the page will flag it: writing it again would create a
second, duplicate entry that gets scored in the wrong bucket and makes the
measurement worse, not better. Stick to **strategy** mistakes: the wrong
creature to block with, the wrong target, mulliganing a keepable hand, playing
a spell now that should wait.

Some boards will also show a red note that `legal_actions` isn't filled in
yet. Ignore it and write the rubric anyway — that field affects automated
scoring later, not whether your key points and common errors are usable. They
aren't wasted; they'll become scorable once that field is filled in.

---

## The rule that matters most here: write the claim, not the behaviour

This is the position-specific version of "lead with the mistake." A common
error should be a **claim a wrong answer could make**, not a description of
what the wrong answer *does*.

Ask: **could a wrong answer contain this sentence verbatim?** If the sentence
describes an action ("adds X to the block") instead of asserting something
("X should be added to the block"), it fails the test.

| | |
| --- | --- |
| ✗ behaviour | `Adds Centaur Courser to the block, spending a 3/3 to save 3 life` |
| ✓ claim | `Centaur Courser should be added to the block alongside Sedge Scorpion` |

Why it matters: the judge is asked *"which of these claims did the answer
make?"* — a behaviour isn't a claim, so that question has no clean answer, and
a judge has been measured charging a **correct** reference line with an
invented error 40% of the time on exactly this failure mode.

---

## A worked example, from a real position

> **Board:** Opening hand — two Forest, Llanowar Elves, two Centaur Courser,
> Giant Growth, Nessian Asp. Mono-green deck, 24 lands. Legal actions: KEEP,
> MULLIGAN.
>
> **Correct line:** Keep. Two Forests plus Llanowar Elves is three mana
> sources, and the hand curves out from turn one.
>
> **Key points**
>
> - Keep — Llanowar Elves is a third mana source, so this plays as a
>   three-land hand rather than a two-land hand
> - The curve is real: Elves on one, Courser on two off the Elves, Courser on
>   three
> - Both lands produce the only colour the hand needs, so nothing in it is
>   stranded
>
> **Common errors**
>
> - Two Forests is not enough mana for this hand, so it should be mulliganed
> - This hand should be mulliganed because Nessian Asp cannot be cast early
> - Centaur Courser can be cast on turn two off the two Forests

Notice every common error is something a wrong answer could actually say —
none of them describe an action, they all assert something that's false.

---

## Everything else is the same as CONTRIBUTING.md

Two to four sharp key points beat five to eight soft ones. Cut anything that
restates the board instead of asserting a decision. One checkable claim per
point — if it contains "and" or "so", it's probably two points. Your work
saves as you go; skip anything you're unsure about and move to the next board.
