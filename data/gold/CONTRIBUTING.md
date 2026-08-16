# Writing rubrics — a guide for contributors

The whole job in two pages. You need no software, no account, and no knowledge
of the project.

**You will get a link.** Open it, type your name in the box at the top, pick a
question from the list on the left. Your name is how your work gets credited —
nothing else about it matters, and there is no password.

The category dropdown filters the list, so you can stay inside one kind of
question while your eye is in. A tick marks what you have already done, and
you can reopen anything to revise it.

---

## What you are doing, and what you are not

Each question comes with its **verified answer**. Your job is to break that
answer into checkable pieces.

You are **not** deciding whether the answer is right. The answers are rules-
verified already. You are decomposing an answer you can trust.

Why bother: the project scores an AI's answers automatically, by asking a
second AI *"which of these specific claims did the answer make?"* That only
works if the claims are sharp. When the rubrics were written by machine, two
scorers agreed at r = +0.30. Rewritten by hand, +0.62. **Rubric wording is the
single biggest lever on whether any of the measurements mean anything.**

---

## The two boxes

**Key points** — the claims a correct answer must make. One per box; use
*+ key point* for another.

**Common errors** — the mistakes a real player makes. One per box.

The page checks as you type and tells you what's missing. **Save** turns on once
you have at least two key points. Skipping a question costs nothing — do as many
as you have time for and stop whenever.

You'll also see a **machine draft**, greyed out and not editable. It is a
sentence-split of the answer, shown so you can see what to avoid: compound
points, bare verdicts, no errors at all. Drafts like it score about half as well
as hand-written ones. It's there to argue with, not to copy.

An example of the finished thing:

> **Question:** Arden controls a Gaea's Liege and 1 Dryad Arbor. Nehemiah
> controls a Reverence and 3 Dryad Arbors. Can Arden attack with the Gaea's
> Liege?
>
> **Answer:** No. Checking to see if any restrictions affect a
> potentially-attacking creature is done before the creature becomes an
> "attacking creature".
>
> **Key points**
>
> - Gaea's Liege cannot be declared as an attacker at all
> - Attack restrictions are checked before a creature becomes an attacking creature
>
> **Common errors**
>
> - Says the attack is declared and then undone once Reverence applies

---

## The four rules

**1. Never a bare "Yes" or "No".** Fold the verdict into a claim that carries
information.

| | |
| --- | --- |
| ✗ | `No` |
| ✓ | `Colorless cannot be chosen, because it is not a color` |

A scorer can credit "No" to an answer that got there by bad reasoning.

**2. One checkable assertion per point.** If a point contains "and" or "so",
it is probably two points — or one point with padding.

**3. Two to four sharp points beat five to eight soft ones.** Cut anything that
restates the question or narrates the process.

**4. `COMMON ERRORS` are the traps, not the negations.** This is the half most
people skip, and it is half the rubric's power.

| | |
| --- | --- |
| ✗ | `Fails to say colorless is not a color` — just rule 1 backwards |
| ✓ | `Treats colorless as a fifth color because mana symbols include it` |

Write the mistake a real player makes, and **lead with what makes it wrong.**
If an error line opens with words that are also true of the correct answer, a
scorer matches the opening and fires before reaching the part that matters.

| | |
| --- | --- |
| ✗ | `Casts Lightning Strike at the opponent instead of the blocker` |
| ✓ | `Leaves the blocker alive, pointing Lightning Strike at the opponent` |

Rewriting three lines that way closed a 28-point gap between two scorers to 6.

---

## A worked before-and-after

This one is really in the set, and it fails rules 1 and 4:

> **Q:** Alex would like to choose colorless for Utopia Sprawl. Can they?
> **A:** No. Colorless is not a color. (105.2c / 105.1 / 105.4)

As it stands — two key points, no errors:

```text
No
Colorless is not a color
```

Rewritten:

```text
Key points
  Colorless cannot be chosen for Utopia Sprawl
  Colorless is not a color; the five colors are white, blue, black, red and green

Common errors
  Treats colorless as a sixth color because it has its own mana symbol
  Confuses "colorless" with "any color", letting the enchanted land tap for anything
```

Same ruling. The second version can tell an answer that understands the rule
from one that guessed "no".

---

## Three questions before you save

- Could an answer hit every key point and still be wrong? → points too soft.
- Could a correct answer miss a key point? → point too specific, or not required.
- Does any common error open with words that are also true of the correct
  answer? → reword it to lead with the mistake.

That's it. Your work saves as you go; there is nothing to send back.

Send the file back however you received it. Nothing else to do.
