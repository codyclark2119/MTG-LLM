# Dependencies, in three tiers

`requirements.txt` at the repo root is the **lock**, and it does not move.
These files say what the lock is *for*.

| File | Question it answers | Install it when |
| --- | --- | --- |
| `../requirements.txt` | what was known to work together, exactly | reproducing a published number |
| `base.txt` | what the code actually imports | reading the dependency list |
| `ci.txt` | what the test suite needs with no model | Linux, CI, any non-Mac |
| `research.txt` | what is in the environment but imported by nothing here | never, deliberately — see below |

## Why the lock is not trimmed

`requirements.txt` is a full `pip freeze` of the Apple Silicon environment that
produced every number in `DEVELOPMENT_PLAN.md`. Five of its pins
(`mlx-audio`, `mlx-vlm`, `opencv-python`, `miniaudio`, `sounddevice`) are not
imported by any script here; they come from the wider MLX environment it was
frozen from, and they are listed in `research.txt` so nobody has to rediscover
that.

**They stay in the lock.** "Not imported by source code" is not the same as
"not present when the numbers were measured" — a reproducibility artifact
describes an environment, not an import graph, and a transitive pin that
happens to be unreachable today is still part of what ran. Trimming the lock to
match the imports would produce a file that installs a *different* environment
from the one the results describe, which is the one thing it exists to prevent.
`base.txt` is where the import graph is written down; that is the file to read
if the question is "what does this code need".

## Why the versions are not loosened

`base.txt` pins to the same versions as the lock, MLX included. `mlx`,
`mlx-metal`, `mlx-lm` and `mlx-embeddings` move fast and are coupled to each
other and to the Metal runtime; a loosened pin here would mean a fresh clone
installs a different inference stack from the one every stored eval run used,
and model output is not stable across that. If an upgrade is wanted it is a
deliberate act with a re-measurement behind it, not a `>=`.

`test_deploy.py` asserts that `base.txt` and `ci.txt` agree with the lock
version for version, so this split cannot quietly become a second source of
truth.

## The deploy image is a fourth thing, and stays separate

`deploy/requirements.txt` is three lines and is not derived from any of these.
It is the public rubric server's image, whose entire virtue is that it contains
almost nothing; `test_deploy.py` fails if it grows past four lines. Do not
point it at `base.txt`.
