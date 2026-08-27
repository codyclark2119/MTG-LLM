# Scryfall `oracle_cards` bulk snapshot

**Written after the fact** (Section 21.110). `fetch_cards.py --bulk` writes this
file automatically now, but this download predates that code, so the upstream
`updated_at` was never captured and cannot be recovered — Scryfall rebuilds bulk
data daily and serves only the current file.

- Source: `https://api.scryfall.com/bulk-data` (type `oracle_cards`)
- Upstream updated: **not recorded** — capture it on the next re-pin
- Downloaded: 2026-08-11T16:06Z *(local file mtime, not the upstream timestamp)*
- Entries: 38,626 raw cards -> 34,933 chunks after `chunk_cards.py`
- Newest `released_at` present: 2026-11-13 (Star Trek). Future-dated because
  Scryfall carries spoiled sets before release, so this bounds SET COVERAGE
  rather than dating the download.
- Card data courtesy of Scryfall (https://scryfall.com); card text and
  templating are Wizards of the Coast IP.

`oracle_cards` is the right bulk type for this project: one row per distinct
game object, which is what a rules corpus needs. `default_cards` would give one
row per printing and `all_cards` one per printing per language, both of which
multiply the same Oracle text across reprints for no gain here.
