# `simplify_targets/` — Track tables & help generation

Enformer's output heads are defined by ordered lists of tracks (5313 human, 1643 mouse). This folder turns Enformer's raw target descriptions into the simplified, queryable tables the Predictor uses for track selection, and generates the `/help` metadata.

```
simplify_targets/
├── enformer_targets_human.txt              # raw Enformer human targets (5313 rows)
├── enformer_targets_mouse.txt              # raw Enformer mouse targets (1643 rows)
├── enformer_human_targets_simplified.txt   # parsed → Assay / Cell Type / Molecule
├── enformer_mouse_targets_simplified.txt   # parsed → Assay / Cell Type / Molecule
├── enformer_help_message.json              # generated /help metadata
├── parse_enformer_target.py                # raw → simplified
├── verify_order_of_targets.py              # row-order safety check
└── generate_help_message.py                # simplified → help JSON
```

---

## Why simplification is needed

The raw targets file has one row per output track, with a free-text `description` like:

- `DNASE:K562` / `ATAC:liver`
- `CAGE:brain, adult`
- `CHIP:CTCF:GM12878`

Track selection needs to filter on **assay**, **cell type**, and (for ChIP) **molecule** separately, so each description is split into columns. The **row index must be preserved exactly**, because the position of a row *is* the track's column index in the model output tensor — selecting the wrong index returns the wrong track.

---

## Simplified schema

Both `*_simplified.txt` files are tab-separated with an integer index column and:

| Column | Meaning |
|--------|---------|
| `Assay` | `DNASE`, `ATAC`, `CAGE`, or `CHIP` |
| `Cell Type` | cell line / tissue (after the first colon) |
| `Molecule` | TF / histone mark, **CHIP only** (e.g. `CTCF`, `H3K27ac`); empty otherwise |

The split is assay-aware (`parse_enformer_target.py`): `CHIP` descriptions split on up to two colons into `Assay / Molecule / Cell Type`; `DNASE`/`ATAC`/`CAGE` split on the first colon into `Assay / Cell Type`.

Track counts after parsing:

| | DNASE | ATAC | CAGE | CHIP | Total |
|--|:----:|:----:|:----:|:----:|:----:|
| Human | 674 | 10 | 638 | 3991 | 5313 |
| Mouse | 101 | 127 | 357 | 1058 | 1643 |

---

## The row-order safety lock

`parse_enformer_target.py` splits the three assay groups into separate frames, then **re-merges while preserving original indices**: it `concat`s the groups, `sort_index()` to restore the exact original file order, then `reset_index(drop=True)` so the final index counts 0..N-1 in lockstep with the model tensor. Without this, concatenating the groups would reorder tracks and silently corrupt every index lookup.

`verify_order_of_targets.py` is the check: it reconstructs each original `description` string from the simplified columns (`CHIP:{Molecule}:{Cell Type}` or `{Assay}:{Cell Type}`) and asserts a 100% exact, in-order match against the raw file. Run it after any change to the parsing.

---

## Help-message generation

`generate_help_message.py` concatenates the human and mouse simplified tables (human first, preserving order) and emits `enformer_help_message.json` with model metadata (`input_size` 196608, `bin_size` 128, publication, authors, `game_schema_version`) plus three index-aligned arrays:

- `species` — `"homo_sapiens"` × 5313 then `"mus_musculus"` × 1643,
- `features` — `Assay`, or `CHIP_{Molecule}` for ChIP rows,
- `cell_types` — the `Cell Type` column.

This is the file served at the `/help` endpoint. Regenerate it whenever the simplified tables change so `/help` stays in sync with the tracks the predictor will actually select.