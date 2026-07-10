# `script_and_utils/` — Prediction engine & model logic

This folder holds everything the model touches: track selection (with Matcher), the single-pass prediction engine, sequence preprocessing for ranges, and Enformer-specific validation/scaling.

| File | Role |
|------|------|
| `enformer_predict_codebase.py` | `predict_enformer()` — the engine: collect track indices, run the model once per sequence, slice/aggregate per task. Also carries an embedded test suite. |
| `enformer_utils.py` | `filter_evaluator_request()` (Matcher-aware track selection) + the two track-slicing functions |
| `api_preprocessing_utils.py` | `pad_sequence`, `subset_sequence_for_ranges` (and minor helper functions) |
| `model_validation.py` | `model_specific_payload_validation()` + `apply_scaling()` |

Model constants (from `enformer_predict_codebase.py`):

```
SEQ_LEN            = 393,216   # TF-Hub input length the SavedModel expects
model_input_len    = 196,608   # SEQ_LEN // 2 — the effective sequence window
prediction_window  = 114,688   # 896 bins × 128 bp — the central region actually predicted
SEQ_CONTEXT        =  40,960   # 320 bins × 128 bp — receptive-field buffer per side
BIN_SIZE           =     128
```

The model returns a dict with `'human'` and `'mouse'` heads, shapes `(1, 896, 5313)` and `(1, 896, 1643)` respectively. The wrapper slices the requested track columns out of whichever head matches the task's species.

---

## Design decisions

### 1. Predict once, filter many

`predict_enformer` runs the model **once per sequence on all required tracks**, not once per task. Before touching the model it walks every unique `(request_type, cell_type, species)` task, resolves the track indices each needs, and unions them into `unique_human_indices` / `unique_mouse_indices`. A `track_to_tasks` map records which tasks need which track so the same column is never predicted twice. After the single prediction, each task selects its own columns from the shared result and averages them (if more than 1).

### 2. Track selection by request type (`filter_evaluator_request`)

Request type determines which assay rows are eligible, matched case-insensitively against the species' simplified target table:

| Request type | Assay rows used | `type_actual` |
|--------------|-----------------|---------------|
| `accessibility` | ATAC **and** DNASE for the cell type, concatenated | e.g. `["ATAC", "DNASE"]` |
| `expression`, `expression_pol2`, `expression_mrna` | CAGE | `["CAGE"]` |
| `binding_{molecule}` | ChIP filtered to `{molecule}` | `["CHIP_{molecule}"]` |

`accessibility` deliberately pools ATAC + DNASE into one estimate (the multi-assay case below). The `binding_` prefix is parsed to extract the molecule (`binding_CTCF` → `CTCF`).

### 3. Matcher fallback (and the two-stage binding case)

Each branch tries an **exact** cell-type match first; on a miss it POSTs to `http://<matcher_ip>:<matcher_port>/match` with the requested value and the list of available values, then uses the returned `*_actual`.

```
accessibility / expression
   exact cell_type?  ── yes ─► use rows           (matcher_version = "N/A")
                     ── no  ─► Matcher(cell_type) ─► use matched rows | request error

binding_{molecule}                                  # TWO-stage
   exact (molecule AND cell_type)?  ── yes ─► use rows
                                    ── no  ─► Matcher(molecule)
                                                 ├─ retry exact cell_type with matched molecule
                                                 └─ still miss ─► Matcher(cell_type) ─► use rows | error
```

The function returns a 4-tuple `(tracks_df_or_error_str, cell_type_actual, type_actual, matcher_version)`. A connection failure returns an error string in slot 1 (and `"error"` as the version) so the caller records a per-task error and keeps going for the other tasks rather than crashing. `MATCHER_NULL_RESPONSE` (`"NULL"`) or a missing `*_actual` is treated as "no match" → request error for that task.

### 4. Dual-species heads

A request's `species` selects which head to read: `mus_musculus` → `raw_preds['mouse']`, otherwise `raw_preds['human']`. Human and mouse indices are accumulated and sliced independently, so a batch mixing both species is predicted in the same pass and routed to the correct head per task.

### 5. Readout & aggregation flags

After a task's tracks are selected they are averaged across tracks (`np.mean(..., axis=-1)`). Then:

- **`track` readout** → the full 896-bin (or range-sliced) vector is returned, plus a `trim_upstream` entry (see ranges below).
- **`point` readout** → the bins are further averaged to a single value per sequence.

The response advertises what was aggregated via an `aggregation` dict, set in the server:
- `bins: "mean"` when the readout is `point`,
- `tracks: "mean"` when `type_actual` has more than one assay (e.g. ATAC+DNASE),
- `replicates: "mean"` when more physical tracks were averaged than there are assay types (i.e. replicate tracks existed).

---

## Prediction ranges

If ranges are sent by the Evaluator they are handled in `subset_sequence_for_ranges` (pre-prediction) and the two slicing functions (post-prediction).

### Subsetting (`subset_sequence_for_ranges`)

Given `[start, end]`, the sequence is trimmed to only include sequence context that affects the ranges that fit into the Enformer context. This subsetting avoids making unnecessary predictions outside the context window of the model since predictions are independent from one another. 

- **range smaller than the prediction window** → center the range in a `prediction_window`-sized window and add `SEQ_CONTEXT` (40,960 bp) flank on each side;
- **range ≥ prediction window** → just add `SEQ_CONTEXT` flank each side.

Both clamp to the sequence ends, and the range coordinates are rebased to the subsetted sequence (`new_range_start`, `new_range_end`).

### Slicing back to the range

- **CASE 1 (subsetted seq ≤ prediction_window)** — one prediction, centered. `slice_prediction_tracks_for_range` converts the rebased range into bin indices, accounting for the padding (`left_padding = total_padding // 2`) and the left buffer (`SEQ_CONTEXT`). It returns the bin slice plus `trim_upstream` = the number of bases in the first bin that fall *before* the range start (0–127).
- **CASE 2 (subsetted seq > prediction_window)** — tiled prediction (below); the concatenated track is sliced by `start_bin = floor(range_start/128)`, `end_bin = ceil(range_end/128)`, and `trim_upstream = range_start − start_bin*128`.

`trim_upstream` is returned to the Evaluator (track readout only) so it can drop the leading bases of the first bin that lie outside the requested range.

---

## Long-sequence handling (tiling)

Enformer predicts 114,688 bp from a 196,608 bp input, with 40,960 bp of receptive-field buffer per side. Sequences are parsed by length.

### CASE 1 — sequence ≤ prediction_window (114,688 bp)

Pad to `SEQ_LEN` (393,216, centered with any leftover base added to the right), one-hot encode, predict once, then `slice_prediction_tracks` keeps only the bins overlapping real sequence (dropping full-N bins), returning `trim_upstream` for the partial leading bin.

### CASE 2 — sequence > prediction_window

1. Prepend `SEQ_CONTEXT` (40,960) N's upstream so the first real base aligns to the start of the first prediction window.
2. Slide a 196,608 bp window in **114,688 bp steps**, tracking `seq_predicted_end`.
3. **Full 196,608 chunk** → predict, keep all 896 bins (114,688 bp).
4. **Trailing partial chunk** → pad downstream with N to 196,608, predict, then crop the full-N bins from the end: `bins_to_crop = (downstream_pad − 40,960) // 128` (floored, never negative, so no real-sequence bin is dropped). The final sub-114,688 chunk does this once and breaks.
5. Concatenate all kept chunks per species into one long track; if a range was requested, slice it as in CASE 2 ("Slicing back to the range") above

Downstream (not centered) padding on trailing chunks keeps the first base aligned to the window start.

---

## Model-specific validation & scaling (`model_validation.py`)

`model_specific_payload_validation` runs after the generic schema passes and collects all violations into one `PredictionFailedError`:

- **readout** `interaction_matrix` is rejected (Enformer supports `point` and `track`);
- **species** must be `homo_sapiens` or `mus_musculus`;
- **type** starting with `conformation_` or `expression_splicing` is rejected;
- **scale**, if present, must be `linear` or `log`.

`apply_scaling(predictions, requested_scale)` returns `(transformed, effective_scale)`:
- `linear` (or unset) → unchanged, `"linear"`;
- `log` → `log2(x + 1)`.

The effective scale is echoed back as `scale_prediction_actual`.

---
