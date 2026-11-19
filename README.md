# Enformer Predictor (GAME)

A RESTful Enformer Predictor for the **Genomic API for Model Evaluation (GAME)** framework. It serves base-pair–resolution predictions of chromatin accessibility, expression, and TF/histone binding from DNA sequence over a Flask API, negotiates JSON / MessagePack wire formats, and resolves requested cell types (and ChIP molecules) against its track catalog via the GAME **Matcher** service.

The underlying model is Enformer (Avsec et al., 2021, *Nature Methods*), loaded from a TF-Hub SavedModel. It is an attention-based model that reads a long DNA context and predicts thousands of genomic tracks at 128 bp resolution across a central window, jointly for **human and mouse** output heads.

## Important Links

- To learn more about the GAME Framework ([Main GAME Repository](https://github.com/de-Boer-Lab/Genomic-API-for-Model-Evaluation), [preprint](https://www.biorxiv.org/content/10.1101/2025.07.04.663250v1.full))
- GAME Documentation: [ReadTheDocs](https://genomic-api-for-model-evaluation-documentation.readthedocs.io)
- Pre-built Enformer container image: [Hugging Face](https://huggingface.co/datasets/deBoerLab/Enformer_Predictor_GAME)
- List of all [GAME Modules](https://github.com/de-Boer-Lab/GAME_modules)

---

## Repository layout

Everything lives under `src/`: the server, the request contract, and the build definition at the top level, with the model-facing logic under `src/script_and_utils/`.

```
Enformer/
├── README.md                           # ← this file: overview, run, tracks, Matcher
└── src/
    ├── enformer_predictor_rest_api.py  # Flask entrypoint: endpoints, request loop, response assembly
    ├── schema_validation.py            # generic GAME schema checks + preprocessing
    ├── error_checking_functions.py     # APIError hierarchy + field-level checks
    ├── predictor_content_handler.py    # JSON / MessagePack decode + encode
    ├── config.py                       # container-aware predictor versioning + wire formats
    ├── enformer_GPU.yml                # conda environment (TF + CUDA) for the model
    ├── predictor_enformer.def          # Apptainer build definition
    ├── dev_run.sh                      # bind-mount dev runner (no rebuild needed)
    │
    ├── Modules/                        # Enformer model wrapper + FASTA/one-hot helpers
    │   ├── Enformer.py                 #   (Enformer class: loads the SavedModel, predict_on_batch)
    │   └── FastaExt.py                 #   (one_hot_encode, sequence helpers)
    │
    └── script_and_utils/
        ├── README.md                   # prediction engine, Matcher track selection, validation, ranges, tiling
        ├── enformer_predict_codebase.py
        ├── enformer_utils.py
        ├── api_preprocessing_utils.py
        ├── model_validation.py
        ├── simplify_targets/
        │   └── README.md               # target-table pipeline + help-message generation
        └── trained_model/
            └── README.md               # the vendored TF-Hub SavedModel
```

> Detailed design rationale lives in [`src/script_and_utils/`](src/script_and_utils/README.md). The track tables are described in [`src/script_and_utils/simplify_targets/`](src/script_and_utils/simplify_targets/README.md); the model itself in [`src/script_and_utils/trained_model/`](src/script_and_utils/trained_model/README.md).

---

## Tracks

Enformer emits two output heads. The Predictor filters them down to the tracks needed for each request.

| Head | Total tracks | DNASE | ATAC | CAGE | CHIP |
|------|:------------:|:-----:|:----:|:----:|:----:|
| Human (`homo_sapiens`) | 5313 | 674 | 10 | 638 | 3991 |
| Mouse (`mus_musculus`) | 1643 | 101 | 127 | 357 | 1058 |

Request types map onto these assays (see [`src/script_and_utils/`](src/script_and_utils/README.md) for the full logic):

- `accessibility` → ATAC **and** DNASE tracks for the cell type (concatenated if available)
- `expression`, `expression_pol2`, `expression_mrna` → CAGE
- `binding_{molecule}` → CHIP filtered to that molecule (e.g. `binding_CTCF`)

- **Supported species:** `homo_sapiens`, `mus_musculus`
- **Effective input:** 196,608 bp · **Prediction window:** 114,688 bp (896 bins) · **Resolution:** 128 bp (`bin_size: 128`)
- **Supported scales:** `linear` (default), `log` (log2(x+1))

---

## How to run

> Enformer **requires a GPU** — the model relies on GPU-accelerated TensorFlow ops.

### Start the predictor

All commands below run from inside `src/`:

```bash
cd src
apptainer run --nv --containall predictor_enformer.sif <HOST> <PORT> <MATCHER_IP> <MATCHER_PORT>
```

| Arg | Meaning |
|-----|---------|
| `HOST` | IP / hostname the predictor binds to |
| `PORT` | Port the predictor listens on |
| `MATCHER_IP` | IP of the Matcher service (optional) |
| `MATCHER_PORT` | Port of the Matcher service (optional) |

The Matcher arguments are **optional**: with both supplied the Predictor runs normally; without them it prints "running in exact-match-only mode" and any task needing a fuzzy match fails gracefully (as a failed task, see [Tasks that cannot be performed](#tasks-that-cannot-be-performed)) rather than crashing. The `--nv` flag exposes the NVIDIA GPU; `--containall` gives a clean, reproducible environment with no local installs causing dependency issues. 

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict`  | Submit sequences + prediction tasks, receive predictions |
| `GET`  | `/formats`  | Supported request/response MIME types |
| `GET`  | `/help`     | Predictor metadata (the generated help message) |

Request and response bodies may be `application/json` or `application/msgpack`; a missing `Content-Type` is assumed to be JSON. The response format follows the client `Accept` header when supported. **Errors are always returned as JSON.**

### Request lifecycle (root-level)

The `/predict` handler runs these in order; each raises an `APIError` on failure, caught by a single Flask error handler:

```
decode_request                     # JSON / MessagePack → dict   (src/predictor_content_handler.py)
        │
validate_request_payload           # generic GAME schema         (src/schema_validation.py)
        │
model_specific_payload_validation  # Enformer rules              (src/script_and_utils/model_validation.py)
        │
preprocess_data                    # flanking + range bounds     (src/schema_validation.py)
        │
predict_enformer                   # prediction engine           (src/script_and_utils/)
        │
apply_scaling (per task)           # linear / log output         (src/script_and_utils/model_validation.py)
```

### Error model

| Class | Status | `error_key` | Use |
|-------|:------:|-------------|-----|
| `BadRequestError` | 400 | `bad_prediction_request` | Malformed/schema-invalid request (missing keys, bad types, an unrecognised `readout` or `scale` value, invalid `prediction_ranges` including `start >= end`, out of bounds (`start` must be < sequence length and `end` must be <= sequence length, counting any `upstream_seq` / `downstream_seq` flanks) or keys that do not match `sequences`, unsupported Content-Type) |
| `PredictionFailedError` | 422 | `prediction_request_failed` | Valid request the model can't fulfill (`interaction_matrix` readout, invalid bases) |
| `ServerError` | 500 | `server_error` | Unexpected backend failure (incl. serialization errors) |
| (per task, inside `predictions`) | 200 | `prediction_request_failed`, `upstream_dependency_failed` | A task that could not be performed: unsupported species/type or no matching tracks (`prediction_request_failed`), or the Matcher was needed but failed or was unreachable (`upstream_dependency_failed`). The other tasks in the request are still returned |

The generic schema checks (mandatory keys, value/type checks, prediction-range format, flank strings) live in `src/schema_validation.py` + `src/error_checking_functions.py`. `prediction_ranges` are 0-indexed and half-open (`[start, end)`, like Python slicing), and `start` must be strictly less than `end`. Enformer-specific narrowing happens in `src/script_and_utils/model_validation.py`: the request-level rule (rejecting `interaction_matrix`) fails the whole request (an unrecognised `scale` is already rejected as 400 by the schema checks), while the task-level rules (non-human/mouse species, `conformation_*` / `expression_splicing` types) fail only that task.

### Tasks that cannot be performed

If a request has several `prediction_tasks`, Enformer predicts every task it can and still returns the ones it cannot (for example, the requested `species` is not `homo_sapiens` or `mus_musculus`, or no tracks match the requested cell type). The HTTP status stays 200. A task that could not be performed has `name` and all `*_requested` keys as normal, every `*_actual` key set to `null`, an error object as `predictions`, and no `aggregation` or `trim_upstream`:

```json
{
    "name": "unmatched_cell_type",
    "type_requested": "expression",
    "type_actual": null,
    "cell_type_requested": "made_up_cell_type_xyz",
    "cell_type_actual": null,
    "species_requested": "homo_sapiens",
    "species_actual": null,
    "scale_prediction_requested": "linear",
    "scale_prediction_actual": null,
    "predictions": {
        "error": [{"prediction_request_failed": "Request Error: No requested tracks in the requested type: expression and cell type: made_up_cell_type_xyz found."}]
    }
}
```

---

## How Matcher is used

When a `/predict` task names a `cell_type` (and, for binding, a molecule), the Predictor first tries an **exact** match in the relevant assay rows of the species' target table. On a miss it queries Matcher; the exact selection and fallback logic — including the **two-stage** fallback for `binding_` (match molecule first, then cell type) — lives in `filter_evaluator_request()`, documented in [`src/script_and_utils/`](src/script_and_utils/README.md).

The response surfaces `cell_type_requested` vs `cell_type_actual`, `type_actual` (the assays actually used), and the `matcher_version`. `matcher_version` is only returned when the Matcher was used (it is omitted when every task had an exact match), and is `null` if the Matcher was used, or needed but unreachable, and did not return a version. If the Matcher is needed but fails or is unreachable, only the tasks that needed it fail, with the distinct error key `upstream_dependency_failed` (a real "no match" uses `prediction_request_failed`); the other tasks are still returned.

### Build the container

```bash
cd src
apptainer build predictor_enformer.sif predictor_enformer.def
```

The build creates the `enformer17` conda environment from `enformer_GPU.yml` (TensorFlow + CUDA 11.2 / cuDNN 8.1) and installs Flask. The predictor name is versioned automatically from the Apptainer build timestamp (e.g. `Enformer_20251128-180629_PST`); outside a container it falls back to `Enformer_dev`.