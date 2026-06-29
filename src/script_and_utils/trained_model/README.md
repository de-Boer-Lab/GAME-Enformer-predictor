# `trained_model/` — Enformer SavedModel

The trained Enformer model, as a TensorFlow SavedModel (the published TF-Hub artifact).

```
trained_model/
├── saved_model.pb
└── variables/
    ├── variables.data-00000-of-00001
    └── variables.index
```

## Loading & interface

The model is loaded once at import time by the `Enformer` wrapper class (`Modules/Enformer.py`):

```python
model = Enformer(f"{ENFORMER_SCRIPT_DIR}/trained_model/")
```

The wrapper exposes `predict_on_batch(x)`, called by `enformer_predict_codebase.py` as `model.predict_on_batch(encoded_seq[np.newaxis])`.

| | |
|--|--|
| **Input** | one-hot DNA `(batch, 393216, 4)` — i.e. `SEQ_LEN` bp, A/C/G/T/N one-hot |
| **Output** | dict with two heads: `'human'` → `(batch, 896, 5313)`, `'mouse'` → `(batch, 896, 1643)` |
| **Bins** | 896 bins × 128 bp = 114,688 bp central prediction window |

The 896-bin window is the central region of the 393,216 bp input; the flanking context (≈ `SEQ_CONTEXT` per side after the `SEQ_LEN // 2` effective window) is consumed by the receptive field and not emitted. Track columns are indexed by the row order of the simplified target tables in [`../simplify_targets/`](../simplify_targets/README.md).

## Notes

-  Enformer, Avsec et al., 2021 (*Nature Methods*); the SavedModel is the published TF-Hub release. The model is frozen — used for inference only, never retrained here.
- **Do not edit** `saved_model.pb` or `variables/`. The bundled simplified target tables and the `/help` arrays assume this exact model and its exact track order; swapping the weights without regenerating the target tables would silently misalign every track lookup.
- **GPU required** — the SavedModel relies on GPU-accelerated TensorFlow ops (CUDA 11.2 / cuDNN 8.1 via `enformer_GPU.yml`). The container must be run with `--nv`.
- The `.def` copies this whole directory into the image; it is runtime data, not code.