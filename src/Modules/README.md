
# Model Implementation codes:

All the modules in this repo are taken from the goggle colab [here](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-usage.ipynb). The GAME predictor imports both files with `*` but uses only a small part of each.

```
Modules/
├── Enformer.py     # Enformer model wrapper (+ unused variant-scoring classes)
└── FastaExt.py     # one_hot_encode (+ unused FASTA/VCF extraction helpers)
```

## What the predictor actually uses

`enformer_predict_codebase.py` does `from Modules.Enformer import *` and `from Modules.FastaExt import *`, but the engine only touches two things:

| Import | Used for |
|--------|----------|
| `Enformer` (class) | `model = Enformer("…/trained_model/")`, then `model.predict_on_batch(seq[np.newaxis])` |
| `one_hot_encode` | one-hot encode the padded sequence before prediction |

**`Enformer`** wraps the TF-Hub SavedModel: `__init__` does `hub.load(path).model`, and `predict_on_batch` runs the model and returns `{head: numpy_array}` for the `'human'` and `'mouse'` heads. It also carries `contribution_input_grad` (gradient-based attribution), which the predictor does not call.

**`one_hot_encode`** wraps `kipoiseq`'s `one_hot_dna` and returns `float32`. (`FastaExt.py` also pulls in `pyfaidx` / `kipoiseq`.)
