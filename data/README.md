# Datasets: Common Sense (CS) and Justice (JS)

This directory holds the two labeled datasets used by the neural-network
experiments (`train_nn.py`, `train_multi_layer_nn.py`). Each is a binary
sentence-classification task, split into training and testing sub-parts.

## Provenance

Both datasets are derived from the **ETHICS** benchmark of Hendrycks et al.:

> Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li,
> Dawn Song, Jacob Steinhardt. *Aligning AI With Shared Human Values.*
> ICLR 2021. arXiv:2008.02275 — https://github.com/hendrycks/ethics

- **CS** (Common Sense / morality) comes from the ETHICS *commonsense morality* split.
- **JS** (Justice) comes from the ETHICS *justice* split.

The original `label, input, is_short, edited` columns are preserved in the CSV
files. If you use these datasets, please cite the ETHICS paper above in addition
to our work.

## Layout

```
data/
├── loader.py          # load_dataset("cs" | "js") — used by both train scripts
├── CS/  and  JS/
│   ├── cm_train.csv  cm_test.csv     # released, human-browsable form
│   ├── <name>0.py    <name>1.py      # TRAIN sentences (Python lists), one per class
│   ├── <name>0_test.py  <name>1_test.py   # TEST sentences, one per class
│   └── comparison.py                 # minimal-pair (SentA, SentB) tuples for pairwise eval
```

- `.csv` — the released form linked from the paper appendix. One row per
  sentence: `label,input,is_short,edited`.
- `.py` — the exact form the code loads. Each file defines a list
  `extended_spairs`; `comparison.py` defines `pairs`. The CSV and the `.py`
  sources contain the same sentences.

## Sizes

| Dataset | Train (per class × 2) | Test (per class × 2) |
|---------|-----------------------|----------------------|
| CS      | 2,500 + 2,500 = 5,000 | 900 + 900 = 1,800    |
| JS      | 2,500 + 2,500 = 5,000 | 900 + 900 = 1,800    |

## Label convention

Both datasets use **Label 0 = acceptable/justified**, **Label 1 = wrong/unjustified**.

- **CS**: `csense0` → Label 0 (acceptable), `csense1` → Label 1 (wrong).
- **JS**: the source files are swapped by `loader.py` so the convention matches
  CS — `jsense1` (reasonable/justified) → Label 0, `jsense0`
  (unreasonable/unjustified) → Label 1. See `DATASET_MODULES` in `loader.py`.

## Loading in code

```python
from data.loader import load_dataset

ds = load_dataset("cs")   # or "js"
ds["train_label0"], ds["train_label1"]   # training sentence lists
ds["test_label0"],  ds["test_label1"]    # test sentence lists
ds["sentence_pairs"]                     # minimal pairs for pairwise eval (or None)
```
