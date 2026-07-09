# Embedding Semantic Features

This repository accompanies the article **"Representation of Deep Semantic Attributes in Sentence Embeddings"**. It contains the code used to examine whether modern sentence embedding models explicitly encode continuous semantic attributes such as *danger* or *size*.

The repository supports the paper's **two complementary experiments**:

1. **Semantic-axis projection** (`main.py`) — a set of sentence pairs is embedded using several pooling strategies from the [SentenceTransformers](https://www.sbert.net/) library and analysed to determine if the contrast between two word groups (e.g. safe vs. dangerous) can be recovered by a simple geometric projection.
2. **Learned neural networks** (`train_nn.py`, `train_multi_layer_nn.py`) — single- and multi-layer classifiers are trained on the labeled **CS** (common-sense) and **JS** (justice) datasets, and their learned weights are compared back against the semantic axes. See [Neural-Network Experiments](#neural-network-experiments) and [`data/README.md`](data/README.md).

## Overview

The repository contains utilities to:

1. **Extract sentence embeddings** using different pooling methods.
2. **Store embeddings** in Excel files for inspection.
3. **Compute a feature vector** representing the difference between two groups of sentences.
4. **Project new sentences** onto this feature vector and measure how well the model orders them.

Results for each model/method/dataset combination are written to `output.yaml` and logged to the console (and optionally to Slack if credentials are provided).

## Primary Modules

- **`main.py`** – Entry point that iterates over models, pooling methods and datasets. It calls `process_configuration` which orchestrates embedding extraction and evaluation.
- **`src/methods/average_st.py`** – Implements the pooling strategies used with `SentenceTransformer`:
  - `extract_cls_pooling_st` – use the CLS token embedding.
  - `extract_max_pooling_st` – maximum over token embeddings.
  - `extract_mean_pooling_st` – mean of token embeddings (default ST behaviour).
  - `extract_mean_sqrt_len_pooling_st` – mean divided by the square root of sentence length.
  - `extract_weightedmean_pooling_st` – weighted mean over tokens.
  - `extract_lasttoken_pooling_st` – embedding of the last token.
- **`src/embedding.py`** – Saves sentence and group embeddings to Excel files via `embed_sentences`, `embed_group_1` and `embed_group_2`.
- **`src/evaluation`** – Contains utilities for loading embeddings (`loading.py`), computing projection scores (`scoring.py`) and evaluating sentence pairs (`analyzing.py`).
- **`src/utils.py`** – Helper functions for logging, storing results and sending optional Slack notifications.

## Installation

1. Install [Poetry](https://python-poetry.org/) and ensure Python ≥3.10 is available.
2. Clone this repository and install dependencies:

```bash
poetry install
```

3. Copy `credentials.yaml` and fill in your Hugging Face token. Slack values are optional but enable progress notifications:

```yaml
HF_TOKEN: "your_huggingface_token_here"
OUTPUT_YAML: "output.yaml"
SLACK_BOT_TOKEN: "xoxb-your-slack-token"    # optional
SLACK_CHANNEL: "C123456789"                # optional
```

## Running the Experiments

### Using a configuration file

Edit `config.yaml` to list models, methods and datasets you wish to run. Then execute:

```bash
python main.py --config config.yaml
```

A typical long run can be started in the background as:

```bash
nohup python main.py --config config.yaml > output.log 2>&1 &
```

### Running a single configuration

Provide the method, model and dataset on the command line:

```bash
python main.py \
  --method mean_pooling_st \
  --model_name sentence-transformers/all-mpnet-base-v2 \
  --mode danger \
  --input_file danger
```

Embeddings will be written to `src/data/` and the evaluation summary appended to `output.yaml`.

## Datasets

The neural-network experiments use two labeled binary-classification datasets, split
into training and testing sub-parts, under [`data/`](data/):

- **CS** — common-sense / morality (`data/CS/`)
- **JS** — justice (`data/JS/`)

Both are derived from the **ETHICS** benchmark (Hendrycks et al., ICLR 2021,
[arXiv:2008.02275](https://arxiv.org/abs/2008.02275),
[github.com/hendrycks/ethics](https://github.com/hendrycks/ethics)) and are provided
both as human-browsable CSV (`cm_train.csv` / `cm_test.csv`) and as the Python list
files the loader imports. See [`data/README.md`](data/README.md) for the layout, sizes,
label convention, and citation.

## Neural-Network Experiments

These reproduce the paper's *learned representation* experiments: a classifier is trained
on the CS/JS embeddings, then its weights are compared against the semantic axes (cosine
similarity, rank-order, geometric separability, ridge probe, PCA).

- **`train_nn.py`** – single-layer network (`N inputs → 1 output`), defined in
  `src/n_networks/single_layer_nn.py`.
- **`train_multi_layer_nn.py`** – multi-layer network (`N → 500 → 250 → 1`), defined in
  `src/n_networks/multi_layer_nn.py`.
- **`src/analysis/`** – weight-vs-axis comparison, separation metrics, ridge probe and PCA.
- **`src/embd_model.py`**, **`src/configure.py`**, **`src/save_embds.py`** – embedding
  generation, YAML config loading, and Excel export used by both scripts.

Configure the run in [`config_nn.yaml`](config_nn.yaml) (model, dataset `cs`/`js`,
hyperparameters, and the semantic axes to compare against), then:

```bash
python train_nn.py --config config_nn.yaml              # single-layer
python train_multi_layer_nn.py --config config_nn.yaml  # multi-layer
```

The defaults reproduce the paper with `Qwen/Qwen3-Embedding-8B` (needs a GPU and a large
download). For a quick CPU smoke-test, set the model to
`sentence-transformers/all-MiniLM-L6-v2` with `pooling: mean` (see the header of
`config_nn.yaml`). Results are written to `nn_results/` (single-layer) and
`nn_multi_layer_results/` (multi-layer); generated embeddings are cached and reused
across the two scripts.

## Citing this work

If you use this code or reproduce the experiments, please cite:

```
....
```

## Notes

- GPU acceleration is used automatically if PyTorch detects CUDA.
- The provided `size` and `danger` sentence pairs are stored in `src/data/dict/`.
- Slack integration requires inviting your bot user to the channel (see `SLACK_SETUP_GUIDE.md`).

