"""
Dynamic data loader that imports sentence data based on config.yaml dataset setting.

Supports:
  - "cs": data/CS/ (csense0, csense1, CSense0_test, CSense1_test)
  - "js": data/JS/ (jsense0, jsense1, JSense0_test, JSense1_test)
"""
import importlib
from pathlib import Path


# Maps dataset name -> module paths for (train_label0, train_label1, test_label0, test_label1)
DATASET_MODULES = {
    "cs": {
        "train_label0": "data.CS.csense0",
        "train_label1": "data.CS.csense1",
        "test_label0": "data.CS.CSense0_test",
        "test_label1": "data.CS.CSense1_test",
        "comparison": "data.CS.comparison",
        "data_dir": Path(__file__).parent / "CS",
    },
    # JS label convention: Label 0 = reasonable/justified (jsense1),
    #                      Label 1 = unreasonable/unjustified (jsense0)
    # Swapped to match CS convention where Label 0 = acceptable, Label 1 = wrong
    "js": {
        "train_label0": "data.JS.jsense1",       # reasonable/justified → Label 0
        "train_label1": "data.JS.jsense0",       # unreasonable/unjustified → Label 1
        "test_label0": "data.JS.JSense1_test",   # reasonable/justified → Label 0
        "test_label1": "data.JS.JSense0_test",   # unreasonable/unjustified → Label 1
        "comparison": "data.JS.comparison",
        "data_dir": Path(__file__).parent / "JS",
    },
}


def load_dataset(dataset_name: str):
    """
    Load train/test sentence lists for the given dataset.

    Returns:
        dict with keys:
            - train_label0: list of label-0 training sentences
            - train_label1: list of label-1 training sentences
            - test_label0: list of label-0 test sentences
            - test_label1: list of label-1 test sentences
            - sentence_pairs: list of (sent_a, sent_b) tuples, or None
            - data_dir: Path to the dataset directory
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_MODULES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_MODULES.keys())}"
        )

    modules = DATASET_MODULES[dataset_name]

    train_label0 = importlib.import_module(modules["train_label0"]).extended_spairs
    train_label1 = importlib.import_module(modules["train_label1"]).extended_spairs
    test_label0 = importlib.import_module(modules["test_label0"]).extended_spairs
    test_label1 = importlib.import_module(modules["test_label1"]).extended_spairs

    sentence_pairs = None
    if modules["comparison"] is not None:
        comp = importlib.import_module(modules["comparison"])
        sentence_pairs = comp.pairs

    return {
        "train_label0": train_label0,
        "train_label1": train_label1,
        "test_label0": test_label0,
        "test_label1": test_label1,
        "sentence_pairs": sentence_pairs,
        "data_dir": modules["data_dir"],
    }
