"""Settings shared by the TrOCR training notebooks.

Only what must be the same in every notebook lives here: where the runs are
saved, the pretrained model, the dataset version, the seed and the decoder
start token. Training hyperparameters (learning rate, epochs, smoothing,
schedule, batch size, beams) stay in each notebook, next to the code that
uses them.

Importing this module also puts the repository's ``src/`` on ``sys.path``, so
the notebooks can import ``ticha`` without installing it.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# The dataloader workers fork after the tokenizer has run; this avoids a warning
# per worker.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Every notebook saves and reads its models here.
RUNS_DIR = Path(os.environ.get("TICHA_RUNS_DIR", "~/ticha-runs")).expanduser()

MODEL_ID = "microsoft/trocr-base-handwritten"  # the pretrained checkpoint
DATASET_ID = "maclenn77/ticha-htr-lines"
DATASET_REVISION = "v1.1"  # the dataset version every run is trained and scored on
SEED = 42
START = 0  # decoder start token the models were trained with (<s>)


def require_gpu(hint: str = "run this notebook on a machine with an NVIDIA GPU"):
    """Stop with a clear message when there is no CUDA GPU; return its name."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"No GPU detected: {hint}.")
    return torch.cuda.get_device_name(0)


def gpu_memory_gb() -> float:
    """Memory of the first CUDA GPU, in GB."""
    import torch

    return torch.cuda.get_device_properties(0).total_memory / 1024**3
