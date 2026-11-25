"""Project-wide paths and settings. Every path can be overridden with an env var."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("MIC_DATA_DIR", ROOT / "data"))
FEATURES_DIR = Path(os.environ.get("MIC_FEATURES_DIR", ROOT / "features"))
CHECKPOINT_DIR = Path(os.environ.get("MIC_CHECKPOINT_DIR", ROOT / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("MIC_RESULTS_DIR", ROOT / "results"))
SAMPLES_DIR = ROOT / "assets" / "samples"

# Zipped run folder (checkpoint + vocab.json + config.json) published as a GitHub
# Release asset. The app downloads it when checkpoints/ is empty.
# Override with the MIC_CHECKPOINT_URL environment variable.
RELEASE_CHECKPOINT_URL = os.environ.get(
    "MIC_CHECKPOINT_URL",
    "https://github.com/Anudeepthi905/multilingual-image-captioning/releases/download/"
    "v1.0/checkpoint.zip",
)

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_TRAIN_ZIP_URL = "http://images.cocodataset.org/zips/train2014.zip"

SEED = 42
VAL_FRACTION = 0.2

# Model defaults (Show, Attend and Tell style)
EMBEDDING_DIM = 256
UNITS = 512
DEFAULT_VOCAB_SIZE = 5000
DEFAULT_NUM_IMAGES = 6000
DEFAULT_BATCH_SIZE = 64
