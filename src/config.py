"""Project-wide paths and settings. Every path can be overridden with an env var."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("MIC_DATA_DIR", ROOT / "data"))
FEATURES_DIR = Path(os.environ.get("MIC_FEATURES_DIR", ROOT / "features"))
CHECKPOINT_DIR = Path(os.environ.get("MIC_CHECKPOINT_DIR", ROOT / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("MIC_RESULTS_DIR", ROOT / "results"))
SAMPLES_DIR = ROOT / "assets" / "samples"

# Where the app finds a trained run when checkpoints/ is empty, tried in this order:
# 1. A Hugging Face Hub model repo holding one exported run folder (see src/export.py):
#    set MIC_HF_MODEL_REPO, e.g. "your-hf-username/multilingual-image-captioning".
#    MIC_HF_MODEL_REVISION pins a branch, tag or commit (default "main").
#    A private repo also needs HF_TOKEN (read by huggingface_hub).
# 2. A zipped run folder at MIC_CHECKPOINT_URL (empty by default).
HF_MODEL_REPO = os.environ.get("MIC_HF_MODEL_REPO", "")
HF_MODEL_REVISION = os.environ.get("MIC_HF_MODEL_REVISION", "main")
RELEASE_CHECKPOINT_URL = os.environ.get("MIC_CHECKPOINT_URL", "")

# Upload limits for the public demo
MAX_UPLOAD_MB = 10
MAX_IMAGE_PIXELS = 40_000_000      # about 6300 x 6300; larger files are refused
MAX_VIDEO_MB = 25

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
