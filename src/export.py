"""Export a trained run for inference (weights only) and optionally upload it to the Hub.

A training checkpoint also stores the Adam optimizer state, about two thirds of its
size. The exported folder keeps only the encoder and decoder weights plus
config.json and vocab.json, which is all the app needs.

Usage:
    python -m src.export --run inceptionv3_v5000_n6000_e20
    python -m src.export --run inceptionv3_v5000_n6000_e20 \
        --push your-hf-username/multilingual-image-captioning     # needs `hf auth login`
"""
import argparse
import json
import shutil
from pathlib import Path

import tensorflow as tf

from src import config
from src.features import BACKBONES
from src.model import build_models


def export_run(run_dir, out_dir) -> Path:
    """Write a weights-only copy of `run_dir` to `out_dir` and return `out_dir`."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    vocab = json.loads((run_dir / "vocab.json").read_text())
    encoder, decoder = build_models(len(vocab), cfg["embedding_dim"], cfg["units"],
                                    BACKBONES[cfg["backbone"]]["locations"])
    latest = tf.train.latest_checkpoint(str(run_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    # expect_partial: the optimizer and epoch counter in the training checkpoint are dropped
    tf.train.Checkpoint(encoder=encoder, decoder=decoder).restore(latest).expect_partial()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    tf.train.CheckpointManager(ckpt, str(out_dir), max_to_keep=1).save()
    for name in ("config.json", "vocab.json"):
        shutil.copy(run_dir / name, out_dir / name)
    return out_dir


def push_to_hub(folder, repo_id: str, private: bool = False) -> str:
    """Create the model repo if needed and upload the exported folder to its root."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                      commit_message="Upload exported captioning run")
    return f"https://huggingface.co/{repo_id}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder name under checkpoints/")
    p.add_argument("--out", type=Path, help="output folder (default: export/<run>)")
    p.add_argument("--push", metavar="REPO_ID", help="upload to this Hugging Face model repo")
    p.add_argument("--private", action="store_true", help="create the repo as private")
    a = p.parse_args()
    out = export_run(config.CHECKPOINT_DIR / a.run, a.out or config.ROOT / "export" / a.run)
    size = sum(f.stat().st_size for f in out.iterdir()) / 2**20
    print(f"Exported {a.run} -> {out} ({size:.1f} MB)")
    if a.push:
        print("Uploaded:", push_to_hub(out, a.push, a.private))


if __name__ == "__main__":
    main()
