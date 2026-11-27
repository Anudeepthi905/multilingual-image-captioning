"""Create or update the Hugging Face Space for the demo.

Uploads only what the app needs (no data, features, checkpoints or notebooks), uses
deploy/space_README.md as the Space's README, and sets the MIC_HF_MODEL_REPO variable.

Usage (after `hf auth login` with a write token):
    python deploy/deploy_space.py --space your-hf-username/multilingual-image-captioning \
        --model-repo your-hf-username/multilingual-image-captioning
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FILES = ["app.py", "requirements.txt", "Dockerfile", ".dockerignore", "LICENSE",
         ".streamlit/config.toml"]
FOLDERS = ["src", "assets/samples"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space", required=True,
                   help="Space id, e.g. user/multilingual-image-captioning")
    p.add_argument("--model-repo", required=True, help="model repo made with src.export --push")
    p.add_argument("--private", action="store_true")
    a = p.parse_args()

    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private,
                    exist_ok=True)
    for f in FILES:
        api.upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f, repo_id=a.space,
                        repo_type="space", commit_message=f"Update {f}")
    for d in FOLDERS:
        api.upload_folder(folder_path=str(ROOT / d), path_in_repo=d, repo_id=a.space,
                          repo_type="space", ignore_patterns=["__pycache__/*", "*.pyc"],
                          commit_message=f"Update {d}/")
    api.upload_file(path_or_fileobj=str(ROOT / "deploy" / "space_README.md"),
                    path_in_repo="README.md", repo_id=a.space, repo_type="space",
                    commit_message="Update README")
    api.add_space_variable(a.space, "MIC_HF_MODEL_REPO", a.model_repo)
    print(f"Deployed: https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
