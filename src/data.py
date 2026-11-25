"""MS-COCO 2014 download, caption cleaning, vocabulary and image-level splits.

Usage:
    python -m src.data --num-images 6000              # fetch only the images needed
    python -m src.data --num-images 30000 --full-zip  # download the 13 GB train zip
"""
import argparse
import json
import random
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from src import config

START, END = "<start>", "<end>"
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


# --------------------------------------------------------------------------- cleaning
def clean_caption(text: str) -> str:
    """Lowercase, strip special characters, collapse whitespace, add start/end tokens."""
    text = _NON_ALNUM.sub(" ", text.lower())
    text = _SPACES.sub(" ", text).strip()
    return f"{START} {text} {END}"


def strip_tokens(caption: str) -> str:
    """Remove <start>/<end> markers from a cleaned caption."""
    return " ".join(w for w in caption.split() if w not in (START, END))


# --------------------------------------------------------------------------- download
def _download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        with open(tmp, "wb") as f, bar:
            for block in r.iter_content(chunk):
                f.write(block)
                bar.update(len(block))
    tmp.replace(dest)
    return dest


def download_annotations(data_dir: Path = config.DATA_DIR) -> Path:
    """Fetch captions_train2014.json (extracted from the 241 MB annotations zip)."""
    ann_path = data_dir / "annotations" / "captions_train2014.json"
    if ann_path.exists():
        return ann_path
    zip_path = data_dir / "annotations_trainval2014.zip"
    if not zip_path.exists():
        print("Downloading COCO 2014 annotations ...")
        _download(config.COCO_ANNOTATIONS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("annotations/captions_train2014.json", data_dir)
    zip_path.unlink()
    return ann_path


def load_annotations(ann_path: Path) -> dict:
    """Return {image_id: {file_name, coco_url, license, flickr_url, captions}}."""
    with open(ann_path, encoding="utf-8") as f:
        raw = json.load(f)
    images = {img["id"]: {"file_name": img["file_name"], "coco_url": img["coco_url"],
                          "license": img.get("license"), "flickr_url": img.get("flickr_url"),
                          "captions": []}
              for img in raw["images"]}
    for ann in raw["annotations"]:
        images[ann["image_id"]]["captions"].append(ann["caption"])
    return images


def _fetch_image(job) -> bool:
    url, dest = job
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def download_images(records: list, img_dir: Path, full_zip: bool = False,
                    workers: int = 16) -> list:
    """Make sure every record's image exists locally. Returns the records that do."""
    img_dir.mkdir(parents=True, exist_ok=True)
    if full_zip:
        zip_path = img_dir.parent / "train2014.zip"
        if not zip_path.exists():
            print("Downloading full COCO train2014 (~13 GB) ...")
            _download(config.COCO_TRAIN_ZIP_URL, zip_path)
        names = {r["file_name"] for r in records}
        with zipfile.ZipFile(zip_path) as zf:
            for member in tqdm(zf.namelist(), desc="extract"):
                name = Path(member).name
                if name in names and not (img_dir / name).exists():
                    with zf.open(member) as src, open(img_dir / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        jobs = [(r["coco_url"], img_dir / r["file_name"]) for r in records]
        with ThreadPoolExecutor(workers) as pool:
            list(tqdm(pool.map(_fetch_image, jobs), total=len(jobs), desc="images"))
    return [r for r in records if (img_dir / r["file_name"]).exists()]


# --------------------------------------------------------------------------- splits
def split_by_image(items: list, val_fraction: float = config.VAL_FRACTION) -> tuple:
    """Split a pre-shuffled list of images so no image lands in both splits."""
    n_train = int(round(len(items) * (1 - val_fraction)))
    return items[:n_train], items[n_train:]


def get_split(manifest: dict, num_images: int) -> tuple:
    """Deterministic train/val split for the first `num_images` images of the manifest.

    The manifest stores images in a fixed shuffled order, so a run on N images always
    sees the same training images and the same held-out set, and evaluation can
    rebuild it from the run's config.
    """
    pool = manifest["images"]
    if num_images > len(pool):
        raise ValueError(f"Requested {num_images} images but only {len(pool)} are prepared. "
                         f"Run: python -m src.data --num-images {num_images}")
    return split_by_image(pool[:num_images], manifest.get("val_fraction", config.VAL_FRACTION))


def load_manifest(data_dir: Path = config.DATA_DIR) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten(items: list) -> tuple:
    """[{image_path, captions}] -> parallel lists (one entry per caption)."""
    paths, caps = [], []
    for item in items:
        for c in item["captions"]:
            paths.append(item["image_path"])
            caps.append(c)
    return paths, caps


# --------------------------------------------------------------------------- vocabulary
def build_vectorizer(captions: list, vocab_size: int = config.DEFAULT_VOCAB_SIZE):
    """TextVectorization fitted on cleaned captions; max_length is taken from the data."""
    import tensorflow as tf

    max_length = max(len(c.split()) for c in captions)
    vec = tf.keras.layers.TextVectorization(max_tokens=vocab_size, standardize=None,
                                            output_sequence_length=max_length)
    vec.adapt(tf.data.Dataset.from_tensor_slices(captions).batch(1024))
    return vec, max_length


def prepare(num_images: int, data_dir: Path = config.DATA_DIR, full_zip: bool = False,
            seed: int = config.SEED) -> dict:
    data_dir = Path(data_dir)
    images = load_annotations(download_annotations(data_dir))
    ids = sorted(images)
    random.Random(seed).shuffle(ids)
    # ask for a few extra in case some downloads fail
    wanted = [images[i] for i in ids[: int(num_images * 1.02) + 5]]
    img_dir = data_dir / "train2014"
    got = download_images(wanted, img_dir, full_zip=full_zip)[:num_images]
    items = [{"image_path": str((img_dir / r["file_name"]).resolve()),
              "captions": [clean_caption(c) for c in r["captions"]]} for r in got]
    manifest = {"seed": seed, "val_fraction": config.VAL_FRACTION, "images": items}
    with open(data_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    train, val = split_by_image(items)
    print(f"Prepared {len(items)} images ({len(train)} train / {len(val)} val, "
          f"{sum(len(i['captions']) for i in items)} captions) -> {data_dir}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-images", type=int, default=config.DEFAULT_NUM_IMAGES)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--full-zip", action="store_true",
                   help="download the full 13 GB train2014.zip instead of single images")
    args = p.parse_args()
    prepare(args.num_images, args.data_dir, args.full_zip)


if __name__ == "__main__":
    main()
