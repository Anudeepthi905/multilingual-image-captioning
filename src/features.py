"""CNN feature extraction with a swappable backbone, cached to .npy files.

    inceptionv3: 299x299 input -> 8x8x2048 -> 64x2048
    resnet50:    224x224 input -> 7x7x2048 -> 49x2048

Usage:
    python -m src.features --backbone inceptionv3 --num-images 6000
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src import config

BACKBONES = {
    "inceptionv3": {"size": 299, "locations": 64},
    "resnet50": {"size": 224, "locations": 49},
}


def _app(backbone: str):
    if backbone == "inceptionv3":
        return tf.keras.applications.InceptionV3, tf.keras.applications.inception_v3
    if backbone == "resnet50":
        return tf.keras.applications.ResNet50, tf.keras.applications.resnet50
    raise ValueError(f"Unknown backbone {backbone!r}; choose from {list(BACKBONES)}")


def preprocess(image: tf.Tensor, backbone: str) -> tf.Tensor:
    """uint8 HxWx3 image tensor -> resized, backbone-normalised float tensor."""
    size = BACKBONES[backbone]["size"]
    image = tf.image.resize(tf.cast(image, tf.float32), (size, size))
    return _app(backbone)[1].preprocess_input(image)


def load_image(path, backbone: str) -> tf.Tensor:
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return preprocess(img, backbone)


def build_extractor(backbone: str) -> tf.keras.Model:
    """ImageNet backbone without the classifier head; outputs the last conv feature map."""
    model_cls, _ = _app(backbone)
    base = model_cls(include_top=False, weights="imagenet")
    return tf.keras.Model(base.input, base.output)


def to_locations(feature_map: tf.Tensor) -> tf.Tensor:
    """(batch, h, w, c) -> (batch, h*w, c)."""
    shape = tf.shape(feature_map)
    return tf.reshape(feature_map, (shape[0], -1, feature_map.shape[-1]))


def feature_path(image_path: str, backbone: str,
                 features_dir: Path = config.FEATURES_DIR) -> Path:
    return Path(features_dir) / backbone / (Path(image_path).stem + ".npy")


def extract_features(image_paths, backbone: str, features_dir: Path = config.FEATURES_DIR,
                     batch_size: int = 16) -> int:
    """Run the CNN once per image and cache (locations, 2048) float16 arrays.

    Returns the number of newly extracted images (cached ones are skipped).
    """
    todo = sorted({p for p in image_paths
                   if not feature_path(p, backbone, features_dir).exists()})
    if not todo:
        return 0
    (Path(features_dir) / backbone).mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(backbone)
    ds = (tf.data.Dataset.from_tensor_slices(todo)
          .map(lambda p: (load_image(p, backbone), p), num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    for imgs, paths in tqdm(ds, total=-(-len(todo) // batch_size),
                            desc=f"features[{backbone}]"):
        feats = to_locations(extractor(imgs, training=False)).numpy()
        for f, p in zip(feats, paths.numpy()):
            np.save(feature_path(p.decode(), backbone, features_dir), f.astype(np.float16))
    return len(todo)


def main():
    from src.data import get_split, load_manifest

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=list(BACKBONES), default="inceptionv3")
    p.add_argument("--num-images", type=int, default=config.DEFAULT_NUM_IMAGES)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    train, val = get_split(load_manifest(), args.num_images)
    n = extract_features([i["image_path"] for i in train + val], args.backbone,
                         batch_size=args.batch_size)
    print(f"Extracted {n} new feature maps ({args.backbone}).")


if __name__ == "__main__":
    main()
