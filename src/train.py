"""Train the attention captioner on cached CNN features.

Usage:
    python -m src.train --backbone inceptionv3 --vocab-size 5000 --num-images 6000 --epochs 20

Artifacts:
    checkpoints/<run>/        ckpt-*, config.json, vocab.json
    results/loss_<run>.png    train/val loss curve
    results/train_log_<run>.csv
"""
import argparse
import csv
import json
import time

import numpy as np
import tensorflow as tf

from src import config
from src.data import build_vectorizer, flatten, get_split, load_manifest
from src.features import BACKBONES, extract_features, feature_path
from src.model import build_models, loss_function
from src.plotting import plot_loss


def run_name(backbone, vocab_size, num_images, epochs) -> str:
    return f"{backbone}_v{vocab_size}_n{num_images}_e{epochs}"


def make_dataset(image_paths, cap_vectors, backbone, batch_size, shuffle=True):
    feat_files = [str(feature_path(p, backbone)) for p in image_paths]

    def _load(path, cap):
        feat = tf.numpy_function(lambda p: np.load(p.decode()).astype(np.float32),
                                 [path], tf.float32)
        feat.set_shape((BACKBONES[backbone]["locations"], 2048))
        return feat, cap

    ds = tf.data.Dataset.from_tensor_slices((feat_files, cap_vectors))
    if shuffle:
        ds = ds.shuffle(len(feat_files), seed=config.SEED, reshuffle_each_iteration=True)
    return (ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))


def train(backbone="inceptionv3", vocab_size=config.DEFAULT_VOCAB_SIZE,
          num_images=config.DEFAULT_NUM_IMAGES, epochs=20,
          batch_size=config.DEFAULT_BATCH_SIZE, learning_rate=1e-3, name=None):
    tf.keras.utils.set_random_seed(config.SEED)
    name = name or run_name(backbone, vocab_size, num_images, epochs)
    run_dir = config.CHECKPOINT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- data
    train_items, val_items = get_split(load_manifest(), num_images)
    extract_features([i["image_path"] for i in train_items + val_items], backbone)
    train_paths, train_caps = flatten(train_items)
    val_paths, val_caps = flatten(val_items)
    vectorizer, max_length = build_vectorizer(train_caps, vocab_size)
    vocab = vectorizer.get_vocabulary()
    train_vec = vectorizer(tf.constant(train_caps)).numpy()
    val_vec = vectorizer(tf.constant(val_caps)).numpy()
    train_ds = make_dataset(train_paths, train_vec, backbone, batch_size)
    val_ds = make_dataset(val_paths, val_vec, backbone, batch_size, shuffle=False)

    run_cfg = {"run": name, "backbone": backbone, "vocab_size": len(vocab),
               "requested_vocab_size": vocab_size, "num_images": num_images,
               "epochs": epochs, "batch_size": batch_size, "max_length": int(max_length),
               "embedding_dim": config.EMBEDDING_DIM, "units": config.UNITS,
               "train_images": len(train_items), "val_images": len(val_items)}
    (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))
    (run_dir / "vocab.json").write_text(json.dumps(vocab))
    print(json.dumps(run_cfg, indent=2))

    # ---- model
    encoder, decoder = build_models(len(vocab), locations=BACKBONES[backbone]["locations"])
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    start_id = vocab.index("<start>")
    epoch_var = tf.Variable(0, dtype=tf.int64)
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer,
                               epoch=epoch_var)
    manager = tf.train.CheckpointManager(ckpt, str(run_dir), max_to_keep=2)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"Resumed from {manager.latest_checkpoint} (epoch {int(epoch_var)})")

    def forward(img_tensor, target, training):
        """Teacher forcing: feed the ground-truth previous word at every step."""
        loss = 0.0
        hidden = decoder.reset_state(tf.shape(target)[0])
        dec_input = tf.fill([tf.shape(target)[0], 1], tf.cast(start_id, target.dtype))
        features = encoder(img_tensor)
        for i in range(1, max_length):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)
        return loss / float(max_length - 1)

    @tf.function
    def train_step(img_tensor, target):
        with tf.GradientTape() as tape:
            loss = forward(img_tensor, target, True)
        variables = encoder.trainable_variables + decoder.trainable_variables
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    @tf.function
    def val_step(img_tensor, target):
        return forward(img_tensor, target, False)

    # ---- loop
    log_path = config.RESULTS_DIR / f"train_log_{name}.csv"
    history = []
    if log_path.exists() and int(epoch_var) > 0:
        with open(log_path, newline="") as f:
            history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        history = history[: int(epoch_var)]
    for epoch in range(int(epoch_var), epochs):
        t0 = time.time()
        train_losses = [float(train_step(x, y)) for x, y in train_ds]
        val_losses = [float(val_step(x, y)) for x, y in val_ds]
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)),
               "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
               "seconds": time.time() - t0}
        history.append(row)
        epoch_var.assign(epoch + 1)
        manager.save()
        print(f"Epoch {epoch + 1}/{epochs}  loss {row['train_loss']:.4f}  "
              f"val_loss {row['val_loss']:.4f}  time {row['seconds']:.1f}s", flush=True)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)
        plot_loss(history, config.RESULTS_DIR / f"loss_{name}.png", title=name)
    return run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=list(BACKBONES), default="inceptionv3")
    p.add_argument("--vocab-size", type=int, default=config.DEFAULT_VOCAB_SIZE)
    p.add_argument("--num-images", type=int, default=config.DEFAULT_NUM_IMAGES)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--name", help="run name (default: derived from the settings)")
    a = p.parse_args()
    out = train(a.backbone, a.vocab_size, a.num_images, a.epochs, a.batch_size, a.lr, a.name)
    print(f"Checkpoint: {out}")


if __name__ == "__main__":
    main()
