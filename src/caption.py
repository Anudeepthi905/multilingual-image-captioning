"""Caption an image with a trained run (greedy or beam search) and plot attention.

Usage:
    python -m src.caption --image assets/samples/dog.jpg            # latest run
    python -m src.caption --image photo.jpg --run inceptionv3_v5000_n6000_e20 --beam 3 --plot
"""
import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf

from src import config
from src.features import BACKBONES, build_extractor, preprocess, to_locations
from src.model import build_models

PAD, UNK, START, END = "", "[UNK]", "<start>", "<end>"


def list_runs(checkpoint_dir: Path = config.CHECKPOINT_DIR) -> list:
    """Run folders that contain a finished checkpoint, newest first."""
    d = Path(checkpoint_dir)
    if not d.exists():
        return []
    runs = [p for p in d.iterdir()
            if (p / "config.json").exists() and tf.train.latest_checkpoint(str(p))]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def download_hf_checkpoint(repo_id: str = config.HF_MODEL_REPO,
                           revision: str = config.HF_MODEL_REVISION,
                           checkpoint_dir: Path = config.CHECKPOINT_DIR) -> list:
    """Download an exported run folder from a Hugging Face model repo into checkpoints/.

    The repo holds one run at its root (config.json, vocab.json, checkpoint, ckpt-*),
    as written by `python -m src.export`. Files already downloaded are reused.
    """
    from huggingface_hub import snapshot_download

    if not repo_id:
        raise ValueError("No Hugging Face model repo set (MIC_HF_MODEL_REPO).")
    target = Path(checkpoint_dir) / ("hf-" + repo_id.split("/")[-1])
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=target,
                      allow_patterns=["config.json", "vocab.json", "checkpoint", "ckpt-*"])
    return list_runs(checkpoint_dir)


def find_or_download_runs(checkpoint_dir: Path = config.CHECKPOINT_DIR) -> list:
    """Local runs first; otherwise the Hugging Face repo; otherwise the zip URL."""
    runs = list_runs(checkpoint_dir)
    if not runs and config.HF_MODEL_REPO:
        runs = download_hf_checkpoint(checkpoint_dir=checkpoint_dir)
    if not runs and config.RELEASE_CHECKPOINT_URL:
        runs = download_release_checkpoint(checkpoint_dir=checkpoint_dir)
    return runs


def download_release_checkpoint(url: str = config.RELEASE_CHECKPOINT_URL,
                                checkpoint_dir: Path = config.CHECKPOINT_DIR) -> list:
    """Download and unzip a zipped run folder into checkpoints/."""
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(checkpoint_dir)
    return list_runs(checkpoint_dir)


class Captioner:
    """Loads a run folder (checkpoint + config.json + vocab.json) for inference."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.cfg = json.loads((self.run_dir / "config.json").read_text())
        self.vocab = json.loads((self.run_dir / "vocab.json").read_text())
        self.backbone = self.cfg["backbone"]
        self.max_length = self.cfg["max_length"]
        self.start_id = self.vocab.index(START)
        self.end_id = self.vocab.index(END)
        self.encoder, self.decoder = build_models(
            len(self.vocab), self.cfg["embedding_dim"], self.cfg["units"],
            BACKBONES[self.backbone]["locations"])
        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
        latest = tf.train.latest_checkpoint(str(self.run_dir))
        if latest is None:
            raise FileNotFoundError(f"No checkpoint in {self.run_dir}")
        ckpt.restore(latest).expect_partial().assert_existing_objects_matched()
        self._extractor = None

    # ------------------------------------------------------------------ features
    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = build_extractor(self.backbone)
        return self._extractor

    def image_features(self, image) -> tf.Tensor:
        """image: file path, bytes, PIL.Image or HxWx3 uint8 array -> (1, L, 2048)."""
        if isinstance(image, (str, Path)):
            image = Path(image).read_bytes()
        if isinstance(image, bytes):
            arr = tf.io.decode_image(image, channels=3, expand_animations=False)
        else:
            arr = tf.convert_to_tensor(np.asarray(image.convert("RGB")
                                                  if hasattr(image, "convert") else image))
        x = preprocess(arr, self.backbone)[tf.newaxis]
        return to_locations(self.extractor(x, training=False))

    def cached_features(self, npy_path) -> tf.Tensor:
        return tf.convert_to_tensor(np.load(npy_path).astype(np.float32))[tf.newaxis]

    # ------------------------------------------------------------------ decoding
    def _step(self, word_id, features, hidden):
        logits, hidden, attn = self.decoder(tf.constant([[word_id]], tf.int64), features,
                                            hidden)
        return tf.nn.log_softmax(logits[0]).numpy(), hidden, attn[0, :, 0].numpy()

    def greedy(self, feats) -> tuple:
        features = self.encoder(feats)
        hidden = self.decoder.reset_state(1)
        word, words, attention = self.start_id, [], []
        for _ in range(self.max_length - 1):
            logp, hidden, attn = self._step(word, features, hidden)
            logp[[0, 1]] = -np.inf  # never emit padding / [UNK]
            word = int(np.argmax(logp))
            if word == self.end_id:
                break
            words.append(self.vocab[word])
            attention.append(attn)
        return words, attention

    def beam_search(self, feats, beam: int = 3, length_penalty: float = 0.7) -> tuple:
        features = self.encoder(feats)
        # each beam: (score, word ids, attention list, hidden, finished)
        beams = [(0.0, [self.start_id], [], self.decoder.reset_state(1), False)]
        for _ in range(self.max_length - 1):
            candidates = []
            for score, ids, attns, hidden, done in beams:
                if done:
                    candidates.append((score, ids, attns, hidden, True))
                    continue
                logp, new_hidden, attn = self._step(ids[-1], features, hidden)
                logp[[0, 1]] = -np.inf
                if len(ids) == 1:  # no empty captions
                    logp[self.end_id] = -np.inf
                for w in np.argsort(logp)[-beam:]:
                    candidates.append((score + float(logp[w]), ids + [int(w)],
                                       attns + [attn], new_hidden, int(w) == self.end_id))

            def norm(c):
                return c[0] / (len(c[1]) ** length_penalty)

            beams = sorted(candidates, key=norm, reverse=True)[:beam]
            if all(b[4] for b in beams):
                break
        best = max(beams, key=lambda c: c[0] / (len(c[1]) ** length_penalty))
        ids, attns = best[1][1:], best[2]
        if ids and ids[-1] == self.end_id:
            ids, attns = ids[:-1], attns[:-1]
        return [self.vocab[i] for i in ids], attns

    def caption(self, image=None, feats=None, beam: int = 1) -> tuple:
        """Return (caption string, words, attention list)."""
        if feats is None:
            feats = self.image_features(image)
        words, attention = (self.greedy(feats) if beam <= 1
                            else self.beam_search(feats, beam))
        return " ".join(words), words, attention


def plot_attention(image, words: list, attention: list, out_path=None, backbone="inceptionv3"):
    """One panel per word with that step's attention map over the image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if isinstance(image, (str, Path)):
        image = Image.open(image)
    img = np.asarray(image.convert("RGB"))
    side = int(math.sqrt(BACKBONES[backbone]["locations"]))
    n = max(len(words), 1)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(3.2 * cols, 3.2 * rows), dpi=100)
    for i, (word, attn) in enumerate(zip(words, attention)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(word, fontsize=13)
        ax.imshow(img)
        ax.imshow(np.reshape(attn, (side, side)), cmap="gray", alpha=0.6,
                  extent=(0, img.shape[1], img.shape[0], 0), interpolation="bilinear")
        ax.axis("off")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--run", help="run folder name under checkpoints/ (default: newest)")
    p.add_argument("--beam", type=int, default=3, help="beam width; 1 = greedy")
    p.add_argument("--plot", action="store_true", help="save an attention heatmap")
    a = p.parse_args()
    runs = list_runs()
    run_dir = config.CHECKPOINT_DIR / a.run if a.run else (runs[0] if runs else None)
    if run_dir is None:
        raise SystemExit("No trained run found. Train one with: python -m src.train")
    cap = Captioner(run_dir)
    for beam in sorted({1, a.beam}):
        text, words, attn = cap.caption(a.image, beam=beam)
        print(f"{'greedy' if beam == 1 else f'beam={beam}'}: {text}")
    if a.plot:
        config.RESULTS_DIR.mkdir(exist_ok=True)
        out = config.RESULTS_DIR / f"attention_{Path(a.image).stem}.png"
        plot_attention(a.image, words, attn, out, cap.backbone)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
