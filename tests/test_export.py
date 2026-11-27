import json

import tensorflow as tf

from src import caption, config
from src.export import export_run
from src.model import build_models

VOCAB = ["", "[UNK]", "<start>", "<end>", "a", "dog", "cat", "on", "grass"]


def _training_run(tmp_path):
    """A tiny 'trained' run folder: random weights saved with an optimizer, as train.py does."""
    run = tmp_path / "checkpoints" / "tiny"
    run.mkdir(parents=True)
    cfg = {"run": "tiny", "backbone": "inceptionv3", "vocab_size": len(VOCAB),
           "max_length": 6, "embedding_dim": 16, "units": 8, "num_images": 10, "epochs": 1}
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "vocab.json").write_text(json.dumps(VOCAB))
    tf.keras.utils.set_random_seed(1)
    enc, dec = build_models(len(VOCAB), 16, 8, 64)
    opt = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(encoder=enc, decoder=dec, optimizer=opt,
                               epoch=tf.Variable(1, dtype=tf.int64))
    tf.train.CheckpointManager(ckpt, str(run), max_to_keep=1).save()
    return run


def test_export_roundtrip_gives_same_caption(tmp_path):
    run = _training_run(tmp_path)
    out = export_run(run, tmp_path / "export" / "tiny")
    assert {"config.json", "vocab.json", "checkpoint"} <= {p.name for p in out.iterdir()}
    feats = tf.random.stateless_normal((1, 64, 2048), seed=(3, 4))
    original = caption.Captioner(run).caption(feats=feats, beam=1)[0]
    exported = caption.Captioner(out).caption(feats=feats, beam=1)[0]
    assert exported == original


def test_find_or_download_prefers_local_then_hub(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "HF_MODEL_REPO", "someone/model")
    monkeypatch.setattr(config, "RELEASE_CHECKPOINT_URL", "")
    monkeypatch.setattr(caption, "download_hf_checkpoint",
                        lambda checkpoint_dir: calls.append(checkpoint_dir) or ["hub-run"])
    empty = tmp_path / "none"
    assert caption.find_or_download_runs(empty) == ["hub-run"] and calls == [empty]

    run = _training_run(tmp_path)
    calls.clear()
    assert caption.find_or_download_runs(run.parent) == [run] and calls == []
