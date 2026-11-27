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


def test_available_runs_prefers_checkpoints_then_model_dir(tmp_path, monkeypatch):
    empty = tmp_path / "none"
    monkeypatch.setattr(config, "BUNDLED_MODEL_DIR", tmp_path / "no-model")
    assert caption.available_runs(empty) == []

    run = _training_run(tmp_path)          # tmp_path/checkpoints/tiny
    monkeypatch.setattr(config, "BUNDLED_MODEL_DIR", run.parent)
    assert caption.available_runs(empty) == [run]        # falls back to model/
    assert caption.available_runs(run.parent) == [run]   # checkpoints/ first
