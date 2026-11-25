import pytest

from src.evaluate import compute_metrics, markdown_table, upsert_row
from src.video.describe import dedupe

REFS = [["a dog runs on the grass", "a brown dog running across a field"],
        ["a man rides a red bike", "a person cycling down the street"]]


def test_perfect_predictions():
    m = compute_metrics(REFS, ["a dog runs on the grass", "a man rides a red bike"])
    for k in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"]:
        assert m[k] == pytest.approx(1.0)
    assert m["WER"] == pytest.approx(0.0)
    assert m["METEOR"] > 0.9


def test_bad_predictions_score_lower():
    good = compute_metrics(REFS, ["a dog runs on grass", "a man rides a bike"])
    bad = compute_metrics(REFS, ["pizza", "blue sky"])
    for k in ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L"]:
        assert good[k] > bad[k]
    assert good["WER"] < bad["WER"]


def test_empty_prediction_does_not_crash():
    m = compute_metrics(REFS[:1], [""])
    assert m["BLEU-1"] == pytest.approx(0.0)


def test_metrics_csv_upsert(tmp_path):
    row = {"run": "r1", "backbone": "inceptionv3", "vocab": 5000, "images": 200,
           "epochs": 1, "decoding": "greedy", "BLEU-1": 0.5, "BLEU-2": 0.3, "BLEU-3": 0.2,
           "BLEU-4": 0.1, "METEOR": 0.2, "ROUGE-L": 0.4, "WER": 0.8}
    path = tmp_path / "metrics.csv"
    upsert_row(row, path)
    rows = upsert_row({**row, "BLEU-1": 0.6}, path)
    assert len(rows) == 1 and float(rows[0]["BLEU-1"]) == 0.6
    assert "inceptionv3" in markdown_table(rows)


def test_video_dedupe_consecutive_only():
    entries = [(0, "a dog"), (2, "a dog"), (4, "a cat"), (6, "a dog")]
    assert dedupe(entries) == [(0, "a dog"), (4, "a cat"), (6, "a dog")]
