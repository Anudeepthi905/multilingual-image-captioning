from src.data import clean_caption, flatten, get_split, split_by_image, strip_tokens


def test_clean_caption_lowercases_and_wraps():
    assert clean_caption("A Dog runs.") == "<start> a dog runs <end>"


def test_clean_caption_removes_special_characters_and_spaces():
    raw = "  Two   people, (one wearing a hat!) & a dog...\n"
    assert clean_caption(raw) == "<start> two people one wearing a hat a dog <end>"


def test_clean_caption_keeps_digits():
    assert clean_caption("3 cats on 1 bed") == "<start> 3 cats on 1 bed <end>"


def test_strip_tokens_roundtrip():
    assert strip_tokens(clean_caption("A red bus")) == "a red bus"


def _items(n):
    return [{"image_path": f"img{i}.jpg", "captions": [f"<start> c{i} <end>"] * 5}
            for i in range(n)]


def test_split_is_by_image_without_overlap():
    train, val = split_by_image(_items(10), 0.2)
    assert len(train) == 8 and len(val) == 2
    assert not {i["image_path"] for i in train} & {i["image_path"] for i in val}


def test_get_split_is_deterministic_prefix():
    manifest = {"images": _items(20), "val_fraction": 0.2}
    a = get_split(manifest, 10)
    b = get_split(manifest, 10)
    assert a == b
    assert [i["image_path"] for i in a[0] + a[1]] == [f"img{i}.jpg" for i in range(10)]


def test_flatten_one_row_per_caption():
    paths, caps = flatten(_items(3))
    assert len(paths) == len(caps) == 15
