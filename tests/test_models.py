import numpy as np

from intentcascade.models import featurize, load_model, tokenize
from intentcascade.registry import list_bundles, load_bundle, save_bundle


def test_featurize_is_deterministic_and_typo_tolerant():
    a = featurize("My card hasn't arrived", 2**16)
    assert a == featurize("My card hasn't arrived", 2**16)
    # A typo changes the word feature but most character n-grams survive.
    b = featurize("My crad hasn't arrived", 2**16)
    assert len(set(a) & set(b)) / len(set(a)) > 0.6
    assert featurize("", 128) == [0]
    assert tokenize("£20 top-up?") == ["£", "20", "top", "up", "?"]


def test_fasttext_learns_the_smoke_task(small_data, trained):
    _, test, labels = small_data
    model, scaler = trained
    probs = scaler(model.logits([e.text for e in test]))
    y = np.array(labels.encode([e.label for e in test]))
    assert (probs.argmax(1) == y).mean() > 0.85
    assert np.allclose(probs.sum(1), 1.0)


def test_embeddings_are_normalised(trained):
    model, _ = trained
    vecs = model.embed(["where is my card", "exchange rate for euros"])
    assert vecs.shape == (2, 32)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_bundle_roundtrip(tmp_path, trained, small_data):
    model, scaler = trained
    _, test, _ = small_data
    texts = [e.text for e in test[:20]]
    save_bundle(tmp_path, "fasttext-test", model, scaler, {"test": {"accuracy": 0.9}}, {"x": 1})
    bundle = load_bundle(tmp_path)
    assert bundle.name == "fasttext-test"
    assert bundle.scaler.temperature == scaler.temperature
    assert np.allclose(bundle.model.logits(texts), model.logits(texts), atol=1e-5)
    assert np.allclose(load_model(bundle.path).logits(texts), model.logits(texts), atol=1e-5)
    assert list_bundles(tmp_path)[0]["latest"] is True
