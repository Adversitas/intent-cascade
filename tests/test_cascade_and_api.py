import pytest
from conftest import FakeMessages, fake_client
from fastapi.testclient import TestClient

from intentcascade.api import create_app
from intentcascade.cascade import Cascade
from intentcascade.db import PredictionLog
from intentcascade.llm import NONE_LABEL, ClaudeIntentClassifier, LLMError, build_schema
from intentcascade.registry import load_bundle, save_bundle
from intentcascade.service import Service
from intentcascade.vectorstore import NumpyVectorStore


@pytest.fixture
def cascade(trained, small_data):
    model, scaler = trained
    train, _, _ = small_data
    c = Cascade(model, scaler, threshold=0.6, n_candidates=3, n_examples=4)
    c.index_examples([e.text for e in train], [e.label for e in train])
    return c


def test_vector_store_returns_nearest_first():
    import numpy as np

    store = NumpyVectorStore()
    store.add(["a", "b", "c"], ["x", "y", "z"], np.eye(3, dtype=np.float32))
    hits = store.search(np.array([0.1, 0.9, 0.0]), k=2)
    assert [h.text for h in hits] == ["b", "a"]
    assert len(store) == 3


def test_retrieved_neighbours_share_the_intent(cascade):
    vec = cascade.model.embed(["I lost my card on the train yesterday"])[0]
    labels = [n.label for n in cascade.store.search(vec, k=5)]
    assert labels.count("lost_or_stolen_card") >= 3


def test_confident_messages_stay_with_the_model(cascade):
    cascade.threshold = 0.0
    [d] = cascade.predict(["What exchange rate will I get?"])
    assert d.route == "model" and d.label == "exchange_rate"


def test_without_llm_low_confidence_goes_to_human(cascade):
    cascade.threshold = 1.01
    [d] = cascade.predict(["hello"])
    assert d.route == "human" and d.label is None


def test_llm_fallback_receives_candidates_and_examples(cascade):
    messages = FakeMessages({"label": "card_arrival"})
    cascade.llm = ClaudeIntentClassifier(client=fake_client(messages))
    cascade.threshold = 1.01
    [d] = cascade.predict(["still waiting for the thing you posted me"])
    assert d.route == "llm" and d.label == "card_arrival"
    assert d.llm_input_tokens == 500

    call = messages.calls[0]
    enum = call["output_config"]["format"]["schema"]["properties"]["label"]["enum"]
    assert NONE_LABEL in enum and set(d.candidates) == set(enum) - {NONE_LABEL}
    assert "<examples>" in call["messages"][0]["content"]
    assert call["fallbacks"] == "default"


def test_llm_none_of_these_hands_off_to_human(cascade):
    cascade.llm = ClaudeIntentClassifier(client=fake_client(FakeMessages({"label": NONE_LABEL})))
    cascade.threshold = 1.01
    [d] = cascade.predict(["can you recommend a good pizza place"])
    assert d.route == "human" and "no matching intent" in d.reason


def test_llm_errors_degrade_to_human(cascade):
    messages = FakeMessages({}, stop_reason="refusal")
    cascade.llm = ClaudeIntentClassifier(client=fake_client(messages))
    cascade.threshold = 1.01
    [d] = cascade.predict(["anything"])
    assert d.route == "human" and "LLM error" in d.reason
    with pytest.raises(LLMError):
        cascade.llm.classify("x", ["card_arrival"], [])


def test_schema_restricts_labels():
    schema = build_schema(["a", "b"])
    assert schema["properties"]["label"]["enum"] == ["a", "b", NONE_LABEL]


@pytest.fixture
def client(tmp_path, cascade):
    save_bundle(
        tmp_path / "models",
        "fasttext-test",
        cascade.model,
        cascade.scaler,
        {"test": {"confidence_histogram": [0.1] * 10}},
        {},
    )
    cascade.threshold = 0.6
    service = Service(bundle=load_bundle(tmp_path / "models"), cascade=cascade)
    log = PredictionLog(f"sqlite:///{tmp_path / 'p.db'}")
    return TestClient(create_app(service, log))


def test_api_classify_feedback_stats_drift_metrics(client):
    assert client.get("/health").json()["model_version"] == "fasttext-test"

    r = client.post("/classify", json={"texts": ["my card still hasn't arrived", "hmm"]})
    assert r.status_code == 200
    first, second = r.json()
    assert first["label"] == "card_arrival" and first["route"] == "model"

    assert client.post(
        "/feedback", json={"prediction_id": first["id"], "true_label": "card_arrival"}
    ).json() == {"ok": True}
    assert (
        client.post(
            "/feedback", json={"prediction_id": 999, "true_label": "card_arrival"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/feedback", json={"prediction_id": first["id"], "true_label": "nope"}
        ).status_code
        == 422
    )

    stats = client.get("/stats").json()
    assert stats["total"] == 2
    assert stats["routes"]["model"]["accuracy_on_feedback"] == 1.0

    drift = client.get("/drift").json()
    assert drift["n"] == 2 and drift["status"] in {"stable", "moderate", "significant"}

    metrics = client.get("/metrics").text
    assert 'intent_requests_total{route="model"}' in metrics
    assert "intent_confidence_psi" in metrics


def test_api_rejects_empty_input(client):
    assert client.post("/classify", json={"texts": []}).status_code == 422
    assert client.post("/classify", json={"texts": ["  "]}).status_code == 422
