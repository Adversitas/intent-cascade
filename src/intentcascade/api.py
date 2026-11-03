"""REST API.

POST /classify   classify one or more messages (logged to SQL)
POST /feedback   attach the correct intent to a logged prediction
GET  /stats      traffic and live accuracy by route
GET  /drift      confidence drift vs the training-time reference
GET  /metrics    Prometheus metrics
GET  /health     model version and configuration
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .db import PredictionLog
from .monitoring import Metrics, drift_report
from .service import Service, build_service


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class DecisionOut(BaseModel):
    id: int
    text: str
    label: str | None
    route: str
    confidence: float
    candidates: list[str]
    latency_ms: float
    reason: str = ""


class FeedbackRequest(BaseModel):
    prediction_id: int
    true_label: str


def create_app(
    service: Service | None = None, prediction_log: PredictionLog | None = None
) -> FastAPI:
    app = FastAPI(
        title="intent-cascade",
        description="Intent classification with confidence-based LLM fallback.",
        version="0.1.0",
    )
    state: dict = {"service": service, "log": prediction_log, "metrics": Metrics()}

    def svc() -> Service:
        if state["service"] is None:
            state["service"] = build_service()
        return state["service"]

    def db() -> PredictionLog:
        if state["log"] is None:
            state["log"] = PredictionLog(os.environ.get("DATABASE_URL", "sqlite:///predictions.db"))
        return state["log"]

    @app.get("/health")
    def health() -> dict:
        s = svc()
        return {
            "status": "ok",
            "model_version": s.version,
            "threshold": s.cascade.threshold,
            "llm_enabled": s.cascade.llm is not None,
            "llm_model": getattr(s.cascade.llm, "model", None),
            "vector_store": type(s.cascade.store).__name__,
            "n_examples": len(s.cascade.store),
            "n_intents": len(s.cascade.model.labels),
        }

    @app.post("/classify", response_model=list[DecisionOut])
    def classify(req: ClassifyRequest) -> list[DecisionOut]:
        texts = [t.strip() for t in req.texts]
        if any(not t for t in texts):
            raise HTTPException(422, "texts must be non-empty")
        s = svc()
        decisions = s.cascade.predict(texts)
        ids = db().record(decisions, s.version)
        state["metrics"].observe(decisions)
        return [
            DecisionOut(
                id=i,
                text=d.text,
                label=d.label,
                route=d.route,
                confidence=round(d.confidence, 4),
                candidates=d.candidates[:5],
                latency_ms=round(d.latency_ms, 2),
                reason=d.reason,
            )
            for i, d in zip(ids, decisions)
        ]

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        if req.true_label not in svc().cascade.model.labels.index:
            raise HTTPException(422, f"unknown intent {req.true_label!r}")
        if not db().add_feedback(req.prediction_id, req.true_label):
            raise HTTPException(404, f"prediction {req.prediction_id} not found")
        return {"ok": True}

    @app.get("/stats")
    def stats() -> dict:
        return db().stats()

    @app.get("/drift")
    def drift(window: int = 1000) -> dict:
        s = svc()
        conf = db().confidences(model_version=s.version, limit=window)
        if not conf:
            raise HTTPException(404, "no predictions logged for the current model yet")
        report = drift_report(s.reference_histogram, conf, s.cascade.threshold)
        state["metrics"].psi.set(report["psi"])
        return report

    @app.get("/metrics")
    def metrics() -> Response:
        body, content_type = state["metrics"].render()
        return Response(body, media_type=content_type)

    return app


app = create_app()
