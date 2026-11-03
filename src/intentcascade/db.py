"""Prediction log in SQL.

Every decision is written to a ``predictions`` table, and agents can attach the
correct intent afterwards through ``/feedback``. The table is what monitoring,
drift reports and future retraining read from. SQLite is the default; set
``DATABASE_URL`` to a PostgreSQL URL (``postgresql+psycopg://...``) in production.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    case,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .cascade import Decision


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    model_version: Mapped[str] = mapped_column(String(100), index=True)
    text: Mapped[str] = mapped_column(Text)
    label: Mapped[str | None] = mapped_column(String(100))
    model_label: Mapped[str] = mapped_column(String(100))
    route: Mapped[str] = mapped_column(String(10), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    llm_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    # Filled in later by a human agent via /feedback.
    true_label: Mapped[str | None] = mapped_column(String(100), nullable=True)


class PredictionLog:
    def __init__(self, url: str = "sqlite:///predictions.db") -> None:
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)

    def record(self, decisions: list[Decision], model_version: str) -> list[int]:
        rows = [
            Prediction(
                model_version=model_version,
                text=d.text,
                label=d.label,
                model_label=d.model_label,
                route=d.route,
                confidence=d.confidence,
                latency_ms=d.latency_ms,
                llm_input_tokens=d.llm_input_tokens,
                llm_output_tokens=d.llm_output_tokens,
            )
            for d in decisions
        ]
        with Session(self.engine) as s:
            s.add_all(rows)
            s.commit()
            return [r.id for r in rows]

    def add_feedback(self, prediction_id: int, true_label: str) -> bool:
        with Session(self.engine) as s:
            row = s.get(Prediction, prediction_id)
            if row is None:
                return False
            row.true_label = true_label
            s.commit()
            return True

    def confidences(self, model_version: str | None = None, limit: int = 5000) -> list[float]:
        q = select(Prediction.confidence).order_by(Prediction.id.desc()).limit(limit)
        if model_version:
            q = q.where(Prediction.model_version == model_version)
        with Session(self.engine) as s:
            return list(s.scalars(q))

    def stats(self) -> dict:
        """Traffic by route, plus live accuracy wherever feedback exists."""
        with Session(self.engine) as s:
            by_route = s.execute(
                select(
                    Prediction.route,
                    func.count(),
                    func.avg(Prediction.confidence),
                    func.avg(Prediction.latency_ms),
                    func.count(Prediction.true_label),
                    func.sum(case((Prediction.label == Prediction.true_label, 1), else_=0)),
                ).group_by(Prediction.route)
            ).all()
        routes = {}
        for route, n, conf, lat, n_fb, n_correct in by_route:
            routes[route] = {
                "count": n,
                "mean_confidence": round(conf or 0.0, 4),
                "mean_latency_ms": round(lat or 0.0, 2),
                "with_feedback": n_fb,
                "accuracy_on_feedback": round((n_correct or 0) / n_fb, 4) if n_fb else None,
            }
        total = sum(r["count"] for r in routes.values())
        return {"total": total, "routes": routes}
