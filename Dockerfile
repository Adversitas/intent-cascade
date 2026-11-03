# Two stages: the first trains the model, so the image always ships with a model
# that matches the code. Swap the training stage for a registry pull in production.
FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,postgres]"
COPY data ./data

FROM base AS train
RUN intent-cascade train --kind fasttext

FROM base
COPY --from=train /app/models ./models
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
# Optional at runtime: ANTHROPIC_API_KEY (enables the LLM fallback),
# DATABASE_URL, VECTOR_DB_URL, INTENT_CASCADE_THRESHOLD.
CMD ["uvicorn", "intentcascade.api:app", "--host", "0.0.0.0", "--port", "8000"]
