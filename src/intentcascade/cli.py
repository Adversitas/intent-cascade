"""Command-line interface.

intent-cascade train      [--kind fasttext|embedding-mlp]
intent-cascade benchmark  TF-IDF baseline vs registered models on the test set
intent-cascade evaluate   threshold table for a registered model
intent-cascade classify   "my card hasn't arrived"
intent-cascade llm-eval   run the full cascade (needs ANTHROPIC_API_KEY)
intent-cascade drift      drift report from the prediction log
intent-cascade models     list registered models
intent-cascade serve
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from .data import DATA_DIR, LabelSet, load_csv


def _fmt(x) -> str:
    return "  -  " if x is None or x != x else f"{x:.3f}"


def print_threshold_table(rows: list[dict]) -> None:
    print("| threshold | handled by model | sent to LLM | accuracy on handled |")
    print("|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['threshold']:.2f} | {r['coverage']:.1%} | {r['deferred']:.1%} "
            f"| {_fmt(r['accuracy_kept'])} |"
        )


def cmd_train(args) -> None:
    from .models import TrainConfig
    from .training import train_and_register

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
    kwargs = {"encoder_name": args.encoder} if args.kind == "embedding-mlp" else {}
    path, metrics = train_and_register(
        args.kind, Path(args.train), Path(args.test), Path(args.registry), cfg, kwargs
    )
    t = metrics["test"]
    print(f"\nregistered {path}")
    print(
        f"test accuracy {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}  "
        f"top-10 {t['top10_accuracy']:.4f}  ECE {t['ece']:.4f} "
        f"(uncalibrated {metrics['test_uncalibrated']['ece']:.4f})"
    )
    print_threshold_table(t["thresholds"])


def cmd_benchmark(args) -> None:
    from .registry import list_bundles, load_bundle
    from .training import tfidf_baseline

    train, test = load_csv(args.train), load_csv(args.test)
    labels = LabelSet([e.label for e in train])
    rows = [("TF-IDF + logistic regression", tfidf_baseline(train, test, labels))]
    for b in list_bundles(args.registry):
        bundle = load_bundle(args.registry, b["name"])
        rows.append((bundle.name, bundle.metrics["test"]))

    print("| model | accuracy | macro-F1 | top-10 | ECE | AURC | coverage @95% acc | ms/msg |")
    print("|---|---|---|---|---|---|---|---|")
    for name, m in rows:
        print(
            f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['top10_accuracy']:.4f} "
            f"| {m['ece']:.4f} | {m['aurc']:.4f} | {m['coverage_at_95_accuracy']:.1%} "
            f"| {m.get('latency_ms_per_message', float('nan')):.3f} |"
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(rows), indent=2))


def cmd_evaluate(args) -> None:
    from .registry import load_bundle
    from .training import evaluate_split

    bundle = load_bundle(args.registry, args.model)
    metrics, _, _ = evaluate_split(bundle.model, bundle.scaler, load_csv(args.test))
    print(f"{bundle.name}: accuracy {metrics['accuracy']:.4f}, ECE {metrics['ece']:.4f}")
    print_threshold_table(metrics["thresholds"])


def cmd_classify(args) -> None:
    from .service import build_service

    svc = build_service(args.registry, args.model, threshold=args.threshold, use_llm=args.llm)
    for d in svc.cascade.predict(args.texts):
        print(f"{d.label or '(human)':<35} {d.route:<6} conf={d.confidence:.3f}  {d.text}")
        if d.route != "model":
            print(f"    model guessed {d.model_label}; candidates: {', '.join(d.candidates[:5])}")


def stratified_sample(examples, n: int, seed: int = 0):
    by_label = defaultdict(list)
    for e in examples:
        by_label[e.label].append(e)
    rng = random.Random(seed)
    per_label = max(1, n // len(by_label))
    out = []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        out += group[:per_label]
    return out


def cmd_llm_eval(args) -> None:
    import numpy as np

    from .service import build_service

    svc = build_service(args.registry, args.model, threshold=args.threshold, use_llm=True)
    sample = stratified_sample(load_csv(args.test), args.n)
    texts, gold = [e.text for e in sample], [e.label for e in sample]
    print(f"running cascade on {len(sample)} test messages (threshold {args.threshold})...")
    decisions = svc.cascade.predict(texts)

    routes = np.array([d.route for d in decisions])
    model_correct = np.array([d.model_label == g for d, g in zip(decisions, gold)])
    final_correct = np.array([d.label == g for d, g in zip(decisions, gold)])
    llm = routes == "llm"
    deferred = routes != "model"
    automated = routes != "human"
    report = {
        "n": len(sample),
        "threshold": args.threshold,
        "model_only_accuracy": float(model_correct.mean()),
        "cascade_accuracy_all": float(final_correct.mean()),
        "cascade_accuracy_automated": float(final_correct[automated].mean()),
        "share_model": float((routes == "model").mean()),
        "share_llm": float(llm.mean()),
        "share_human": float((routes == "human").mean()),
        "model_accuracy_on_deferred": float(model_correct[deferred].mean())
        if deferred.any()
        else None,
        "llm_accuracy_on_answered": float(final_correct[llm].mean()) if llm.any() else None,
        "llm_input_tokens": int(sum(d.llm_input_tokens for d in decisions)),
        "llm_output_tokens": int(sum(d.llm_output_tokens for d in decisions)),
        "latency_ms_p50": float(np.percentile([d.latency_ms for d in decisions], 50)),
        "latency_ms_p95": float(np.percentile([d.latency_ms for d in decisions], 95)),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "text": d.text,
                "gold": g,
                "label": d.label,
                "route": d.route,
                "model_label": d.model_label,
                "confidence": d.confidence,
                "candidates": d.candidates,
            }
            for d, g in zip(decisions, gold)
        ]
        Path(args.output).write_text(json.dumps({"summary": report, "examples": rows}, indent=2))


def cmd_drift(args) -> None:
    from .db import PredictionLog
    from .monitoring import drift_report
    from .registry import load_bundle

    bundle = load_bundle(args.registry, args.model)
    conf = PredictionLog(args.db).confidences(bundle.name, limit=args.window)
    if not conf:
        sys.exit(f"no predictions logged for {bundle.name} in {args.db}")
    ref = bundle.metrics["test"]["confidence_histogram"]
    print(json.dumps(drift_report(ref, conf, args.threshold), indent=2))


def cmd_models(args) -> None:
    from .registry import list_bundles

    for b in list_bundles(args.registry):
        mark = "*" if b["latest"] else " "
        print(
            f"{mark} {b['name']:<35} test_acc={_fmt(b['test_accuracy'])} ece={_fmt(b['test_ece'])}"
        )


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("intentcascade.api:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="intent-cascade",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--registry", default="models", help="model registry directory")
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(sp):
        sp.add_argument("--train", default=str(DATA_DIR / "train.csv"))
        sp.add_argument("--test", default=str(DATA_DIR / "test.csv"))

    sp = sub.add_parser("train", help="train, calibrate, evaluate and register a model")
    data_args(sp)
    sp.add_argument("--kind", default="fasttext", choices=["fasttext", "embedding-mlp"])
    sp.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--lr", type=float, default=3e-3)
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("benchmark", help="compare the TF-IDF baseline with registered models")
    data_args(sp)
    sp.add_argument("--output")
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("evaluate", help="threshold table for a registered model")
    data_args(sp)
    sp.add_argument("--model")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("classify", help="classify messages from the command line")
    sp.add_argument("texts", nargs="+")
    sp.add_argument("--model")
    sp.add_argument("--threshold", type=float, default=0.6)
    sp.add_argument("--llm", action="store_true", help="use the LLM fallback")
    sp.set_defaults(func=cmd_classify)

    sp = sub.add_parser("llm-eval", help="evaluate the full cascade with the LLM fallback")
    data_args(sp)
    sp.add_argument("--model")
    sp.add_argument("--threshold", type=float, default=0.6)
    sp.add_argument("-n", type=int, default=308, help="stratified sample size")
    sp.add_argument("--output")
    sp.set_defaults(func=cmd_llm_eval)

    sp = sub.add_parser("drift", help="confidence drift report from the prediction log")
    sp.add_argument("--db", default="sqlite:///predictions.db")
    sp.add_argument("--model")
    sp.add_argument("--window", type=int, default=1000)
    sp.add_argument("--threshold", type=float, default=0.6)
    sp.set_defaults(func=cmd_drift)

    sp = sub.add_parser("models", help="list registered models")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("serve", help="run the REST API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
