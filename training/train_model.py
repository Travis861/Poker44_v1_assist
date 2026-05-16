from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from poker44.score.scoring import reward
from poker44_ml.inference import Poker44Model
from training.build_dataset import (
    build_human_chunk_examples,
    load_benchmark_examples,
    load_human_hands,
    resolve_benchmark_paths,
    resolve_human_path,
)

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover
    ExtraTreesClassifier = None
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None
    LogisticRegression = None
    average_precision_score = None
    brier_score_loss = None
    log_loss = None
    matthews_corrcoef = None
    roc_auc_score = None
    train_test_split = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clean Poker44 benchmark model.")
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--human-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "models" / "poker44_clean_restart.joblib"))
    parser.add_argument("--holdout-latest-days", type=int, default=2)
    parser.add_argument("--holdout-source-dates", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--extra-trees-weight", type=float, default=0.45)
    parser.add_argument("--random-forest-weight", type=float, default=0.25)
    parser.add_argument("--hist-gradient-weight", type=float, default=0.30)
    parser.add_argument("--enable-aux-human", action="store_true")
    parser.add_argument("--aux-human-weight", type=float, default=3.0)
    parser.add_argument("--max-aux-human-chunks", type=int, default=5000)
    parser.add_argument("--aux-human-calibration-fraction", type=float, default=0.25)
    parser.add_argument("--aux-human-chunk-sizes", type=str, default="38,40,48,56,64,72,80,88")
    parser.add_argument("--min-chunk-size", type=int, default=20)
    parser.add_argument("--human-guard-strength", type=float, default=0.18)
    parser.add_argument("--human-guard-quantile", type=float, default=0.995)
    parser.add_argument("--feature-distance-calibration", action="store_true")
    parser.add_argument("--feature-distance-strength", type=float, default=0.08)
    parser.add_argument("--feature-distance-max-shift", type=float, default=0.025)
    return parser.parse_args()


def _repo_metadata() -> dict[str, str]:
    def run(args: list[str]) -> str:
        try:
            completed = subprocess.run(
                args,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return ""
        return completed.stdout.strip()

    return {
        "repo_commit": run(["git", "rev-parse", "HEAD"]),
        "repo_url": run(["git", "config", "--get", "remote.origin.url"]),
    }


def _feature_schema_hash(feature_names: list[str]) -> str:
    payload = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_matrix(examples: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    return [
        [float(example["features"].get(name, 0.0)) for name in feature_names]
        for example in examples
    ]


def _split_benchmark(
    examples: list[dict[str, Any]],
    *,
    holdout_source_dates: str | None,
    holdout_latest_days: int,
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dates = sorted(
        {
            str(example.get("source_date", "")).strip()
            for example in examples
            if str(example.get("source_date", "")).strip()
        }
    )
    requested = [item.strip() for item in str(holdout_source_dates or "").split(",") if item.strip()]
    holdout_dates = requested or dates[-max(1, int(holdout_latest_days)) :]
    if holdout_dates:
        holdout_set = set(holdout_dates)
        train = [example for example in examples if str(example.get("source_date", "")).strip() not in holdout_set]
        test = [example for example in examples if str(example.get("source_date", "")).strip() in holdout_set]
        if train and test and len({int(example["label"]) for example in test}) >= 2:
            return train, test, {
                "split_strategy": "holdout_source_dates",
                "holdout_source_dates": holdout_dates,
                "train_source_dates": [date for date in dates if date not in holdout_set],
            }

    labels = [int(example["label"]) for example in examples]
    train, test = train_test_split(
        examples,
        test_size=min(max(float(test_size), 0.05), 0.45),
        random_state=seed,
        stratify=labels,
    )
    return train, test, {
        "split_strategy": "random_stratified",
        "holdout_source_dates": [],
        "train_source_dates": dates,
    }


def _split_calibration(
    examples: list[dict[str, Any]],
    *,
    calibration_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [int(example["label"]) for example in examples]
    if len(examples) < 30 or len(set(labels)) < 2 or calibration_size <= 0:
        return examples, []
    fit, calibration = train_test_split(
        examples,
        test_size=min(max(float(calibration_size), 0.05), 0.4),
        random_state=seed,
        stratify=labels,
    )
    return fit, calibration


def _model_scores(models: list[object], weights: list[float], rows: list[list[float]]) -> list[float]:
    per_model: list[list[float]] = []
    for model in models:
        probabilities = model.predict_proba(rows)
        per_model.append([float(row[1]) for row in probabilities])
    clean_weights = [max(0.0, float(weight)) for weight in weights[: len(per_model)]]
    if len(clean_weights) != len(per_model) or sum(clean_weights) <= 0.0:
        clean_weights = [1.0 for _ in per_model]
    total = sum(clean_weights)
    return [
        float(sum(weight * scores[index] for weight, scores in zip(clean_weights, per_model)) / total)
        for index in range(len(rows))
    ]


def _fit_calibrator(labels: list[int], scores: list[float]) -> object | None:
    if len(set(int(label) for label in labels)) < 2:
        return None
    calibrator = LogisticRegression(solver="lbfgs", random_state=42)
    calibrator.fit([[float(score)] for score in scores], labels)
    return calibrator


def _apply_calibrator(calibrator: object | None, scores: list[float]) -> list[float]:
    if calibrator is None:
        return [max(0.0, min(1.0, float(score))) for score in scores]
    probabilities = calibrator.predict_proba([[float(score)] for score in scores])
    return [max(0.0, min(1.0, float(row[1]))) for row in probabilities]


def _median_iqr_profile(rows: list[list[float]]) -> dict[str, list[float]]:
    if not rows:
        return {}
    values = np.asarray(rows, dtype=float)
    center = np.median(values, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return {
        "center": [float(value) for value in center],
        "scale": [float(value) for value in scale],
    }


def _robust_distances(rows: list[list[float]], profile: dict[str, Any]) -> list[float]:
    if not rows or not profile:
        return []
    center = np.asarray(profile.get("center") or [], dtype=float)
    scale = np.asarray(profile.get("scale") or [], dtype=float)
    if center.size == 0 or scale.size != center.size:
        return [0.0 for _ in rows]
    values = np.asarray(rows, dtype=float)
    if values.shape[1] != center.size:
        return [0.0 for _ in rows]
    z = np.clip(np.abs((values - center) / np.maximum(scale, 1e-6)), 0.0, 12.0)
    return [float(value) for value in np.mean(z, axis=1)]


def _feature_distance_calibration_from_rows(
    human_rows: list[list[float]],
    bot_rows: list[list[float]],
    *,
    strength: float,
    max_shift: float,
) -> dict[str, Any]:
    if not human_rows or not bot_rows or strength <= 0.0 or max_shift <= 0.0:
        return {}
    human_profile = _median_iqr_profile(human_rows)
    bot_profile = _median_iqr_profile(bot_rows)
    human_distances = _robust_distances(human_rows, human_profile)
    human_to_bot = _robust_distances(human_rows, bot_profile)
    bot_to_human = _robust_distances(bot_rows, human_profile)
    bot_distances = _robust_distances(bot_rows, bot_profile)
    margins = [
        human_distance - bot_distance
        for human_distance, bot_distance in zip(human_distances, human_to_bot)
    ] + [
        human_distance - bot_distance
        for human_distance, bot_distance in zip(bot_to_human, bot_distances)
    ]
    margin_scale = float(np.std(margins)) if margins else 0.0
    return {
        "kind": "feature_distance_after_logistic_v1",
        "human_center": human_profile["center"],
        "human_scale": human_profile["scale"],
        "bot_center": bot_profile["center"],
        "bot_scale": bot_profile["scale"],
        "strength": min(max(float(strength), 0.0), 0.5),
        "max_shift": min(max(float(max_shift), 0.0), 0.10),
        "margin_scale": max(margin_scale, 0.25),
    }


def _apply_feature_distance_calibration(
    rows: list[list[float]],
    scores: list[float],
    calibration: dict[str, Any],
) -> list[float]:
    if not scores or not calibration:
        return [max(0.0, min(1.0, float(score))) for score in scores]
    human_profile = {
        "center": calibration.get("human_center") or [],
        "scale": calibration.get("human_scale") or [],
    }
    bot_profile = {
        "center": calibration.get("bot_center") or [],
        "scale": calibration.get("bot_scale") or [],
    }
    human_distances = _robust_distances(rows, human_profile)
    bot_distances = _robust_distances(rows, bot_profile)
    if len(human_distances) != len(scores) or len(bot_distances) != len(scores):
        return [max(0.0, min(1.0, float(score))) for score in scores]
    strength = min(max(float(calibration.get("strength", 0.0)), 0.0), 0.5)
    max_shift = min(max(float(calibration.get("max_shift", 0.0)), 0.0), 0.10)
    margin_scale = max(float(calibration.get("margin_scale", 0.25)), 1e-6)
    output: list[float] = []
    for score, human_distance, bot_distance in zip(scores, human_distances, bot_distances):
        value = max(0.0, min(1.0, float(score)))
        signed_margin = (float(human_distance) - float(bot_distance)) / margin_scale
        shift = max_shift * np.tanh(strength * signed_margin)
        output.append(max(0.0, min(1.0, value + float(shift))))
    return output


def _human_guard_from_scores(scores: list[float], *, quantile: float, strength: float) -> dict[str, float]:
    if not scores or strength <= 0.0:
        return {}
    values = np.asarray(scores, dtype=float)
    anchor = float(np.quantile(values, min(max(float(quantile), 0.50), 0.999)))
    spread = float(np.std(values))
    return {
        "kind": "aux_human_score_guard_v1",
        "anchor": anchor,
        "softness": max(spread * 0.5, 0.015),
        "strength": min(max(float(strength), 0.0), 0.8),
        "quantile": float(quantile),
    }


def _apply_human_guard(scores: list[float], guard: dict[str, float]) -> list[float]:
    if not scores or not guard:
        return [max(0.0, min(1.0, float(score))) for score in scores]
    anchor = float(guard.get("anchor", 0.0))
    softness = max(float(guard.get("softness", 1.0)), 1e-6)
    strength = min(max(float(guard.get("strength", 0.0)), 0.0), 1.0)
    output: list[float] = []
    for score in scores:
        value = max(0.0, min(1.0, float(score)))
        human_like = 1.0 / (1.0 + np.exp((value - anchor) / softness))
        output.append(max(0.0, min(1.0, value * (1.0 - strength * human_like))))
    return output


def _binary_counts(labels: list[int], scores: list[float]) -> dict[str, float]:
    preds = [score >= 0.5 for score in scores]
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred)
    tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and not pred)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and not pred)
    positives = max(sum(1 for label in labels if label == 1), 1)
    negatives = max(sum(1 for label in labels if label == 0), 1)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "recall_at_0_5": float(tp / positives),
        "precision_at_0_5": float(tp / max(tp + fp, 1)),
        "fpr_at_0_5": float(fp / negatives),
    }


def _validator_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    if not labels:
        return {}
    val_reward, details = reward(
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=int),
    )
    return {
        "validator_reward": float(val_reward),
        "validator_fpr": float(details.get("fpr", 1.0)),
        "validator_bot_recall": float(details.get("bot_recall", 0.0)),
        "validator_ap_score": float(details.get("ap_score", 0.0)),
        "validator_human_safety_penalty": float(
            details.get("human_safety_penalty", 0.0)
        ),
        "validator_base_score": float(details.get("base_score", 0.0)),
    }


def _enrich_probability_metrics(
    labels: list[int],
    probabilities: list[float],
    raw_probabilities: list[float] | None = None,
) -> dict[str, float]:
    safe = [max(1e-6, min(1.0 - 1e-6, float(value))) for value in probabilities]
    metrics: dict[str, float] = {}
    if len(set(labels)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, safe))
        metrics["pr_auc"] = float(average_precision_score(labels, safe))
        metrics["mcc_at_0_5"] = float(matthews_corrcoef(labels, [value >= 0.5 for value in safe]))
    metrics["log_loss"] = float(log_loss(labels, safe, labels=[0, 1]))
    metrics["brier_score"] = float(brier_score_loss(labels, safe))
    metrics.update(_binary_counts(labels, safe))
    metrics.update(_validator_metrics(labels, safe))

    humans = [score for label, score in zip(labels, safe) if label == 0]
    bots = [score for label, score in zip(labels, safe) if label == 1]
    metrics["prob_min"] = float(min(safe)) if safe else 0.0
    metrics["prob_max"] = float(max(safe)) if safe else 0.0
    metrics["prob_mean"] = float(sum(safe) / max(len(safe), 1))
    metrics["human_prob_max"] = float(max(humans)) if humans else 0.0
    metrics["bot_prob_min"] = float(min(bots)) if bots else 1.0
    metrics["human_clearance_to_0_5"] = float(0.5 - metrics["human_prob_max"])
    metrics["bot_clearance_to_0_5"] = float(metrics["bot_prob_min"] - 0.5)
    metrics["score_gap_at_0_5"] = float(metrics["bot_prob_min"] - metrics["human_prob_max"])
    metrics["threshold_margin_at_0_5"] = float(
        min(metrics["human_clearance_to_0_5"], metrics["bot_clearance_to_0_5"])
    )
    if raw_probabilities is not None:
        raw = [max(0.0, min(1.0, float(value))) for value in raw_probabilities]
        raw_humans = [score for label, score in zip(labels, raw) if label == 0]
        raw_bots = [score for label, score in zip(labels, raw) if label == 1]
        metrics["raw_human_prob_max"] = float(max(raw_humans)) if raw_humans else 0.0
        metrics["raw_bot_prob_min"] = float(min(raw_bots)) if raw_bots else 1.0
        metrics["raw_score_gap_at_0_5"] = metrics["raw_bot_prob_min"] - metrics["raw_human_prob_max"]
    return metrics


def train_model(args: argparse.Namespace) -> tuple[list[object], list[str], dict[str, float], dict[str, Any]]:
    required = [
        joblib,
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        LogisticRegression,
        train_test_split,
    ]
    if any(item is None for item in required):
        raise RuntimeError("joblib and scikit-learn are required for training.")

    benchmark_paths = resolve_benchmark_paths(args.benchmark_path)
    benchmark_examples = load_benchmark_examples(benchmark_paths)
    labels = [int(example["label"]) for example in benchmark_examples]
    label_counts = Counter(labels)
    if len(label_counts) != 2:
        raise RuntimeError(f"Benchmark must contain both labels, got {dict(label_counts)}")

    train_examples, test_examples, split_info = _split_benchmark(
        benchmark_examples,
        holdout_source_dates=args.holdout_source_dates,
        holdout_latest_days=args.holdout_latest_days,
        test_size=args.test_size,
        seed=args.seed,
    )
    fit_examples, calibration_examples = _split_calibration(
        train_examples,
        calibration_size=args.calibration_size,
        seed=args.seed,
    )

    benchmark_feature_rows = [example["features"] for example in benchmark_examples]
    feature_names = sorted(benchmark_feature_rows[0])
    benchmark_chunk_sizes = sorted({len(example["chunk"]) for example in benchmark_examples})

    aux_train: list[dict[str, Any]] = []
    aux_calibration: list[dict[str, Any]] = []
    human_path = None
    if args.enable_aux_human:
        human_path = resolve_human_path(args.human_path)
        human_hands = load_human_hands(human_path)
        chunk_sizes = [
            int(item.strip())
            for item in str(args.aux_human_chunk_sizes).split(",")
            if item.strip()
        ] or benchmark_chunk_sizes
        aux_examples = build_human_chunk_examples(
            human_hands,
            chunk_sizes=chunk_sizes,
            count=max(0, int(args.max_aux_human_chunks)),
            min_chunk_size=max(1, int(args.min_chunk_size)),
            seed=args.seed,
            source_path=str(human_path),
        )
        if aux_examples and args.aux_human_calibration_fraction > 0:
            aux_train, aux_calibration = train_test_split(
                aux_examples,
                test_size=min(max(float(args.aux_human_calibration_fraction), 0.05), 0.5),
                random_state=args.seed,
                shuffle=True,
            )
        else:
            aux_train = aux_examples

    X_fit = _build_matrix(fit_examples + aux_train, feature_names)
    y_fit = [int(example["label"]) for example in fit_examples] + [0 for _ in aux_train]
    weights = [1.0 for _ in fit_examples] + [float(args.aux_human_weight) for _ in aux_train]

    X_cal = _build_matrix(calibration_examples + aux_calibration, feature_names)
    y_cal = [int(example["label"]) for example in calibration_examples] + [0 for _ in aux_calibration]
    X_test = _build_matrix(test_examples, feature_names)
    y_test = [int(example["label"]) for example in test_examples]

    models: list[object] = [
        ExtraTreesClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=max(100, int(args.n_estimators // 2)),
            max_depth=int(args.max_depth),
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=args.seed + 7,
            n_jobs=1,
        ),
        HistGradientBoostingClassifier(
            learning_rate=float(args.learning_rate),
            max_iter=int(args.n_estimators),
            max_depth=int(args.max_depth),
            min_samples_leaf=2,
            random_state=args.seed + 13,
        ),
    ]
    for model in models:
        model.fit(X_fit, y_fit, sample_weight=weights)

    model_weights = [
        float(args.extra_trees_weight),
        float(args.random_forest_weight),
        float(args.hist_gradient_weight),
    ]
    raw_cal = _model_scores(models, model_weights, X_cal) if X_cal else []
    calibrator = _fit_calibrator(y_cal, raw_cal) if raw_cal and len(set(y_cal)) >= 2 else None
    calibrated_cal = _apply_calibrator(calibrator, raw_cal)

    calibration_human_examples = [
        example for example in (fit_examples + calibration_examples) if int(example["label"]) == 0
    ]
    calibration_bot_examples = [
        example for example in (fit_examples + calibration_examples) if int(example["label"]) == 1
    ]
    feature_distance_calibration = (
        _feature_distance_calibration_from_rows(
            _build_matrix(calibration_human_examples, feature_names),
            _build_matrix(calibration_bot_examples, feature_names),
            strength=args.feature_distance_strength,
            max_shift=args.feature_distance_max_shift,
        )
        if args.feature_distance_calibration
        else {}
    )
    feature_calibrated_cal = _apply_feature_distance_calibration(
        X_cal,
        calibrated_cal,
        feature_distance_calibration,
    )

    aux_cal_count = len(aux_calibration)
    aux_cal_scores = feature_calibrated_cal[-aux_cal_count:] if aux_cal_count else []
    human_guard = _human_guard_from_scores(
        aux_cal_scores,
        quantile=args.human_guard_quantile,
        strength=args.human_guard_strength,
    )

    raw_test = _model_scores(models, model_weights, X_test)
    calibrated_test = _apply_calibrator(calibrator, raw_test)
    feature_calibrated_test = _apply_feature_distance_calibration(
        X_test,
        calibrated_test,
        feature_distance_calibration,
    )
    final_test = _apply_human_guard(feature_calibrated_test, human_guard)
    metrics = _enrich_probability_metrics(y_test, final_test, raw_probabilities=raw_test)

    metadata: dict[str, Any] = {
        "framework": "clean-restart:ExtraTrees+RandomForest+HistGradientBoosting",
        "task_type": "supervised-benchmark-clean-restart",
        **_repo_metadata(),
        "feature_schema_hash": _feature_schema_hash(feature_names),
        "benchmark_paths": [str(path) for path in benchmark_paths],
        "benchmark_rows": float(len(benchmark_examples)),
        "benchmark_positive_rows": float(label_counts.get(1, 0)),
        "benchmark_negative_rows": float(label_counts.get(0, 0)),
        "benchmark_chunk_sizes": ",".join(str(size) for size in benchmark_chunk_sizes),
        "train_rows": float(len(X_fit)),
        "calibration_rows": float(len(X_cal)),
        "test_rows": float(len(X_test)),
        "aux_human_rows": float(len(aux_train)),
        "aux_human_calibration_rows": float(len(aux_calibration)),
        "aux_human_weight": float(args.aux_human_weight),
        "human_path": str(human_path or ""),
        "human_guard": human_guard,
        "feature_distance_calibration": feature_distance_calibration,
        "feature_distance_calibration_enabled": 1.0 if feature_distance_calibration else 0.0,
        "probability_calibration": (
            "logistic+feature_distance_after_logistic_v1"
            if feature_distance_calibration
            else "logistic"
        ),
        "model_weights": model_weights,
        "n_estimators": float(args.n_estimators),
        "max_depth": float(args.max_depth),
        "learning_rate": float(args.learning_rate),
        **split_info,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "model_weights": model_weights,
            "feature_names": feature_names,
            "metadata": metadata,
            "calibrator": calibrator,
        },
        output_path,
    )

    loaded = Poker44Model(output_path)
    latency = loaded.benchmark_latency([example["chunk"] for example in test_examples[:4]])
    metrics["latency_per_chunk_ms"] = latency["latency_per_chunk_ms"]
    return models, feature_names, metrics, metadata


def main() -> None:
    args = parse_args()
    _, feature_names, metrics, metadata = train_model(args)
    print(f"Saved model to {args.output}")
    print(f"Feature count: {len(feature_names)}")
    print(
        "Selected config: "
        f"framework={metadata.get('framework')} "
        f"split_strategy={metadata.get('split_strategy')} "
        f"holdout_dates={metadata.get('holdout_source_dates')} "
        f"benchmark_rows={metadata.get('benchmark_rows')} "
        f"aux_human_rows={metadata.get('aux_human_rows')} "
        f"aux_human_calibration_rows={metadata.get('aux_human_calibration_rows')}"
    )
    for key in (
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "mcc_at_0_5",
        "validator_reward",
        "validator_fpr",
        "validator_bot_recall",
        "validator_ap_score",
        "validator_human_safety_penalty",
        "validator_base_score",
        "recall_at_0_5",
        "precision_at_0_5",
        "fpr_at_0_5",
        "tp",
        "fp",
        "tn",
        "fn",
        "prob_min",
        "prob_max",
        "human_prob_max",
        "bot_prob_min",
        "human_clearance_to_0_5",
        "bot_clearance_to_0_5",
        "score_gap_at_0_5",
        "threshold_margin_at_0_5",
        "prob_mean",
        "raw_human_prob_max",
        "raw_bot_prob_min",
        "raw_score_gap_at_0_5",
        "latency_per_chunk_ms",
    ):
        if key in metrics:
            print(f"{key}={float(metrics[key]):.6f}")


if __name__ == "__main__":
    main()
