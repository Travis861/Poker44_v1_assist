"""Challenge-aligned Poker44 miner with deterministic chunk heuristics."""

import logging as stdlogging
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

try:
    from poker44_ml.inference import Poker44Model
except ImportError:  # pragma: no cover - optional local-model path.
    Poker44Model = None


class _ScannerNoiseFilter(stdlogging.Filter):
    """Suppress common public-port probe errors emitted before miner routing."""

    _NOISY_SNIPPETS = (
        "UnknownSynapseError",
        "InvalidRequestNameError",
    )
    _NOISY_REQUEST_NAMES = (
        "Synapse name ''",
        "Synapse name 'api'",
        "Synapse name 'mcp'",
        "Synapse name 'jsonrpc'",
        "Synapse name 'robots.txt'",
        "Could not parser request .",
    )

    def filter(self, record: stdlogging.LogRecord) -> bool:
        message = record.getMessage()
        if not any(snippet in message for snippet in self._NOISY_SNIPPETS):
            return True
        if any(name in message for name in self._NOISY_REQUEST_NAMES):
            return False
        return True


class Miner(BaseMinerNeuron):
    """
    Reference miner for the current provider-runtime challenge path.

    This miner scores chunks directly from the incoming hand payloads without
    any local training artifacts. The heuristic emphasizes chunk-level behavior
    consistency, passive regularity, street progression, and showdown tendency.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        self._install_scanner_log_filter()
        self.max_hands_per_chunk_eval = max(
            0, int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK_EVAL", "120"))
        )
        self.query_log_preview = (
            os.getenv("POKER44_LOG_QUERY_PREVIEW", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.component_debug_logging = (
            os.getenv("POKER44_LOG_SCORE_COMPONENTS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.score_array_logging = (
            os.getenv("POKER44_LOG_SCORE_ARRAYS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.batch_score_normalization = (
            os.getenv("POKER44_BATCH_SCORE_NORMALIZATION", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.batch_normalization_min_chunks = max(
            4, int(os.getenv("POKER44_BATCH_NORMALIZATION_MIN_CHUNKS", "12"))
        )
        self.batch_normalization_min_spread = max(
            0.0, float(os.getenv("POKER44_BATCH_NORMALIZATION_MIN_SPREAD", "0.18"))
        )
        self.model_path = Path(
            os.getenv(
                "POKER44_MODEL_PATH",
                str(repo_root / "models" / "poker44_benchmark_supervised.joblib"),
            )
        )
        self.predictor = None
        self.backend = "heuristic"
        if Poker44Model is not None and self.model_path.exists():
            try:
                self.predictor = Poker44Model(self.model_path)
                self.backend = "benchmark-supervised"
            except Exception as err:
                bt.logging.warning(
                    f"Failed to load local benchmark model at {self.model_path}: {err}. "
                    "Continuing with heuristic backend."
                )

        bt.logging.info(f"🤖 Poker44 Miner started with backend={self.backend}")
        runtime_commit = self._repo_head(repo_root)
        runtime_repo_url = self._repo_url(repo_root)
        published_repo_commit = (
            str(self.predictor.metadata.get("repo_commit", "")).strip()
            if self.predictor is not None
            else runtime_commit
        ) or runtime_commit
        published_repo_url = (
            str(self.predictor.metadata.get("repo_url", "")).strip()
            if self.predictor is not None
            else runtime_repo_url
        ) or runtime_repo_url
        model_metadata = dict(self.predictor.metadata) if self.predictor is not None else {}
        aux_human_rows = int(float(model_metadata.get("aux_human_rows", 0.0) or 0.0))
        aux_human_calibration_rows = int(
            float(model_metadata.get("aux_human_calibration_rows", 0.0) or 0.0)
        )
        benchmark_rows = int(float(model_metadata.get("benchmark_rows", 0.0) or 0.0))
        ensemble_combiner = str(model_metadata.get("ensemble_combiner", "") or "").strip()
        ensemble_max_blend = model_metadata.get("ensemble_max_blend")
        feature_distance_calibrator = model_metadata.get("feature_distance_calibrator") or {}
        score_expansion = model_metadata.get("score_expansion") or {}
        score_remap = model_metadata.get("score_remap") or {}
        score_logit_bias = model_metadata.get("score_logit_bias")
        score_logit_temperature = model_metadata.get("score_logit_temperature")
        trained_with_aux_humans = aux_human_rows > 0 or aux_human_calibration_rows > 0
        supervised_notes = (
            "Supervised benchmark model trained on released evaluation chunks"
        )
        if trained_with_aux_humans:
            supervised_notes += (
                " plus a human-only baseline corpus for human-safety calibration"
            )
        if ensemble_combiner:
            supervised_notes += f"; ensemble_combiner={ensemble_combiner}"
            if ensemble_max_blend is not None:
                supervised_notes += f", ensemble_max_blend={ensemble_max_blend}"
        if feature_distance_calibrator:
            supervised_notes += (
                "; feature_distance_calibrator="
                f"{feature_distance_calibrator.get('kind', 'enabled')}"
                f" blend={feature_distance_calibrator.get('blend', 'unknown')}"
            )
        if score_expansion:
            supervised_notes += f"; score_expansion={score_expansion.get('kind', 'enabled')}"
        if score_remap:
            supervised_notes += (
                f"; score_remap={score_remap.get('kind', 'enabled')} "
                f"threshold={score_remap.get('threshold', 'unknown')}"
            )
        if score_logit_bias is not None:
            supervised_notes += (
                f"; score_logit_bias={score_logit_bias}"
                f", score_logit_temperature={score_logit_temperature or 1.0}"
            )
        training_data_statement = (
            f"Trained on {benchmark_rows or 'released'} benchmark chunks with groundTruth labels."
            if self.predictor is not None
            else "Reference heuristic miner. No training step. Uses only runtime chunk features."
        )
        if self.predictor is not None and trained_with_aux_humans:
            training_data_statement += (
                f" Added {aux_human_rows} auxiliary human-only training chunks"
                f" and {aux_human_calibration_rows} auxiliary human-only calibration chunks"
                " from hands_generator/human_hands/poker_hands_combined.json.gz."
            )
        training_data_sources = (
            ["released_training_benchmark", "human_only_baseline_corpus"]
            if self.predictor is not None and trained_with_aux_humans
            else (["released_training_benchmark"] if self.predictor is not None else ["none"])
        )
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                repo_root / "poker44_ml" / "inference.py",
                repo_root / "poker44_ml" / "features.py",
            ],
            defaults={
                "model_name": (
                    "poker44_benchmark_supervised_human_baseline"
                    if trained_with_aux_humans
                    else "poker44_benchmark_supervised"
                    if self.predictor is not None
                    else "poker44-reference-heuristic"
                ),
                "model_version": (
                    "human-baseline-logit-bias-v2"
                    if trained_with_aux_humans
                    else ("1" if self.predictor is not None else "2")
                ),
                "framework": (
                    self.predictor.metadata.get("framework", "benchmark-supervised")
                    if self.predictor is not None
                    else "python-heuristic"
                ),
                "repo_commit": published_repo_commit,
                "repo_url": published_repo_url,
                "notes": (
                    supervised_notes
                    if self.predictor is not None
                    else "Challenge-aligned heuristic miner that scores chunk-level "
                    "behavioral regularity and action patterns."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": training_data_statement,
                "training_data_sources": training_data_sources,
                "private_data_attestation": (
                    "No validator-private data used. Supervised artifacts use "
                    "released benchmark labels"
                    " and, when present, local human-only baseline hands."
                ),
            },
        )
        override_repo_url = os.getenv("POKER44_MODEL_REPO_URL", "").strip()
        override_repo_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
        if override_repo_url and override_repo_url != published_repo_url:
            bt.logging.warning(
                "Ignoring POKER44_MODEL_REPO_URL override because it does not match "
                f"the active model/runtime identity | override={override_repo_url} "
                f"published={published_repo_url}"
            )
        if override_repo_commit and override_repo_commit != published_repo_commit:
            bt.logging.warning(
                "Ignoring POKER44_MODEL_REPO_COMMIT override because it does not match "
                f"the active model/runtime identity | override={override_repo_commit} "
                f"published={published_repo_commit}"
            )
        self.model_manifest["repo_url"] = published_repo_url
        self.model_manifest["repo_commit"] = published_repo_commit
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Axon created: {self.axon}")

    @staticmethod
    def _repo_head(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _repo_url(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _install_scanner_log_filter() -> None:
        enabled = os.getenv("POKER44_SUPPRESS_SCANNER_ERRORS", "1").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return

        scanner_filter = _ScannerNoiseFilter()
        configured = False

        for handler in getattr(bt.logging, "_handlers", []):
            handler.addFilter(scanner_filter)
            configured = True

        for logger_name in ("bittensor", "uvicorn.access", "uvicorn.error"):
            logger = stdlogging.getLogger(logger_name)
            for handler in logger.handlers:
                handler.addFilter(scanner_filter)
                configured = True

        if configured:
            bt.logging.info("Scanner-noise log filter enabled for invalid public-port probes.")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            f"Model path={self.model_path} backend={self.backend}"
        )
        bt.logging.info(
            "Runtime config | "
            f"max_hands_per_chunk_eval={self.max_hands_per_chunk_eval} "
            f"query_log_preview={self.query_log_preview} "
            f"component_debug_logging={self.component_debug_logging} "
            f"score_array_logging={self.score_array_logging} "
            f"batch_score_normalization={self.batch_score_normalization} "
            f"batch_normalization_min_chunks={self.batch_normalization_min_chunks} "
            f"batch_normalization_min_spread={self.batch_normalization_min_spread}"
        )
        if self.predictor is not None:
            artifact_commit = str(self.predictor.metadata.get("repo_commit", ""))
            runtime_commit = self._repo_head(repo_root)
            bt.logging.info(
                f"Model metadata: feature_count={len(self.predictor.feature_names)} "
                f"framework={self.predictor.metadata.get('framework', 'unknown')} "
                f"artifact_commit={artifact_commit or 'unknown'} "
                f"runtime_commit={runtime_commit or 'unknown'} "
                f"feature_schema_hash={self.predictor.metadata.get('feature_schema_hash', 'unknown')}"
            )
            if artifact_commit and runtime_commit and artifact_commit != runtime_commit:
                bt.logging.warning(
                    "Model artifact commit does not match current checkout | "
                    f"artifact_commit={artifact_commit} runtime_commit={runtime_commit}"
                )
        whitelist = sorted(self.validator_hotkey_whitelist)
        bt.logging.info(
            "Access policy | "
            f"force_validator_permit={self.config.blacklist.force_validator_permit} "
            f"allow_non_registered={self.config.blacklist.allow_non_registered} "
            f"validator_allowlist_count={len(whitelist)}"
        )
        if whitelist:
            bt.logging.info(f"Validator allowlist={whitelist}")
        bt.logging.info(
            "Miner docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    @staticmethod
    def _caller_hotkey(synapse: DetectionSynapse) -> str:
        return getattr(getattr(synapse, "dendrite", None), "hotkey", "unknown")

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _compress_chunk(self, chunk: list[dict]) -> list[dict]:
        limit = self.max_hands_per_chunk_eval
        if limit <= 0 or len(chunk) <= limit:
            return chunk
        if limit == 1:
            return [chunk[len(chunk) // 2]]

        last_index = len(chunk) - 1
        slots = limit - 1
        indices = {
            min(last_index, round(index * last_index / slots))
            for index in range(limit)
        }
        return [chunk[index] for index in sorted(indices)]

    def _normalize_batch_scores(self, scores: list[float]) -> tuple[list[float], bool]:
        if not self.batch_score_normalization:
            return scores, False
        if len(scores) < self.batch_normalization_min_chunks:
            return scores, False

        cleaned = [self._clamp01(score) for score in scores]
        score_min = min(cleaned)
        score_max = max(cleaned)
        score_spread = score_max - score_min
        crosses_threshold = score_min < 0.5 < score_max
        has_enough_spread = score_spread >= self.batch_normalization_min_spread
        if crosses_threshold and has_enough_spread:
            return cleaned, False

        if score_spread <= 1e-9:
            return cleaned, False

        normalized = [
            round(0.05 + 0.90 * ((score - score_min) / score_spread), 6)
            for score in cleaned
        ]
        return normalized, True

    @classmethod
    def _score_hand(cls, hand: dict) -> tuple[float, dict[str, float]]:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter((action.get("action_type") or "").lower() for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )
        aggressive_actions = action_counts.get("bet", 0) + action_counts.get("raise", 0)
        passive_actions = action_counts.get("call", 0) + action_counts.get("check", 0)

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        bet_ratio = action_counts.get("bet", 0) / meaningful_actions
        aggression_ratio = aggressive_actions / max(aggressive_actions + passive_actions, 1)
        street_depth = len(streets) / 4.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0
        player_count_signal = (6 - min(len(players), 6)) / 4.0 if players else 0.0
        action_diversity = len(
            [kind for kind in ("call", "check", "bet", "raise", "fold") if action_counts.get(kind, 0)]
        ) / 5.0

        score = 0.0
        score += 0.24 * cls._clamp01(street_depth)
        score += 0.16 * cls._clamp01(showdown_flag)
        score += 0.18 * cls._clamp01(call_ratio / 0.32)
        score += 0.10 * cls._clamp01(check_ratio / 0.28)
        score += 0.08 * cls._clamp01(player_count_signal)
        score += 0.10 * cls._clamp01(action_diversity / 0.60)
        score -= 0.14 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.12 * cls._clamp01(raise_ratio / 0.22)
        score -= 0.06 * cls._clamp01(bet_ratio / 0.18)
        score -= 0.08 * cls._clamp01(aggression_ratio / 0.55)

        features = {
            "call_ratio": call_ratio,
            "check_ratio": check_ratio,
            "fold_ratio": fold_ratio,
            "raise_ratio": raise_ratio,
            "bet_ratio": bet_ratio,
            "aggression_ratio": aggression_ratio,
            "street_depth": street_depth,
            "showdown_flag": showdown_flag,
        }
        return cls._clamp01(score), features

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores: list[float] = []
        call_ratios: list[float] = []
        aggression_ratios: list[float] = []
        street_depths: list[float] = []
        showdown_flags: list[float] = []

        for hand in chunk:
            hand_score, features = cls._score_hand(hand)
            hand_scores.append(hand_score)
            call_ratios.append(features["call_ratio"])
            aggression_ratios.append(features["aggression_ratio"])
            street_depths.append(features["street_depth"])
            showdown_flags.append(features["showdown_flag"])

        avg_score = sum(hand_scores) / len(hand_scores)
        consistency_bonus = 0.0
        if len(hand_scores) > 1:
            call_spread = max(call_ratios) - min(call_ratios)
            aggression_spread = max(aggression_ratios) - min(aggression_ratios)
            street_spread = max(street_depths) - min(street_depths)
            showdown_rate = sum(showdown_flags) / len(showdown_flags)

            consistency_bonus += 0.10 * cls._clamp01(1.0 - call_spread / 0.60)
            consistency_bonus += 0.08 * cls._clamp01(1.0 - aggression_spread / 0.70)
            consistency_bonus += 0.05 * cls._clamp01(1.0 - street_spread)
            consistency_bonus += 0.05 * cls._clamp01(showdown_rate / 0.60)

        return round(cls._clamp01(avg_score + consistency_bonus), 6)

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        caller = self._caller_hotkey(synapse)
        chunks = [self._compress_chunk(list(chunk or [])) for chunk in (synapse.chunks or [])]
        chunk_sizes = [len(chunk) for chunk in chunks]
        bt.logging.info(
            "Validator query received | "
            f"caller={caller} "
            f"incoming_chunk_count={len(chunks)} "
            f"chunk_size_range={ [min(chunk_sizes), max(chunk_sizes)] if chunk_sizes else [0, 0] }"
        )
        if self.query_log_preview:
            bt.logging.info(
                "Validator query preview | "
                f"caller={caller} "
                f"first_chunk_hand_count={chunk_sizes[0] if chunk_sizes else 0}"
            )

        started = time.perf_counter()
        backend_used = self.backend
        component_debug = {}
        if self.predictor is not None:
            try:
                scores = self.predictor.predict_chunk_scores(chunks)
                if self.component_debug_logging and hasattr(
                    self.predictor,
                    "debug_score_components",
                ):
                    component_debug = self.predictor.debug_score_components(chunks)
            except Exception as err:
                bt.logging.warning(
                    f"Predictor failure during chunk scoring: {err}. "
                    "Falling back to heuristic backend."
                )
                backend_used = "heuristic-fallback"
                scores = [self.score_chunk(chunk) for chunk in chunks]
        else:
            scores = [self.score_chunk(chunk) for chunk in chunks]
        raw_score_range = [min(scores), max(scores)] if scores else [0.0, 0.0]
        scores, normalized_scores = self._normalize_batch_scores(scores)
        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_hands = sum(chunk_sizes)
        per_chunk_ms = elapsed_ms / max(len(chunks), 1)
        per_hand_ms = elapsed_ms / max(total_hands, 1)
        message = (
            f"Scored {len(chunks)} chunks with backend={backend_used} "
            f"elapsed_ms={elapsed_ms:.2f} "
            f"per_chunk_ms={per_chunk_ms:.2f} "
            f"per_hand_ms={per_hand_ms:.2f} "
            f"chunk_size_range={ [min(chunk_sizes), max(chunk_sizes)] if chunk_sizes else [0, 0] } "
            f"raw_score_range={raw_score_range} "
            f"score_range={ [min(scores), max(scores)] if scores else [0.0, 0.0] }"
            f" batch_normalized={normalized_scores}"
        )
        if self.query_log_preview:
            message += (
                f" score_preview={scores[:5]} "
                f"prediction_preview={synapse.predictions[:5]}"
            )
        if component_debug:
            for name, values in component_debug.items():
                if values:
                    message += f" {name}_range={[min(values), max(values)]}"
        bt.logging.info(message)
        if self.score_array_logging:
            score_payload = {
                "chunk_sizes": chunk_sizes,
                "risk_scores": [round(float(score), 6) for score in scores],
                "predictions": [bool(prediction) for prediction in synapse.predictions],
            }
            if component_debug:
                score_payload["components"] = {
                    name: [round(float(value), 6) for value in values]
                    for name, values in component_debug.items()
                }
            bt.logging.info(f"Detailed chunk scores | {score_payload}")
        bt.logging.success(
            "Validator response sent successfully | "
            f"caller={caller} "
            f"incoming_chunk_count={len(chunks)} "
            f"risk_scores_length={len(scores)} "
            f"elapsed_ms={elapsed_ms:.2f} "
            f"per_chunk_ms={per_chunk_ms:.2f} "
            f"per_hand_ms={per_hand_ms:.2f}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        blocked, reason = self.common_blacklist(synapse)
        caller = self._caller_hotkey(synapse)
        if blocked:
            bt.logging.warning(
                f"Blocked miner request | caller={caller} reason={reason}"
            )
        else:
            bt.logging.info(
                f"Accepted miner request | caller={caller} reason={reason}"
            )
        return blocked, reason

    async def priority(self, synapse: DetectionSynapse) -> float:
        caller = self._caller_hotkey(synapse)
        priority = self.caller_priority(synapse)
        bt.logging.debug(
            f"Assigned caller priority | caller={caller} priority={priority}"
        )
        return priority


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
