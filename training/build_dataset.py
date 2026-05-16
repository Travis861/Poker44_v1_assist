from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any

from poker44_ml.features import chunk_features


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = (
    REPO_ROOT / "hands_generator" / "evaluation_datas" / "training_benchmark.txt"
)
DEFAULT_HUMAN_PATH = (
    REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
)


def load_json_or_gz(path: str | Path) -> Any:
    file_path = Path(path)
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_benchmark_paths(path: str | Path | None) -> list[Path]:
    if path:
        candidate = Path(path)
        if candidate.is_dir():
            paths = sorted(candidate.glob("training_benchmark*.txt"))
        else:
            paths = [candidate]
    else:
        paths = [DEFAULT_BENCHMARK_PATH]
    existing = [candidate for candidate in paths if candidate.exists()]
    if not existing:
        raise FileNotFoundError(f"No benchmark files found for {path or DEFAULT_BENCHMARK_PATH}")
    return existing


def resolve_human_path(path: str | Path | None) -> Path:
    candidate = Path(path) if path else DEFAULT_HUMAN_PATH
    if not candidate.exists():
        raise FileNotFoundError(f"Human baseline file not found: {candidate}")
    return candidate


def _as_root(payload: Any) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload


def _feature_row(chunk: list[dict[str, Any]]) -> dict[str, float]:
    row = chunk_features(chunk)
    row["hand_count"] = float(len(chunk))
    return row


def _iter_release_groups(payload: Any) -> list[dict[str, Any]]:
    root = _as_root(payload)
    if isinstance(root, dict) and isinstance(root.get("chunks"), list):
        return [group for group in root["chunks"] if isinstance(group, dict)]
    return []


def _load_labeled_benchmark_file(path: Path) -> list[dict[str, Any]]:
    payload = load_json_or_gz(path)
    root = _as_root(payload)
    groups = _iter_release_groups(payload)
    if not groups:
        raise RuntimeError(f"Benchmark file has no labeled chunk groups: {path}")

    examples: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        chunks = group.get("chunks") or []
        labels = group.get("groundTruth") or group.get("groundTruthLabels") or []
        if len(chunks) != len(labels):
            raise RuntimeError(
                f"Benchmark group {group_index} has {len(chunks)} chunks but "
                f"{len(labels)} labels in {path}"
            )
        source_date = str(group.get("sourceDate") or root.get("sourceDate") or "")
        group_id = str(group.get("chunkId") or f"group_{group_index}")
        group_hash = str(group.get("chunkHash") or "")
        for item_index, (chunk, label) in enumerate(zip(chunks, labels)):
            if not isinstance(chunk, list):
                continue
            hand_chunk = [hand for hand in chunk if isinstance(hand, dict)]
            if not hand_chunk:
                continue
            examples.append(
                {
                    "chunk": hand_chunk,
                    "label": int(label),
                    "source_date": source_date,
                    "group_id": group_id,
                    "group_hash": group_hash,
                    "item_index": item_index,
                    "source_path": str(path),
                    "features": _feature_row(hand_chunk),
                }
            )
    if not examples:
        raise RuntimeError(f"No usable labeled chunks found in {path}")
    return examples


def load_benchmark_examples(paths: str | Path | list[str | Path]) -> list[dict[str, Any]]:
    path_list = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in path_list:
        for example in _load_labeled_benchmark_file(path):
            key = "|".join(
                [
                    str(example.get("source_path", "")),
                    str(example.get("group_hash", "")),
                    str(example.get("group_id", "")),
                    str(example.get("item_index", "")),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
    if not examples:
        raise RuntimeError("No benchmark examples loaded.")
    return examples


def load_human_hands(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json_or_gz(path)
    root = _as_root(payload)
    if isinstance(root, list):
        return [hand for hand in root if isinstance(hand, dict)]

    hands: list[dict[str, Any]] = []
    for group in _iter_release_groups(payload):
        for chunk in group.get("chunks") or []:
            if not isinstance(chunk, list):
                continue
            hands.extend(hand for hand in chunk if isinstance(hand, dict))
    if not hands:
        raise RuntimeError(f"No human hands found in {path}")
    return hands


def build_human_chunk_examples(
    human_hands: list[dict[str, Any]],
    *,
    chunk_sizes: list[int],
    count: int,
    min_chunk_size: int = 20,
    seed: int = 42,
    source_path: str = "",
) -> list[dict[str, Any]]:
    if not human_hands or count <= 0:
        return []
    sizes = [max(min_chunk_size, int(size)) for size in chunk_sizes if int(size) > 0]
    if not sizes:
        sizes = [80]

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    max_start = max(0, len(human_hands) - min(sizes))
    for item_index in range(count):
        size = rng.choice(sizes)
        if len(human_hands) <= size:
            chunk = list(human_hands)
        else:
            start = rng.randint(0, max(0, min(max_start, len(human_hands) - size)))
            chunk = list(human_hands[start : start + size])
        if len(chunk) < min_chunk_size:
            continue
        examples.append(
            {
                "chunk": chunk,
                "label": 0,
                "source_date": "human_baseline",
                "group_id": "human_baseline",
                "group_hash": "",
                "item_index": item_index,
                "source_path": source_path,
                "features": _feature_row(chunk),
            }
        )
    return examples
