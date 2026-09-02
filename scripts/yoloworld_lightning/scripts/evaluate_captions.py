#!/usr/bin/env python3
"""Evaluate video captions with the standard COCO-caption protocol.

References are tokenized with PTBTokenizer and scored with the pycocoevalcap
BLEU, METEOR, ROUGE-L, and CIDEr implementations. Prediction/reference IDs
must match exactly; silently evaluating only an intersection is not allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


CaptionMap = Dict[str, List[str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _captions(value: object, *, context: str) -> List[str]:
    if isinstance(value, str):
        result = [value.strip()]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        result = [item.strip() for item in value]
    else:
        raise TypeError(f"{context} must be a string or list of strings")
    if not result or any(not item for item in result):
        raise ValueError(f"{context} contains an empty caption")
    return result


def load_references(path: Path) -> CaptionMap:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Reference JSON must be a list of records")

    references: CaptionMap = {}
    for index, record in enumerate(payload):
        if not isinstance(record, dict) or "video_id" not in record or "caption" not in record:
            raise ValueError(f"Invalid reference record at index {index}")
        video_id = str(record["video_id"])
        if video_id in references:
            raise ValueError(f"Duplicate reference video_id: {video_id}")
        references[video_id] = _captions(record["caption"], context=f"reference {video_id}")
    return references


def _records_to_predictions(records: Iterable[object]) -> CaptionMap:
    predictions: CaptionMap = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "video_id" not in record:
            raise ValueError(f"Invalid prediction record at index {index}")
        key = "prediction" if "prediction" in record else "caption"
        if key not in record:
            raise ValueError(f"Prediction record {index} has no prediction/caption field")
        video_id = str(record["video_id"])
        if video_id in predictions:
            raise ValueError(f"Duplicate prediction video_id: {video_id}")
        values = _captions(record[key], context=f"prediction {video_id}")
        if len(values) != 1:
            raise ValueError(f"Prediction {video_id} must contain exactly one caption")
        predictions[video_id] = values
    return predictions


def load_predictions(path: Path) -> CaptionMap:
    if path.is_dir():
        predictions: CaptionMap = {}
        for text_path in sorted(path.glob("*.txt")):
            video_id = text_path.stem
            values = _captions(
                text_path.read_text(encoding="utf-8").splitlines(),
                context=f"prediction {video_id}",
            )
            if len(values) != 1:
                raise ValueError(f"{text_path} must contain exactly one non-empty line")
            predictions[video_id] = values
        return predictions

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]
    if isinstance(payload, dict):
        predictions = {}
        for video_id, caption in payload.items():
            values = _captions(caption, context=f"prediction {video_id}")
            if len(values) != 1:
                raise ValueError(f"Prediction {video_id} must contain exactly one caption")
            predictions[str(video_id)] = values
        return predictions
    if isinstance(payload, list):
        return _records_to_predictions(payload)
    raise TypeError("Prediction JSON must be a mapping, record list, or results payload")


def evaluate(references: CaptionMap, predictions: CaptionMap) -> Dict[str, float]:
    reference_ids = set(references)
    prediction_ids = set(predictions)
    missing = sorted(reference_ids - prediction_ids)
    extra = sorted(prediction_ids - reference_ids)
    if missing or extra:
        raise ValueError(
            "Prediction/reference ID mismatch: "
            f"missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}"
        )

    ordered_ids = sorted(reference_ids)
    ground_truth = {
        index: [{"caption": text} for text in references[video_id]]
        for index, video_id in enumerate(ordered_ids)
    }
    results = {
        index: [{"caption": predictions[video_id][0]}]
        for index, video_id in enumerate(ordered_ids)
    }

    tokenizer = PTBTokenizer()
    ground_truth = tokenizer.tokenize(ground_truth)
    results = tokenizer.tokenize(results)

    metrics: Dict[str, float] = {}
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE-L"]),
        (Cider(), ["CIDEr"]),
    ]
    try:
        for scorer, names in scorers:
            score, _ = scorer.compute_score(ground_truth, results)
            values = score if isinstance(score, (list, tuple)) else [score]
            metrics.update({name: float(value) for name, value in zip(names, values)})
    finally:
        for scorer, _ in scorers:
            close = getattr(scorer, "close", None)
            if callable(close):
                close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    references = load_references(args.references)
    predictions = load_predictions(args.predictions)
    metrics = evaluate(references, predictions)
    payload = {
        "dataset": args.dataset,
        "protocol": "pycocoevalcap + PTBTokenizer + all references",
        "num_videos": len(references),
        "references": str(args.references.resolve()),
        "references_sha256": _sha256(args.references),
        "predictions": str(args.predictions.resolve()),
        "metrics_raw": metrics,
        "metrics_x100": {key: value * 100.0 for key, value in metrics.items()},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
