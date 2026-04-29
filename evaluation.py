import json
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

SAMPLE_EVALUATION_DATA = [
    {
        "text": "Patient with uncontrolled hypertension, nausea, and blurred vision. CT scan ordered. Lisinopril started.",
        "gold": {
            ("hypertension", "Disease"),
            ("nausea", "Symptom"),
            ("blurred vision", "Symptom"),
            ("CT scan", "Procedure"),
            ("Lisinopril", "Medication"),
        },
    },
    {
        "text": "History of COPD exacerbation with productive cough. Nebulizer treatment given and chest x-ray advised.",
        "gold": {
            ("COPD exacerbation", "Disease"),
            ("productive cough", "Symptom"),
            ("Nebulizer treatment", "Procedure"),
            ("chest x-ray", "Procedure"),
        },
    },
    {
        "text": "Type 2 diabetes follow-up. HbA1c elevated. Metformin continued. Foot exam performed.",
        "gold": {
            ("Type 2 diabetes", "Disease"),
            ("HbA1c", "Procedure"),
            ("Metformin", "Medication"),
            ("Foot exam", "Procedure"),
        },
    },
    {
        "text": "Migraine symptoms with headache and vomiting. MRI brain recommended.",
        "gold": {
            ("Migraine", "Disease"),
            ("headache", "Symptom"),
            ("vomiting", "Symptom"),
            ("MRI brain", "Procedure"),
        },
    },
    {
        "text": "Possible sepsis with fever, tachycardia, and shortness of breath. Blood cultures ordered and ceftriaxone started.",
        "gold": {
            ("sepsis", "Disease"),
            ("fever", "Symptom"),
            ("tachycardia", "Symptom"),
            ("shortness of breath", "Symptom"),
            ("Blood cultures", "Procedure"),
            ("ceftriaxone", "Medication"),
        },
    },
]

LABEL_FIELD_ALIASES = {
    "Disease": ["diseases", "disease", "disease_labels", "gold_diseases", "actual_diseases"],
    "Symptom": ["symptoms", "symptom", "symptom_labels", "gold_symptoms", "actual_symptoms"],
    "Medication": ["medications", "medication", "medication_labels", "gold_medications", "actual_medications"],
    "Procedure": ["procedures", "procedure", "procedure_labels", "gold_procedures", "actual_procedures"],
}


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_label(value: object) -> str:
    return str(value or "").strip()


def _normalize_entity_pair(text: object, label: object) -> Optional[Tuple[str, str]]:
    normalized_text = _normalize_text(text)
    normalized_label = _normalize_label(label)
    if not normalized_text or not normalized_label:
        return None
    return normalized_text, normalized_label


def _normalize_entities(entities: Iterable[Dict[str, object]]) -> Set[Tuple[str, str]]:
    normalized: Set[Tuple[str, str]] = set()
    for entity in entities or []:
        pair = _normalize_entity_pair(entity.get("text"), entity.get("label"))
        if pair is not None:
            normalized.add(pair)
    return normalized


def _normalize_gold(gold: Iterable[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    normalized: Set[Tuple[str, str]] = set()
    for text, label in gold or []:
        pair = _normalize_entity_pair(text, label)
        if pair is not None:
            normalized.add(pair)
    return normalized


def _parse_label_values(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and value != value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in raw.replace("|", ",").replace(";", ",").split(",") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def extract_gold_labels_from_row(row: Dict[str, object]) -> Set[Tuple[str, str]]:
    gold: Set[Tuple[str, str]] = set()
    for label, aliases in LABEL_FIELD_ALIASES.items():
        for alias in aliases:
            if alias not in row:
                continue
            for value in _parse_label_values(row.get(alias)):
                pair = _normalize_entity_pair(value, label)
                if pair is not None:
                    gold.add(pair)
            break
    return gold


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _to_percent(value: float) -> float:
    return round(value * 100.0, 2)


def _entity_accuracy(true_positive: int, false_positive: int, false_negative: int) -> float:
    return _safe_divide(true_positive, true_positive + false_positive + false_negative)


def evaluate_predictions(
    predicted: Iterable[Dict[str, object]] | Set[Tuple[str, str]],
    ground_truth: Iterable[Tuple[str, str]],
) -> Dict[str, object]:
    predicted_items = list(predicted or [])
    if predicted_items and isinstance(predicted_items[0], tuple):
        predicted_set = _normalize_gold(predicted_items)  # type: ignore[arg-type]
    else:
        predicted_set = _normalize_entities(predicted_items)  # type: ignore[arg-type]
    ground_truth_set = _normalize_gold(ground_truth)

    true_positives = predicted_set & ground_truth_set
    false_positives = predicted_set - ground_truth_set
    false_negatives = ground_truth_set - predicted_set

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1_score = _f1_score(precision, recall)
    accuracy = _entity_accuracy(tp, fp, fn)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": _to_percent(precision),
        "recall": _to_percent(recall),
        "f1_score": _to_percent(f1_score),
        "accuracy": _to_percent(accuracy),
        "matched_entities": sorted(true_positives),
        "unexpected_predictions": sorted(false_positives),
        "missed_ground_truth": sorted(false_negatives),
    }


def compute_metrics_from_pairs(pairs: List[Dict[str, object]]) -> Dict[str, object]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_sample: List[Dict[str, object]] = []
    macro_precisions: List[float] = []
    macro_recalls: List[float] = []
    macro_f1_scores: List[float] = []
    macro_accuracies: List[float] = []

    for pair in pairs:
        sample_metrics = evaluate_predictions(pair.get("predicted", []), pair.get("gold", []))
        total_tp += sample_metrics["true_positive"]
        total_fp += sample_metrics["false_positive"]
        total_fn += sample_metrics["false_negative"]
        macro_precisions.append(float(sample_metrics["precision"]))
        macro_recalls.append(float(sample_metrics["recall"]))
        macro_f1_scores.append(float(sample_metrics["f1_score"]))
        macro_accuracies.append(float(sample_metrics["accuracy"]))
        per_sample.append(
            {
                "patient_id": pair.get("patient_id"),
                **sample_metrics,
            }
        )

    micro_precision_ratio = _safe_divide(total_tp, total_tp + total_fp)
    micro_recall_ratio = _safe_divide(total_tp, total_tp + total_fn)
    micro_f1_ratio = _f1_score(micro_precision_ratio, micro_recall_ratio)
    micro_accuracy_ratio = _entity_accuracy(total_tp, total_fp, total_fn)
    exact_matches = sum(
        1
        for pair in pairs
        if _normalize_entities(pair.get("predicted", [])) == _normalize_gold(pair.get("gold", []))
    )

    macro_precision = round(sum(macro_precisions) / len(macro_precisions), 2) if macro_precisions else 0.0
    macro_recall = round(sum(macro_recalls) / len(macro_recalls), 2) if macro_recalls else 0.0
    macro_f1 = round(sum(macro_f1_scores) / len(macro_f1_scores), 2) if macro_f1_scores else 0.0
    macro_accuracy = round(sum(macro_accuracies) / len(macro_accuracies), 2) if macro_accuracies else 0.0

    return {
        "precision": _to_percent(micro_precision_ratio),
        "recall": _to_percent(micro_recall_ratio),
        "f1_score": _to_percent(micro_f1_ratio),
        "accuracy": _to_percent(micro_accuracy_ratio),
        "exact_match_rate": _to_percent(_safe_divide(exact_matches, len(pairs))),
        "micro_average": {
            "precision": _to_percent(micro_precision_ratio),
            "recall": _to_percent(micro_recall_ratio),
            "f1_score": _to_percent(micro_f1_ratio),
            "accuracy": _to_percent(micro_accuracy_ratio),
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
        },
        "macro_average": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "accuracy": macro_accuracy,
        },
        "support": len(pairs),
        "samples": per_sample,
    }


def compute_metrics(run_inference: bool = True) -> Dict[str, object]:
    pairs = []
    for index, sample in enumerate(SAMPLE_EVALUATION_DATA, start=1):
        if run_inference:
            from ner_engine import extract_entities

            predicted = extract_entities(sample["text"])["entities"]
        else:
            predicted = [{"text": text, "label": label} for text, label in sample["gold"]]
        pairs.append(
            {
                "patient_id": f"sample_{index:03d}",
                "predicted": predicted,
                "gold": sample["gold"],
            }
        )

    return {
        **compute_metrics_from_pairs(pairs),
        "dataset_type": "manual_sample_dataset",
        "source": "live_inference" if run_inference else "bundled_baseline",
    }


def compute_metrics_for_patient_results(patient_results: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    pairs = []
    for result in patient_results:
        gold = extract_gold_labels_from_row(result.get("metadata", {}))
        if not gold:
            continue
        pairs.append(
            {
                "patient_id": result["patient_id"],
                "predicted": result.get("entities", []),
                "gold": gold,
            }
        )

    if not pairs:
        return None

    return {
        **compute_metrics_from_pairs(pairs),
        "dataset_type": "uploaded_dataset_labels",
    }
