import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from transformers import pipeline

from config import MODEL_CANDIDATES, MODEL_LABEL_HINTS
from postprocessing import finalize_entities


RAW_LABEL_TO_CATEGORY = {
    "disease_disorder": "Disease",
    "disease": "Disease",
    "disorder": "Disease",
    "sign_symptom": "Symptom",
    "symptom": "Symptom",
    "symptoms": "Symptom",
    "medication": "Medication",
    "drug": "Medication",
    "chemical": "Medication",
    "procedure": "Procedure",
    "diagnostic_procedure": "Procedure",
    "therapeutic_procedure": "Procedure",
    "lab_value": "Procedure",
}

DEFAULT_BATCH_SIZE = 8
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120


def _resolve_hf_token() -> Optional[str]:
    for env_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        token = os.getenv(env_name, "").strip()
        if token:
            return token
    return None


def _resolve_device() -> int:
    return 0 if torch.cuda.is_available() else -1


def _map_raw_label_to_category(raw_label: str) -> Optional[str]:
    label = (raw_label or "").strip().lower().replace("-", "_")
    if label in RAW_LABEL_TO_CATEGORY:
        return RAW_LABEL_TO_CATEGORY[label]
    for hint, category in MODEL_LABEL_HINTS.items():
        if hint in label:
            return category
    return None


@lru_cache(maxsize=1)
def get_ner_pipeline() -> Tuple[Optional[object], Dict[str, object]]:
    failures = []
    hf_token = _resolve_hf_token()
    device = _resolve_device()
    for candidate in MODEL_CANDIDATES:
        model_name = candidate["name"]
        role = candidate["role"]
        try:
            ner_pipe = pipeline(
                "token-classification",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=device,
                token=hf_token,
            )
            return ner_pipe, {
                "model_name": model_name,
                "role": role,
                "failures": failures,
                "available": True,
                "auth_enabled": bool(hf_token),
                "device": "cuda" if device == 0 else "cpu",
            }
        except Exception as exc:
            failures.append({"model": model_name, "reason": str(exc)})
    return None, {
        "model_name": "unavailable",
        "role": "dictionary_only_fallback",
        "failures": failures,
        "available": False,
        "auth_enabled": bool(hf_token),
        "device": "cuda" if device == 0 else "cpu",
    }


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[int, str]]:
    if len(text) <= chunk_size:
        return [(0, text)]

    chunks: List[Tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            soft_break = text.rfind(" ", start + max(chunk_size - 200, 1), end)
            if soft_break > start:
                end = soft_break
        chunks.append((start, text[start:end]))
        if end >= len(text):
            break
        start = max(end - overlap, 0)
    return chunks


def _extract_from_model(text: str) -> List[Dict[str, object]]:
    ner_pipe, _ = get_ner_pipeline()
    if ner_pipe is None:
        return []

    model_entities: List[Dict[str, object]] = []
    for offset, chunk in _chunk_text(text):
        raw_entities = ner_pipe(chunk)
        for item in raw_entities:
            raw_label = item.get("entity_group", "") or item.get("entity", "")
            category = _map_raw_label_to_category(raw_label)
            if not category:
                continue

            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            if start < 0 or end <= start:
                continue

            global_start = offset + start
            global_end = offset + end
            model_entities.append(
                {
                    "text": text[global_start:global_end],
                    "label": category,
                    "start": global_start,
                    "end": global_end,
                    "confidence": float(item.get("score", 0.0)),
                    "source": "model",
                }
            )
    return model_entities


def extract_entities_batch(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, object]:
    batch_size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    ner_pipe, model_meta = get_ner_pipeline()
    if ner_pipe is None:
        return {"entities": [finalize_entities(text, []) for text in texts], "model_meta": model_meta}

    chunk_jobs = []
    for text_index, text in enumerate(texts):
        for offset, chunk in _chunk_text(text):
            chunk_jobs.append({"text_index": text_index, "offset": offset, "chunk": chunk})

    model_entities_by_text: List[List[Dict[str, object]]] = [[] for _ in texts]
    for batch_start in range(0, len(chunk_jobs), batch_size):
        batch_jobs = chunk_jobs[batch_start : batch_start + batch_size]
        batch_outputs = ner_pipe([job["chunk"] for job in batch_jobs], batch_size=batch_size)
        for job, raw_entities in zip(batch_jobs, batch_outputs):
            text = texts[job["text_index"]]
            for item in raw_entities:
                raw_label = item.get("entity_group", "") or item.get("entity", "")
                category = _map_raw_label_to_category(raw_label)
                if not category:
                    continue

                start = int(item.get("start", -1))
                end = int(item.get("end", -1))
                if start < 0 or end <= start:
                    continue

                global_start = job["offset"] + start
                global_end = job["offset"] + end
                model_entities_by_text[job["text_index"]].append(
                    {
                        "text": text[global_start:global_end],
                        "label": category,
                        "start": global_start,
                        "end": global_end,
                        "confidence": float(item.get("score", 0.0)),
                        "source": "model",
                    }
                )

    results = [finalize_entities(text, model_entities_by_text[index]) for index, text in enumerate(texts)]
    return {"entities": results, "model_meta": model_meta}


def extract_entities(text: str) -> Dict[str, object]:
    batch_result = extract_entities_batch([text], batch_size=1)
    return {"entities": batch_result["entities"][0], "model_meta": batch_result["model_meta"]}
