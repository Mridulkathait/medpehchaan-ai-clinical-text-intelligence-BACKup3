import re
from typing import Dict, List, Optional, Tuple

from config import (
    CANONICAL_ENTITY_LABELS,
    ENTITY_LABELS,
    FALLBACK_CONFIDENCE_BASE,
    FALLBACK_CONFIDENCE_CANONICAL_BONUS,
    FALLBACK_CONFIDENCE_EXACT_MATCH_BONUS,
    FALLBACK_CONFIDENCE_MAX,
    FALLBACK_CONFIDENCE_MULTIWORD_BONUS,
    HIGH_CONFIDENCE_THRESHOLD,
    LABEL_PRIORITY,
    MAX_ENTITY_WORDS,
    MEDIUM_CONFIDENCE_THRESHOLD,
    MIN_ENTITY_CONFIDENCE,
    SUPPORT_DICTIONARY,
)

EDGE_STOPWORDS = {
    "the",
    "a",
    "an",
    "patient",
    "history",
    "reports",
    "reporting",
    "noted",
    "notes",
    "with",
    "of",
    "and",
    "or",
    "has",
    "have",
    "had",
    "is",
    "are",
}


def normalize_entity_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip(" ,;:.!?()[]{}"))
    words = text.split()
    while words and words[0].lower() in EDGE_STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in EDGE_STOPWORDS:
        words.pop()
    return " ".join(words)


def normalize_confidence(score: float) -> float:
    score = max(0.0, min(1.0, float(score)))
    return round(score * 100.0, 2)


def confidence_band(confidence_pct: float) -> str:
    if confidence_pct > HIGH_CONFIDENCE_THRESHOLD * 100:
        return "High confidence"
    if confidence_pct >= MEDIUM_CONFIDENCE_THRESHOLD * 100:
        return "Medium confidence"
    return "Low confidence"


def _lookup_canonical_label(text: str) -> Optional[str]:
    return CANONICAL_ENTITY_LABELS.get(text.lower())


def _fallback_confidence(term: str, matched_text: str) -> float:
    confidence = FALLBACK_CONFIDENCE_BASE
    if matched_text.strip().lower() == term.strip().lower():
        confidence += FALLBACK_CONFIDENCE_EXACT_MATCH_BONUS
    if _lookup_canonical_label(matched_text):
        confidence += FALLBACK_CONFIDENCE_CANONICAL_BONUS
    if len(term.split()) > 1:
        confidence += FALLBACK_CONFIDENCE_MULTIWORD_BONUS
    confidence = min(confidence, FALLBACK_CONFIDENCE_MAX)
    return round(confidence, 2)


def _quality_score(entity: Dict[str, object]) -> Tuple[float, int, int]:
    confidence = float(entity.get("confidence", 0.0))
    label_priority = LABEL_PRIORITY.get(str(entity.get("label", "")), 0)
    source_priority = 1 if entity.get("source") == "model" else 0
    return confidence, label_priority, source_priority


def _has_word_boundaries(text: str, start: int, end: int) -> bool:
    left_ok = start == 0 or not text[start - 1].isalnum()
    right_ok = end >= len(text) or not text[end].isalnum()
    return left_ok and right_ok


def _is_valid_span(text: str, entity_text: str, start: int, end: int) -> bool:
    if start < 0 or end <= start or end > len(text):
        return False
    if not _has_word_boundaries(text, start, end):
        return False
    return normalize_entity_text(text[start:end]).lower() == normalize_entity_text(entity_text).lower()


def _allow_entity(entity_text: str) -> bool:
    normalized = normalize_entity_text(entity_text)
    if not normalized:
        return False
    if len(normalized.split()) > MAX_ENTITY_WORDS:
        return False
    if len(normalized) < 3:
        return False
    if not re.search(r"[A-Za-z]", normalized):
        return False
    return True


def _prefer_more_specific(primary: Dict[str, object], contender: Dict[str, object]) -> Dict[str, object]:
    primary_words = len(str(primary["text"]).split())
    contender_words = len(str(contender["text"]).split())
    if contender_words > primary_words and abs(float(primary["confidence"]) - float(contender["confidence"])) <= 8:
        return contender
    return primary


def _resolve_overlap(existing: Dict[str, object], candidate: Dict[str, object]) -> Optional[Dict[str, object]]:
    existing_key = str(existing["text"]).lower()
    candidate_key = str(candidate["text"]).lower()
    if existing_key == candidate_key:
        winner = max([existing, candidate], key=_quality_score)
        if _lookup_canonical_label(candidate_key):
            winner["label"] = _lookup_canonical_label(candidate_key)
        return winner

    existing_range = set(range(int(existing["start"]), int(existing["end"])))
    candidate_range = set(range(int(candidate["start"]), int(candidate["end"])))
    if not (existing_range & candidate_range):
        return None

    winner = max([existing, candidate], key=_quality_score)
    winner = _prefer_more_specific(winner, candidate if winner is existing else existing)
    canonical_label = _lookup_canonical_label(str(winner["text"]))
    if canonical_label:
        winner["label"] = canonical_label
    return winner


def apply_dictionary_fallback(text: str, entities: List[Dict[str, object]]) -> List[Dict[str, object]]:
    occupied = [(int(ent["start"]), int(ent["end"])) for ent in entities]
    output = list(entities)

    for label, terms in SUPPORT_DICTIONARY.items():
        for term in sorted(terms, key=lambda value: (-len(value), value)):
            for match in re.finditer(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                start, end = match.span()
                overlapping = [
                    entity for entity in output if not (end <= int(entity["start"]) or start >= int(entity["end"]))
                ]
                if overlapping:
                    best_overlap = max(overlapping, key=_quality_score)
                    if term.lower() == str(best_overlap["text"]).lower():
                        continue
                    overlap_text = str(best_overlap["text"]).lower()
                    if term.lower().find(overlap_text) != -1 and len(term.split()) >= len(overlap_text.split()):
                        output.remove(best_overlap)
                        occupied = [span for span in occupied if span != (int(best_overlap["start"]), int(best_overlap["end"]))]
                    else:
                        continue
                output.append(
                    {
                        "text": text[start:end],
                        "label": _lookup_canonical_label(text[start:end]) or label,
                        "start": start,
                        "end": end,
                        "source": "fallback_dictionary",
                    }
                )
                output[-1]["confidence"] = _fallback_confidence(term, output[-1]["text"])
                output[-1]["confidence_label"] = confidence_band(output[-1]["confidence"])
                occupied.append((start, end))
    return output


def finalize_entities(text: str, raw_entities: List[Dict[str, object]]) -> List[Dict[str, object]]:
    cleaned: List[Dict[str, object]] = []
    for entity in raw_entities:
        label = entity.get("label")
        if label not in ENTITY_LABELS:
            continue

        start = int(entity.get("start", -1))
        end = int(entity.get("end", -1))
        entity_text = normalize_entity_text(str(entity.get("text", "")))
        if not _allow_entity(entity_text):
            continue
        if not _is_valid_span(text, entity_text, start, end):
            continue

        confidence_pct = normalize_confidence(float(entity.get("confidence", 0.0)))
        if confidence_pct < MIN_ENTITY_CONFIDENCE * 100:
            continue

        cleaned.append(
            {
                "text": text[start:end],
                "label": _lookup_canonical_label(entity_text) or label,
                "start": start,
                "end": end,
                "confidence": confidence_pct,
                "confidence_label": confidence_band(confidence_pct),
                "source": entity.get("source", "model"),
            }
        )

    selected: List[Dict[str, object]] = []
    for candidate in sorted(cleaned, key=lambda item: (int(item["start"]), int(item["end"]))):
        replaced = False
        for index, existing in enumerate(selected):
            resolved = _resolve_overlap(existing, candidate)
            if resolved is None:
                continue
            selected[index] = resolved
            replaced = True
            break
        if not replaced:
            selected.append(candidate)

    deduped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for entity in selected:
        normalized_text = normalize_entity_text(str(entity["text"]))
        canonical_label = _lookup_canonical_label(normalized_text)
        if canonical_label:
            entity["label"] = canonical_label
        key = (normalized_text.lower(), entity["label"])
        current = deduped.get(key)
        if current is None or _quality_score(entity) > _quality_score(current):
            deduped[key] = entity

    final_entities = sorted(deduped.values(), key=lambda item: (int(item["start"]), int(item["end"])))
    final_entities = apply_dictionary_fallback(text, final_entities)
    final_entities.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return final_entities
