import re
from typing import Dict, List, Optional


PATIENT_SPLIT_PATTERN = re.compile(
    r"(?im)(?=^\s*(?:patient|pt|record|case|encounter|mrn)\s*(?:id|no|number)?\s*[:#-]?\s*\w+)"
)


def _remove_non_printable(text: str) -> str:
    return "".join(ch if ch.isprintable() or ch in {"\n", "\r", "\t"} else " " for ch in text)


def _normalize_punctuation(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([,;:.!?]){2,}", r"\1", text)
    text = re.sub(r"[ \t]+([,;:.!?])", r"\1", text)
    text = re.sub(r"([,;:.!?])([A-Za-z0-9])", r"\1 \2", text)
    return text


def _extract_patient_hint(text: str) -> Optional[str]:
    match = re.search(
        r"(?im)^\s*(?:patient|pt|record|case|encounter|mrn)\s*(?:id|no|number)?\s*[:#-]?\s*([A-Za-z0-9._-]+)",
        text,
    )
    return match.group(1).strip() if match else None


def normalize_patient_id(value: object, fallback_index: int) -> str:
    patient_hint = re.sub(r"\s+", "_", str(value or "").strip())
    if patient_hint:
        return patient_hint
    return f"patient_{fallback_index:04d}"


def preprocess_clinical_text(text: str) -> Dict[str, object]:
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    original = text.strip()
    printable = _remove_non_printable(original)
    punctuation_fixed = _normalize_punctuation(printable)
    cleaned = re.sub(r"[^\w\s,;:.!?%/\-()\n]+", " ", punctuation_fixed)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    lowered = cleaned.lower()
    tokens: List[str] = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/]*", cleaned)

    preview_text = cleaned.replace("\n", " ")
    return {
        "original_text": original,
        "clean_text": cleaned,
        "clean_text_lower": lowered,
        "tokens": tokens,
        "token_count": len(tokens),
        "char_count": len(cleaned),
        "preview": preview_text[:380] + ("..." if len(preview_text) > 380 else ""),
    }


def split_text_into_patient_records(text: str, source_name: str = "text_input") -> List[Dict[str, object]]:
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    normalized = _normalize_punctuation(_remove_non_printable(text.strip()))
    if not normalized.strip():
        return []

    candidate_blocks = []
    split_blocks = [block.strip() for block in PATIENT_SPLIT_PATTERN.split(normalized) if block.strip()]
    if len(split_blocks) > 1:
        candidate_blocks = split_blocks
    else:
        candidate_blocks = [block.strip() for block in re.split(r"\n\s*\n+", normalized) if block.strip()]

    if not candidate_blocks:
        candidate_blocks = [normalized.strip()]

    records: List[Dict[str, object]] = []
    for index, block in enumerate(candidate_blocks, start=1):
        patient_hint = _extract_patient_hint(block)
        records.append(
            {
                "patient_id": normalize_patient_id(patient_hint, index),
                "record_index": index,
                "source": source_name,
                "raw_text": block.strip(),
                "metadata": {"patient_hint": patient_hint, "source_name": source_name},
            }
        )
    return records
