from typing import Dict, List


def _collect(entities: List[Dict[str, object]], label: str) -> List[str]:
    seen = set()
    values = []
    for entity in entities:
        if entity["label"] != label:
            continue
        text = str(entity["text"]).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
    return values


def _join(values: List[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def generate_summary(entities: List[Dict[str, object]]) -> str:
    diseases = _collect(entities, "Disease")[:2]
    symptoms = _collect(entities, "Symptom")[:2]
    medications = _collect(entities, "Medication")[:2]
    procedures = _collect(entities, "Procedure")[:2]

    sentences: List[str] = []
    if diseases and symptoms:
        sentences.append(f"Patient has {_join(diseases)} and reports {_join(symptoms)}.")
    elif diseases:
        sentences.append(f"Patient has {_join(diseases)}.")
    elif symptoms:
        sentences.append(f"Patient reports {_join(symptoms)}.")

    action_parts = []
    if medications:
        action_parts.append(f"{_join(medications)} has been prescribed or mentioned")
    if procedures:
        action_parts.append(f"{_join(procedures)} was performed or recommended")
    if action_parts:
        sentences.append(" and ".join(action_parts) + ".")

    if not sentences:
        return "No clinically meaningful entities were extracted from the provided text."
    return " ".join(sentences[:2])
