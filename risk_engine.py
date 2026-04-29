from typing import Dict, List

from config import RISK_RULES


RISK_EXPLANATIONS = {
    "High": "High Risk: {terms} detected, which may be associated with serious clinical conditions.",
    "Medium": "Medium Risk: {terms} detected, which can indicate an active or worsening clinical issue.",
    "Low": "Low Risk: {terms} detected, which are usually lower-acuity findings in this ruleset.",
    "None": "Low Risk: no configured high, medium, or low risk terms were matched.",
}


def classify_risk(text: str, entities: List[Dict[str, object]]) -> Dict[str, object]:
    evidence = {str(entity["text"]).lower() for entity in entities}
    evidence.update(text.lower().split())
    full_text = text.lower()

    matched = {"High": [], "Medium": [], "Low": []}
    for level, terms in RISK_RULES.items():
        for term in terms:
            if term in full_text or term.lower() in evidence:
                matched[level].append(term)

    if matched["High"]:
        risk_level = "High"
    elif matched["Medium"]:
        risk_level = "Medium"
    elif matched["Low"]:
        risk_level = "Low"
    else:
        risk_level = "Low"

    matched_terms = matched["High"] + matched["Medium"] + matched["Low"]
    matched_terms = sorted(set(matched_terms))
    explanation = (
        RISK_EXPLANATIONS[risk_level].format(terms=", ".join(matched_terms))
        if matched_terms
        else RISK_EXPLANATIONS["None"]
    )

    return {
        "risk_level": risk_level,
        "matched_terms": matched_terms,
        "explanation": explanation,
        "matched_by_level": matched,
    }
