from typing import Dict, List

from config import CHRONIC_DISEASE_TERMS


def generate_insights(entities: List[Dict[str, object]], risk_level: str) -> List[str]:
    diseases = [str(e["text"]).lower() for e in entities if e["label"] == "Disease"]
    symptoms = [str(e["text"]).lower() for e in entities if e["label"] == "Symptom"]
    medications = [str(e["text"]).lower() for e in entities if e["label"] == "Medication"]
    procedures = [str(e["text"]).lower() for e in entities if e["label"] == "Procedure"]

    insights: List[str] = []
    if any(term in CHRONIC_DISEASE_TERMS for term in diseases):
        insights.append("Chronic disease detected.")
    if medications:
        insights.append("Medication prescribed or mentioned.")
    if diseases and symptoms:
        insights.append("Symptom and disease correlation observed.")
    if procedures:
        insights.append("Clinical procedure or diagnostic test detected.")
    if risk_level in {"High", "Medium"}:
        insights.append(f"{risk_level} risk rule triggered from extracted findings.")
    if not insights:
        insights.append("No additional rule-based insight was supported by the extracted entities.")
    return insights
