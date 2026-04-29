import gc
from collections import Counter
from math import ceil
from typing import Callable, Dict, Generator, List, Optional

from insight_engine import generate_insights
from ner_engine import extract_entities_batch
from preprocessing import preprocess_clinical_text, split_text_into_patient_records
from risk_engine import classify_risk
from summary_engine import generate_summary
from utils import highlight_entities_html, structured_output
from config import ENABLE_MEMORY_OPTIMIZATION, STREAMING_FLUSH_INTERVAL


def process_single_patient(
    patient_record: Dict[str, object],
    preprocessed: Optional[Dict[str, object]] = None,
    entities: Optional[List[Dict[str, object]]] = None,
    model_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    prep = preprocessed or preprocess_clinical_text(str(patient_record["raw_text"]))
    patient_entities = entities
    resolved_model_meta = model_meta

    if patient_entities is None:
        ner_result = extract_entities_batch([prep["clean_text"]], batch_size=1)
        patient_entities = ner_result["entities"][0]
        resolved_model_meta = ner_result["model_meta"]

    risk = classify_risk(str(prep["clean_text_lower"]), patient_entities)
    insights = generate_insights(patient_entities, risk["risk_level"])
    summary = generate_summary(patient_entities)

    return {
        "patient_id": patient_record["patient_id"],
        "record_index": patient_record["record_index"],
        "source": patient_record.get("source", "unknown"),
        "metadata": patient_record.get("metadata", {}),
        "raw_text": patient_record["raw_text"],
        "preprocessing": prep,
        "entities": patient_entities,
        "structured_data": {
            "patient_id": patient_record["patient_id"],
            **structured_output(patient_entities),
        },
        "risk": risk,
        "insights": insights,
        "summary": summary,
        "highlighted_html": highlight_entities_html(str(prep["clean_text"]), patient_entities),
        "model_meta": resolved_model_meta or {},
        "status": "ok" if patient_entities else "no_entities",
    }


def build_aggregate_report(patient_results: List[Dict[str, object]]) -> Dict[str, object]:
    symptom_counter: Counter[str] = Counter()
    disease_total = 0
    risk_counter: Counter[str] = Counter()

    for result in patient_results:
        risk_counter[result["risk"]["risk_level"]] += 1
        structured = result["structured_data"]
        disease_total += len(structured["diseases"])
        for item in structured["symptoms"]:
            symptom_counter[str(item["name"]).lower()] += 1

    most_common_symptoms = [
        {"symptom": symptom, "count": count}
        for symptom, count in symptom_counter.most_common(10)
    ]

    return {
        "total_patients_processed": len(patient_results),
        "total_diseases_detected": disease_total,
        "most_common_symptoms": most_common_symptoms,
        "overall_risk_distribution": {
            "High": risk_counter.get("High", 0),
            "Medium": risk_counter.get("Medium", 0),
            "Low": risk_counter.get("Low", 0),
        },
        "patients_with_no_entities": sum(1 for result in patient_results if not result["entities"]),
    }


def _empty_aggregate_state() -> Dict[str, object]:
    return {
        "symptom_counter": Counter(),
        "disease_total": 0,
        "risk_counter": Counter(),
        "patients_with_no_entities": 0,
        "total_patients_processed": 0,
    }


def _update_aggregate_state(aggregate_state: Dict[str, object], patient_result: Dict[str, object]) -> None:
    symptom_counter = aggregate_state["symptom_counter"]
    risk_counter = aggregate_state["risk_counter"]

    risk_counter[patient_result["risk"]["risk_level"]] += 1
    structured = patient_result["structured_data"]
    aggregate_state["disease_total"] += len(structured["diseases"])
    for item in structured["symptoms"]:
        symptom_counter[str(item["name"]).lower()] += 1

    aggregate_state["total_patients_processed"] += 1
    if not patient_result["entities"]:
        aggregate_state["patients_with_no_entities"] += 1


def _finalize_aggregate_report(aggregate_state: Dict[str, object]) -> Dict[str, object]:
    symptom_counter = aggregate_state["symptom_counter"]
    risk_counter = aggregate_state["risk_counter"]
    return {
        "total_patients_processed": aggregate_state["total_patients_processed"],
        "total_diseases_detected": aggregate_state["disease_total"],
        "most_common_symptoms": [
            {"symptom": symptom, "count": count}
            for symptom, count in symptom_counter.most_common(10)
        ],
        "overall_risk_distribution": {
            "High": risk_counter.get("High", 0),
            "Medium": risk_counter.get("Medium", 0),
            "Low": risk_counter.get("Low", 0),
        },
        "patients_with_no_entities": aggregate_state["patients_with_no_entities"],
    }


def process_dataset(
    patient_records: List[Dict[str, object]],
    batch_size: int = 8,
    processing_chunk_size: int = 256,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    enable_streaming: bool = False,
) -> Dict[str, object]:
    """
    Process patient records with memory optimization and optional streaming.
    
    Args:
        patient_records: List of patient records to process
        batch_size: NER batch size (larger for big datasets)
        processing_chunk_size: Records per chunk (larger for big datasets)
        progress_callback: Optional callback for progress updates
        enable_streaming: If True, yields results instead of collecting all
    
    Returns:
        Dictionary with processed patients, aggregate report, and model metadata
    """
    if not patient_records:
        return {"patients": [], "aggregate_report": build_aggregate_report([]), "model_meta": {}}

    processing_chunk_size = max(1, int(processing_chunk_size or 256))
    patient_results = []
    aggregate_state = _empty_aggregate_state()
    model_meta: Dict[str, object] = {}
    total_chunks = ceil(len(patient_records) / processing_chunk_size)

    for chunk_index, chunk_start in enumerate(range(0, len(patient_records), processing_chunk_size), start=1):
        chunk_records = patient_records[chunk_start : chunk_start + processing_chunk_size]
        
        # Preprocess chunk
        preprocessed_records = []
        for record in chunk_records:
            prep = preprocess_clinical_text(str(record["raw_text"]))
            preprocessed_records.append(prep)
        
        clean_texts = [record["clean_text"] for record in preprocessed_records]
        
        # Extract entities batch
        ner_result = extract_entities_batch(clean_texts, batch_size=batch_size)
        model_meta = ner_result["model_meta"]

        # Process each result
        for record, prep, entities in zip(chunk_records, preprocessed_records, ner_result["entities"]):
            patient_result = process_single_patient(
                patient_record=record,
                preprocessed=prep,
                entities=entities,
                model_meta=model_meta,
            )
            patient_results.append(patient_result)
            _update_aggregate_state(aggregate_state, patient_result)

        # Memory optimization
        if ENABLE_MEMORY_OPTIMIZATION:
            if chunk_index % max(1, STREAMING_FLUSH_INTERVAL // processing_chunk_size) == 0:
                gc.collect()  # Force garbage collection periodically

        if progress_callback is not None:
            progress_callback(chunk_index, total_chunks, len(patient_results))

    return {
        "patients": patient_results,
        "aggregate_report": _finalize_aggregate_report(aggregate_state),
        "model_meta": model_meta,
    }


def process_dataset_streaming(
    patient_records: List[Dict[str, object]],
    batch_size: int = 8,
    processing_chunk_size: int = 256,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> Generator[Dict[str, object], None, Dict[str, object]]:
    """
    Generator-based processing for very large datasets.
    Yields processed patients one at a time to minimize memory usage.
    
    Args:
        patient_records: List of patient records to process
        batch_size: NER batch size
        processing_chunk_size: Records per chunk
        progress_callback: Optional callback for progress updates
    
    Yields:
        Processed patient results one at a time
    
    Returns:
        Final aggregate report and model metadata
    """
    if not patient_records:
        return {"patients": [], "aggregate_report": build_aggregate_report([]), "model_meta": {}}

    processing_chunk_size = max(1, int(processing_chunk_size or 256))
    aggregate_state = _empty_aggregate_state()
    model_meta: Dict[str, object] = {}
    total_chunks = ceil(len(patient_records) / processing_chunk_size)
    results_count = 0

    for chunk_index, chunk_start in enumerate(range(0, len(patient_records), processing_chunk_size), start=1):
        chunk_records = patient_records[chunk_start : chunk_start + processing_chunk_size]
        
        # Preprocess chunk
        preprocessed_records = [preprocess_clinical_text(str(r["raw_text"])) for r in chunk_records]
        clean_texts = [p["clean_text"] for p in preprocessed_records]
        
        # Extract entities
        ner_result = extract_entities_batch(clean_texts, batch_size=batch_size)
        model_meta = ner_result["model_meta"]

        # Yield results one by one
        for record, prep, entities in zip(chunk_records, preprocessed_records, ner_result["entities"]):
            patient_result = process_single_patient(
                patient_record=record,
                preprocessed=prep,
                entities=entities,
                model_meta=model_meta,
            )
            _update_aggregate_state(aggregate_state, patient_result)
            results_count += 1
            
            if progress_callback is not None:
                progress_callback(chunk_index, total_chunks, results_count)
            
            yield patient_result

        # Aggressive memory cleanup in streaming mode
        if ENABLE_MEMORY_OPTIMIZATION:
            del preprocessed_records, clean_texts, ner_result
            gc.collect()

    return {
        "aggregate_report": _finalize_aggregate_report(aggregate_state),
        "model_meta": model_meta,
    }


def process_text_dataset(text: str, source_name: str = "text_input", batch_size: int = 8) -> Dict[str, object]:
    patient_records = split_text_into_patient_records(text, source_name=source_name)
    return process_dataset(patient_records, batch_size=batch_size)
