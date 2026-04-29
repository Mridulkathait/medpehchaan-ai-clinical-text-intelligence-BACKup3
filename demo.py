#!/usr/bin/env python3
"""
MedPehchaan AI+ - Demo Script
============================

This script demonstrates how to use the MedPehchaan AI+ clinical text intelligence
system programmatically without the Streamlit web interface.

Usage:
    python demo.py

Requirements:
    - All dependencies from requirements.txt
    - Sample clinical text data
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from intelligence import process_dataset
from preprocessing import split_text_into_patient_records
from config import load_model_meta

def demo_clinical_text_analysis():
    """
    Demonstrate clinical text analysis with sample patient data.
    """
    print("🏥 MedPehchaan AI+ - Clinical Text Intelligence Demo")
    print("=" * 60)

    # Sample clinical text data
    sample_text = """
    Patient ID: PAT_001
    45-year-old male presents with chest pain and shortness of breath.
    History of diabetes mellitus type 2, hypertension.
    Currently on metformin 500mg twice daily and lisinopril 10mg daily.
    ECG shows ST elevation. Troponin elevated at 0.8 ng/mL.
    Diagnosed with acute myocardial infarction.
    Started on aspirin 325mg, heparin infusion, and morphine for pain.

    Patient ID: PAT_002
    32-year-old female with fever and productive cough.
    Sputum culture positive for Streptococcus pneumoniae.
    Prescribed amoxicillin 500mg three times daily for 7 days.
    Advised rest and hydration. Follow up in 1 week.

    Patient ID: PAT_003
    28-year-old male with headache and photophobia.
    Diagnosed with migraine headache.
    Prescribed sumatriptan 100mg as needed and propranolol 40mg daily.
    Recommended stress reduction techniques.
    """

    print("📝 Sample Clinical Text:")
    print("-" * 30)
    print(sample_text.strip())
    print("\n" + "=" * 60)

    try:
        # Load model metadata
        print("🤖 Loading AI Models...")
        model_meta = load_model_meta()
        print(f"✅ Model loaded: {model_meta.get('model_name', 'Unknown')}")

        # Split text into patient records
        print("\n📊 Processing Patient Records...")
        patient_records = split_text_into_patient_records(sample_text)

        print(f"📋 Found {len(patient_records)} patient records")

        # Process the dataset
        print("\n🔍 Analyzing Clinical Text...")
        results = process_dataset(
            patient_records,
            batch_size=8,
            processing_chunk_size=100,
            progress_callback=None
        )

        # Display results
        print("\n📈 Analysis Results:")
        print("=" * 60)

        for i, patient in enumerate(results["patients"], 1):
            print(f"\n👤 Patient {i}: {patient['patient_id']}")
            print(f"   Risk Level: {patient['risk']['risk_level']}")
            print(f"   Entities Found: {len(patient['entities'])}")

            # Show extracted entities by type
            entities_by_type = {}
            for entity in patient["entities"]:
                entity_type = entity.get("label", "unknown")
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity.get("text", ""))

            for entity_type, texts in entities_by_type.items():
                print(f"   {entity_type.title()}: {', '.join(texts[:3])}{'...' if len(texts) > 3 else ''}")

            print(f"   Summary: {patient['summary'][:100]}...")

        # Aggregate statistics
        aggregate = results["aggregate_report"]
        print("
📊 Aggregate Statistics:"        print(f"   Total Patients: {aggregate['total_patients_processed']}")
        print(f"   Total Diseases Detected: {aggregate['total_diseases_detected']}")
        print(f"   Risk Distribution: High: {aggregate['overall_risk_distribution']['High']}, "
              f"Medium: {aggregate['overall_risk_distribution']['Medium']}, "
              f"Low: {aggregate['overall_risk_distribution']['Low']}")

        print("\n✅ Demo completed successfully!")
        print("\n💡 Tip: Run 'streamlit run app.py' for the full web interface experience!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        return False

    return True

def demo_file_processing():
    """
    Demonstrate processing of uploaded files.
    """
    print("\n" + "=" * 60)
    print("📁 File Processing Demo")
    print("=" * 60)

    sample_file_path = project_root / "data" / "sample_demo_input.txt"

    if sample_file_path.exists():
        print(f"📖 Processing sample file: {sample_file_path}")
        try:
            with open(sample_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            print(f"📏 File size: {len(file_content)} characters")
            print(f"📝 Content preview: {file_content[:200]}...")

            # Process the file content
            patient_records = split_text_into_patient_records(file_content)
            print(f"📋 Extracted {len(patient_records)} patient records from file")

        except Exception as e:
            print(f"❌ Error processing file: {e}")
    else:
        print(f"ℹ️ Sample file not found: {sample_file_path}")
        print("💡 Create sample_demo_input.txt in the data/ directory to test file processing")

if __name__ == "__main__":
    print("🚀 Starting MedPehchaan AI+ Demo...")
    print("This may take a moment to download AI models on first run.\n")

    # Run the main demo
    success = demo_clinical_text_analysis()

    if success:
        # Run file processing demo
        demo_file_processing()

    print("\n" + "=" * 60)
    print("🎓 Educational Demo Complete!")
    print("🔗 Repository: https://github.com/Mridulkathait/medpehchaan-ai-clinical-text-intelligence-BACKup3")
    print("📧 Questions? Check the README.md for contact information")
    print("=" * 60)