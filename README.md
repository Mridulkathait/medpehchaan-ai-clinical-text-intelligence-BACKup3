# 🏥 MedPehchaan AI+ - Intelligent Clinical Text Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![AI/ML](https://img.shields.io/badge/AI-Transformers-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Class_Project-blueviolet.svg)

**🎓 Class Project: Advanced AI-Powered Clinical Text Analysis**

*Transform clinical text into actionable medical insights using cutting-edge AI*

[🚀 Live Demo](#-quick-start) • [📖 Documentation](#-features) • [🛠️ Installation](#-installation)

</div>

---

## ⚠️ Medical Disclaimer

**This is an educational/research prototype for demonstration purposes only.**  
**NOT for clinical diagnosis, treatment decisions, or medical practice.**  
Always consult qualified healthcare professionals for medical advice.

---

## 🌟 Overview

MedPehchaan AI+ is a sophisticated web application that leverages state-of-the-art AI models to analyze clinical text and extract critical medical information. Built with modern web technologies and advanced natural language processing, it provides healthcare professionals and researchers with powerful tools for clinical text intelligence.

### 🎯 Key Capabilities

- **📝 Multi-format Input**: Process typed text, PDFs, CSVs, Excel files, and JSONL datasets
- **🔍 Advanced NER**: Biomedical entity extraction using transformer models
- **📊 Risk Assessment**: Intelligent patient risk classification
- **💡 Clinical Insights**: AI-generated medical insights and recommendations
- **📋 Automated Summaries**: Concise, entity-grounded clinical summaries
- **🎨 Modern UI**: Beautiful, responsive interface with real-time processing
- **📈 Analytics Dashboard**: Comprehensive patient and aggregate analytics

---

## ✨ Features

### 🤖 AI-Powered Analysis
- **Biomedical NER**: Extracts diseases, symptoms, medications, and procedures
- **Confidence Scoring**: Quality assessment for all extracted entities
- **Noise Filtering**: Removes low-quality and irrelevant text spans
- **Context Preservation**: Maintains clinical meaning during processing

### 📊 Data Processing
- **Large Dataset Support**: Handles 100k+ patient records efficiently
- **Chunked Processing**: Memory-optimized for large-scale analysis
- **Multiple Formats**: CSV, TSV, Excel, PDF, TXT, JSONL support
- **Streaming Mode**: For extremely large datasets

### 🎨 User Experience
- **Modern UI**: Gradient-based design with glassmorphism effects
- **Real-time Processing**: Live progress indicators and status updates
- **Interactive Dashboard**: Patient-wise and aggregate analysis views
- **Download Reports**: PDF and CSV export capabilities

### 🔒 Safety & Quality
- **Medical Compliance**: Designed with healthcare standards in mind
- **Quality Assurance**: Built-in validation and error handling
- **Transparent Processing**: Clear visibility into AI decision-making
- **Educational Focus**: Optimized for learning and demonstration

---

## 🏗️ Architecture

```
MedPehchaan AI+
├── 🎨 UI Layer (Streamlit)
│   ├── Modern Web Interface
│   ├── Real-time Processing
│   └── Interactive Dashboards
├── 🧠 Intelligence Engine
│   ├── Biomedical NER
│   ├── Risk Assessment
│   ├── Insight Generation
│   └── Summary Creation
├── 🔧 Processing Pipeline
│   ├── Text Preprocessing
│   ├── Entity Extraction
│   ├── Post-processing
│   └── Report Generation
└── 📊 Analytics & Reporting
    ├── Patient Analysis
    ├── Aggregate Reports
    └── Export Functions
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mridulkathait/medpehchaan-ai-clinical-text-intelligence-BACKup3.git
   cd medpehchaan-ai-clinical-text-intelligence-BACKup3
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start analyzing clinical text!

---

## 📖 Usage Guide

### Input Methods

1. **📝 Typed Text**: Direct text input for quick analysis
2. **📎 File Upload**: Support for multiple formats:
   - `.txt` - Plain text files
   - `.pdf` - PDF documents
   - `.csv` - Comma-separated values
   - `.tsv` - Tab-separated values
   - `.xlsx` - Excel spreadsheets
   - `.jsonl` - JSON Lines format

### Analysis Workflow

1. **Input Selection**: Choose your clinical text source
2. **Processing**: AI analyzes text for medical entities
3. **Risk Assessment**: Automatic risk level classification
4. **Insights Generation**: AI-powered clinical insights
5. **Report Generation**: Downloadable PDF/CSV reports

### Output Formats

- **Patient Reports**: Individual patient analysis with entities, risks, and insights
- **Aggregate Reports**: Population-level analytics and trends
- **Visual Analytics**: Interactive charts and dashboards
- **Export Options**: PDF reports and CSV data exports

---

## 🛠️ Technical Details

### Dependencies
- **streamlit**: Modern web app framework
- **transformers**: Hugging Face transformers for AI models
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **PyPDF2**: PDF text extraction
- **openpyxl**: Excel file processing

### AI Models
- **Biomedical NER**: Clinical entity recognition models
- **Risk Classification**: Rule-based and ML-powered assessment
- **Text Summarization**: Advanced summarization techniques
- **Insight Generation**: Pattern-based clinical insights

### Performance
- **Large Dataset Support**: Optimized for 100k+ records
- **Memory Efficient**: Chunked processing for scalability
- **Fast Processing**: GPU acceleration support
- **Real-time Updates**: Live progress indicators

---

## 📊 Sample Output

### Patient Analysis
```
Patient ID: PAT_001
Risk Level: Medium
Entities Found: 12
- Diseases: Diabetes, Hypertension
- Symptoms: Chest pain, Fatigue
- Medications: Aspirin, Metformin
- Procedures: ECG, Blood test

Clinical Insights:
• Monitor blood glucose levels closely
• Consider cardiovascular risk assessment
• Regular follow-up recommended
```

### Aggregate Analytics
- Total Patients: 1,247
- High Risk: 23% | Medium Risk: 45% | Low Risk: 32%
- Most Common Diseases: Diabetes (28%), Hypertension (22%)
- Average Entities per Patient: 8.3

---

## 🎓 Educational Value

This project demonstrates:

- **🤖 AI/ML Integration**: Real-world application of transformers
- **🏥 Healthcare AI**: Medical NLP and clinical decision support
- **🎨 UI/UX Design**: Modern web application development
- **📊 Data Engineering**: Large-scale data processing pipelines
- **🔒 Ethical AI**: Responsible AI development practices

### Learning Objectives
- Advanced Python programming
- Machine learning model deployment
- Web application development
- Healthcare data processing
- UI/UX design principles
- Software engineering best practices

---

## 🚀 Deployment

### Render Deployment (Recommended)

1. **Push to GitHub**: Upload your project to a GitHub repository

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Create a new Web Service
   - Connect your GitHub repository

3. **Configure Build Settings**:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.headless true --server.runOnSave false`

4. **Environment Variables** (Optional):
   - Add `HF_TOKEN` if using Hugging Face authentication for faster downloads

5. **Deploy**: Click "Create Web Service" and wait for deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### Alternative Platforms

- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Requires `Procfile` with `web: streamlit run app.py --server.port $PORT`
- **Vercel/Netlify**: Not recommended (not optimized for ML workloads)

---

## 🤝 Contributing

This is a class project, but contributions and feedback are welcome!

### Ways to Contribute
- 🐛 Bug reports and fixes
- ✨ Feature suggestions
- 📖 Documentation improvements
- 🎨 UI/UX enhancements
- 🔧 Performance optimizations

### Development Setup
```bash
# Fork the repository
# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
# Commit changes
git commit -m "Add amazing feature"

# Push to branch
git push origin feature/amazing-feature

# Create Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License - Academic/Research Use Only
Not for commercial medical applications without proper validation
```

---

## 🙏 Acknowledgments

- **Hugging Face** for transformer models and datasets
- **Streamlit** for the amazing web app framework
- **Open-source AI Community** for research and tools
- **Healthcare AI Research** for inspiration and guidance

---

## 📞 Contact

**Project Author**: Mridul Kathait  
**Institution**: Manav Rachna University , Faridabad 
**Course**: Btech CSE AIML
**Project**: Class Assignment - AI Clinical Text Intelligence

For questions or feedback:
- 📧 Kathaitmridul@gmail.com
- 🔗 www.linkedin.com/in/mridul-kathait-66a684281

---

<div align="center">

**Made with ❤️ for learning and healthcare innovation**

⭐ **Star this repository** if you found it helpful!

[⬆️ Back to Top](#-medpehchaan-ai--intelligent-clinical-text-intelligence-system)

</div>
```

## Demo Input

Use `data/sample_demo_input.txt` as a quick test sample.

## Notes

- This app is built to prefer **precision over recall**.
- If confidence is weak, entities are filtered or flagged instead of being forced.
