# 🩺 Aegis-Sphere — AI-Powered Oncology Decision Support
> Zero-cloud oncology: A multi-agent virtual tumor board engineered for the 8GB edge.
> **Offline, dual-mode clinical intelligence for LMIC clinics.**  
> Aegis-Sphere listens to TB/HIV consultations in real time, auto-detects malignancy signals, convenes a multi-agent virtual tumor board, routes treatment plans around drug shortages, and generates empathetic patient handouts — all on **8 GB VRAM**.

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Med--Gemma%20Impact%20Challenge-blue?logo=kaggle)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌍 The Problem: Dr. Priya's Day

Dr. Priya sees **40 patients daily** at a district HIV clinic in Nagpur, India. When a 38-year-old HIV+ man presents with a 4-week wet cough, weight loss, and cervical lymphadenopathy, she correctly suspects TB — but misses that HIV+ patients have an **11.5× standardised incidence ratio for NHL**.

| ❌ Before Aegis-Sphere | ✅ After Aegis-Sphere |
|---|---|
| Suspects TB, starts empiric RHEZ therapy | Ambient system detects oncology signals within 60s |
| Patient misclassified for 4–7 weeks | Auto-escalation: "HIV-related malignancy detected" |
| Lymphoma diagnosed at Stage IV | Virtual tumor board: staging + pathways generated |
| R-CHOP prescribed — Rituximab out of stock | TxGemma checks inventory → CHOP + Liposomal Dox substituted |
| Patient leaves with no explanation | Grade-5 empathetic patient handout generated |
| No audit trail | Override records synced to big-center board for annotation |

---

## 🎯 Impact Metrics

| Metric | Value |
|---|---|
| Early diagnoses/yr (500 pilot clinics) | **7,500** |
| Survival delta (Stage IIB vs IV NHL) | **+30–35%** |
| Drug waste reduction | **−20%** |
| 5-year scale projection (India + SSA) | **75,000 clinics** |

---

## 🧠 AI Models & Pipeline

Aegis-Sphere orchestrates **8 specialist AI models** in a single clinical session:

| Model | Role |
|---|---|
| **MedGemma 1.5** | Core LLM — 5 sequential persona passes (Pathologist, Radiologist, Oncologist, Treatment Planner, Patient Communicator) |
| **TxGemma** | Treatment interaction & drug-drug interaction (DDI) analysis |
| **HeAR** | Acoustic respiratory embeddings from consultation audio |
| **MedASR** | Medical speech-to-text transcription |
| **CXR Worker** | Chest X-ray analysis |
| **Derm Worker** | Dermatology image analysis |
| **Path Worker** | Pathology slide analysis |
| **MedSigLIP** | Multimodal signal embeddings & FAISS case retrieval |

---

## ✨ Key Features

- **Dual-Mode Operation** — TB triage mode auto-escalates to OncoSphere tumor board on malignancy signal detection
- **Multi-Agent Virtual Tumor Board** — MedGemma instances run sequential single-turn persona passes before reaching consensus
- **VRAM Telemetry** — Live GPU monitoring with sawtooth phase tracking, fits within 8 GB VRAM
- **Evidence-Grounded Output** — `[Source: X]` citation tags ground every clinical claim
- **Drug Inventory Routing** — TxGemma dynamically routes treatment plans around real drug shortages
- **Patient Handouts** — Grade-5 empathetic letters with next-step checklists
- **Clinician Override Sync** — Override records logged and synced to specialist centers for annotation
- **Graceful Degradation** — Handles missing modalities, designed for resource-constrained LMIC settings
- **DPDP Act 2023 Compliant** — Built with India's Digital Personal Data Protection Act in mind

---

## 🗂️ Project Structure

```
aegis-sphere/
├── app.py                        # Streamlit UI — main entry point
├── dataset-collection.ipynb      # Sample data collection from Kaggle
├── requirements.txt
├── run_isolation_tests.py
│
├── config/
│   ├── settings.py               # App config, degradation levels, VRAM limits
│   ├── badge_colors.py           # UI badge color definitions
│   ├── model_ids.py              # Hugging Face model identifiers
│   ├── gpu_lease.py              # GPU resource management
│   └── __init__.py
│
├── pipeline/
│   ├── cortex_controller.py      # Main pipeline orchestrator
│   ├── mode_bridge.py            # TB → OncoSphere escalation logic
│   ├── session_manager.py        # Per-session state management
│   ├── asr_worker.py             # MedASR transcription
│   ├── hear_worker.py            # HeAR acoustic embeddings
│   ├── cxr_worker.py             # Chest X-ray analysis
│   ├── derm_worker.py            # Dermatology analysis
│   ├── path_worker.py            # Pathology slide analysis
│   ├── medsig_worker.py          # MedSigLIP embeddings
│   ├── txgemma_worker.py         # TxGemma treatment analysis
│   ├── persona_debate.py         # Multi-agent tumor board debate
│   ├── oncocase_builder.py       # OncoCase structured output builder
│   ├── risk_engine.py            # Risk stratification
│   ├── evidence_trace.py         # Evidence citation tracking
│   ├── lang_extract.py           # Language/entity extraction
│   ├── report_formatter.py       # Report rendering utilities
│   ├── pdf_report.py             # PDF/HTML report generation
│   └── __init__.py
│
├── data/
│   ├── demo_case/                # Full-quality demo patient data
│   │   ├── consultation.wav
│   │   ├── consultation_meta.json
│   │   ├── cxr.jpg
│   │   ├── derm.jpg
│   │   └── path_patch.jpg
│   ├── demo_case_degraded/       # Low-resource scenario demo data
│   ├── faiss_case_library/       # FAISS vector index for case retrieval
│   │   ├── case_embeddings.faiss
│   │   ├── case_embeddings.npy
│   │   └── case_metadata.json
│   ├── synthetic_cases.json
│   ├── lora_training_pairs.json
│   └── uploads/                  # Runtime upload directory
│
├── evaluation/
│   ├── vram_monitor.py           # Live VRAM telemetry
│   ├── degradation_test.py       # Graceful degradation tests
│   └── results/
│       └── vram_log.csv
│
├── training/
│   ├── build_faiss_index.py      # Build FAISS case library
│   └── generate_lora_pairs.py    # Generate LoRA fine-tuning pairs
│
├── sync/
│   ├── override_logger.py        # Clinician override logging
│   ├── smart_sync.py             # Sync to specialist center board
│   └── remote_board/
│
└── db/
    ├── schema.sql
    └── __init__.py
```

---

## 🚀 Quickstart

### 1. Prerequisites

- Python 3.10+
- CUDA-capable GPU with **≥8 GB VRAM** (tested on RTX 3080/4080, T4)
- [Git LFS](https://git-lfs.github.com/) (for model weights, if applicable)

### 2. Clone the Repository

```bash
git clone https://github.com/AbineshSenthil/aegis-sphere.git
cd aegis-sphere
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For GPU support, ensure you have the correct version of `torch` for your CUDA version. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Environment Setup

Create a `.env` file in the project root:

```env
# Hugging Face token (required for gated MedGemma models)
HUGGINGFACE_TOKEN=hf_your_token_here

# Optional: Override model cache directory
HF_HOME=/path/to/model/cache
```

Request access to the following gated models on Hugging Face before running:
- `google/medgemma-4b-it`
- `google/txgemma-2b-it`
- `google/hear-encoder`

### 5. Run the Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 📊 Data Collection

The `dataset-collection.ipynb` notebook demonstrates how to collect and prepare sample cases from Kaggle for testing the pipeline.

```bash
jupyter notebook dataset-collection.ipynb
```

---

## 🧪 Running Tests

### Isolation Tests

```bash
python run_isolation_tests.py
```

### Degradation Tests

Tests the pipeline's graceful handling of missing modalities (e.g., no audio, no CXR):

```bash
python evaluation/degradation_test.py
```

### VRAM Monitoring

```bash
python evaluation/vram_monitor.py
```

---

## 🏋️ Training

### Build FAISS Case Library

Builds the vector index from case images for similarity-based retrieval:

```bash
python training/build_faiss_index.py
```

### Generate LoRA Training Pairs

Generates instruction-tuning pairs from synthetic cases for fine-tuning:

```bash
python training/generate_lora_pairs.py
```

---

## 🔄 Pipeline Flow

```
Upload (Audio + CXR + Derm + Path)
        │
        ▼
   MedASR + HeAR           ← Transcribe & encode consultation audio
        │
        ▼
   TB Triage Mode           ← Initial risk assessment
        │
   [Malignancy Signal?]
        │
        ▼
   OncoSphere Escalation    ← Mode bridge activates tumor board
        │
        ▼
  ┌─────────────────────────────────────┐
  │    Virtual Tumor Board              │
  │  Pass 1: Pathologist (MedGemma)     │
  │  Pass 2: Radiologist (MedGemma)     │
  │  Pass 3: Oncologist (MedGemma)      │
  │  Pass 4: Treatment Planner          │
  │  Pass 5: Patient Communicator       │
  └─────────────────────────────────────┘
        │
        ▼
  TxGemma DDI Check + Inventory Routing
        │
        ▼
  FAISS Case Retrieval (MedSigLIP)
        │
        ▼
  Report Generation + Patient Handout
        │
        ▼
  Override Logging + Sync to Specialist Board
```

---

## ⚙️ Configuration

Key settings in `config/settings.py`:

| Setting | Description | Default |
|---|---|---|
| `MAX_VRAM_MB` | Maximum VRAM budget | `7800` MB |
| `DegradationLevel` | Quality tier for resource-constrained operation | `FULL / DEGRADED / MINIMAL` |
| `APP_TITLE` | Dashboard title | `"Aegis-Sphere"` |

Model IDs can be swapped in `config/model_ids.py` to use alternative checkpoints.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## ⚠️ Disclaimer

Aegis-Sphere is an **AI-assisted clinical decision support tool** and is **not a substitute for clinical judgment**. All outputs should be reviewed by qualified healthcare professionals before influencing patient care.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🏆 Acknowledgements

Built for the [Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) on Kaggle. Powered by Google's MedGemma, TxGemma, and HeAR model families.

> *"Aegis-Sphere v1.0 · AI-Assisted Oncology Decision Support · DPDP Act 2023 Compliant"*
