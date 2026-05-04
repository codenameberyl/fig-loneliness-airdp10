# FIG-Loneliness: Detecting Loneliness Self-Disclosure in Reddit Posts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> An empirical evaluation of text representations and classification models for automated loneliness self-disclosure detection in Reddit posts, using the FIG-Loneliness dataset (Jiang et al., ICWSM 2022).

---

## Research Questions

| ID | Question |
|----|----------|
| **RQ1** | What linguistic and structural characteristics differentiate Reddit posts expressing loneliness from those that do not? |
| **RQ2** | How well do baseline structural text classification models perform on this task? |
| **RQ3** | How do different text representations (TF-IDF, Word2Vec, SBERT, DistilBERT) affect predictive performance? |
| **RQ4** | What trade-offs exist between predictive performance and interpretability across representations? |

---

## Key Results

| Representation | Best Model | Test F1 | Test AUC |
|---|---|---|---|
| **DistilBERT** | DistilBERT | **0.9615** | **0.9965** |
| TF-IDF + Linguistic | Logistic Regression | 0.9570 | 0.9893 |
| Sentence-BERT | Logistic Regression | 0.9498 | 0.9923 |
| TF-IDF | Logistic Regression | 0.9474 | 0.9910 |
| Word2Vec | SVM | 0.9240 | 0.9813 |
| Linguistic Only | Random Forest | 0.8792 | 0.9441 |

---

## Repository Structure

```
fig-loneliness/
├── src/
│   ├── config.py                 # Global configuration, seeds, paths
│   ├── dataset_loader.py         # HuggingFace dataset loader
│   ├── eda.py                    # Exploratory data analysis & plots
│   ├── error_analysis.py         # FP/FN qualitative inspection
│   ├── evaluation.py             # Metrics, ROC, confusion matrix       
│   ├── features.py               # Feature extraction (13 linguistic + 5 representations)
│   ├── inference.py              # Inference with ROC-derived threshold
│   ├── interpretability.py       # LR coefficients & attention weights
│   ├── models.py                 # Classical ML & DistilBERT trainer
│   ├── preprocessing.py          # 8-step text preprocessing pipeline
│   ├── reproducibility.py        # seed_everything() — sets Python/NumPy/PyTorch seedspipeline
│   └── results                   # saving and loading cached objects, JSON results, and plots.
├── api/
│   ├── main.py                   # FastAPI application entry point
│   ├── routers/
│   │   ├── eda.py                # EDA data endpoints
│   │   ├── models.py             # Model results & evaluation endpoints
│   │   └── predict.py            # Single & batch inference endpoints
│   └── schemas.py                # Pydantic request/response models
├── results/                      # Pre-computed pipeline outputs (JSON + PNG)
│   ├── bert/
│   ├── cache/
│   ├── json/
│   └── plots/
├── run_pipeline.py               # Runs all 8 pipeline stages
├── runserver.py                  # Starts the FASTAPI server
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dataset

The **FIG-Loneliness** dataset (Jiang et al., 2022) is loaded automatically from HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("FIG-Loneliness/FIG-Loneliness")
```

| Split | Total | Lonely | Non-Lonely |
|-------|-------|--------|------------|
| Train | 3,943 | 1,840 | 2,103 |
| Validation | 1,126 | 536 | 590 |
| Test | 564 | 257 | 307 |

**Source**: r/loneliness, r/lonely, r/youngadults, r/college · 2018–2020 · CC BY-NC 4.0

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/codenameberyl/fig-loneliness-airdp10.git
cd fig-loneliness-airdp10
```

### 2. Create a Python environment

```bash
python -m venv venv
source venv/bin/activate         # Linux/macOS
# venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. (Recommended) Open in Google Colab for GPU access

DistilBERT fine-tuning requires a GPU. Use Google Colab with a T4 runtime:

```bash
# In a Colab cell:
!git clone https://github.com/codenameberyl/fig-loneliness-airdp10.git
%cd fig-loneliness-airdp10
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm
```

---

## Running the Pipeline

The pipeline is executed as a single script that runs all eight stages in sequence:

```bash
python run_pipeline.py
```

### Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1. `load_dataset` | `src/dataset.py` | Load FIG-Loneliness from HuggingFace |
| 2. `preprocess` | `src/preprocessing.py` | 8-step text cleaning pipeline |
| 3. `eda` | `src/eda.py` | Class distribution, n-grams, POS, word clouds |
| 4. `build_features` | `src/features.py` | 6 text representations |
| 5. `train_models` | `src/models.py` | 15 classical + 1 DistilBERT experiment |
| 6. `evaluation` | `src/evaluation.py` | F1, AUC, confusion matrix, ROC |
| 7. `error_analysis` | `src/error_analysis.py` | FP/FN qualitative inspection |
| 8. `interpretability` | `src/interpretability.py` | LR coefficients, attention weights |

All outputs are saved to `results/`.

---

## Running the API

```bash
python runserver.py
```

API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Pipeline completion status |
| GET | `/api/eda/dataset` | Dataset splits and class balance |
| GET | `/api/models/results` | All 16 experiment results |
| GET | `/api/models/interpretability/summary` | Interpretability analysis |
| POST | `/api/predict` | Classify a single text |
| POST | `/api/predict/batch` | Classify up to 100 texts |

### Example Inference Request

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been feeling completely alone lately, even when surrounded by people."}'
```

```json
{
  "label": 1,
  "label_name": "lonely",
  "confidence": 0.9423,
  "threshold_used": 0.4361,
  "representation": "sbert",
  "model": "lr"
}
```

---

## Deployment

Results are committed to HuggingFace Spaces, which hosts the FastAPI backend. The Next.js dashboard connects to the Space API.

```bash
# After running the pipeline:
git add results/
git commit -m "Add pipeline results"
git push  # Pushes to HuggingFace Space
```

- **API**: [https://codenameberyl-fig-lone.hf.space/api](https://codenameberyl-fig-lone.hf.space/api)
- **API Docs**: [https://codenameberyl-fig-lone.hf.space/docs](https://codenameberyl-fig-lone.hf.space/docs)
- **Dataset**: [https://huggingface.co/datasets/FIG-Loneliness/FIG-Loneliness](https://huggingface.co/datasets/FIG-Loneliness/FIG-Loneliness)

---

## Requirements

See `requirements.txt` for the full list. Core dependencies:

```
transformers==4.40.0
sentence-transformers==2.7.0
scikit-learn==1.4.2
gensim==4.3.2
spacy==3.7.4
ftfy==6.2.0
bleach==6.1.0
emoji==2.11.1
datasets==2.19.0
fastapi==0.111.0
uvicorn==0.29.0
torch==2.3.0
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.0
seaborn==0.13.2
```

---

## Reproducibility

All random seeds are fixed globally via `src/reproducibility.py`:

```python
from src.reproducibility import seed_everything
seed_everything(42)  # Python, NumPy, PyTorch, CUDA
```

---

## Citation

If you use this work or the FIG-Loneliness dataset, please cite:

```bibtex
@inproceedings{jiang2022ways,
  title={Many Ways to Be Lonely: Fine-Grained Characterization of Loneliness and Its Potential Changes in COVID-19},
  author={Jiang, Yiren and Jiang, Yiwen and Liu, Leqi and Winkielman, Piotr},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={16},
  pages={405--416},
  year={2022}
}
```

---

## Licence

Code: MIT · Dataset: CC BY-NC 4.0