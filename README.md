# Parameter-Efficient Deep Learning for Computational Drug Discovery 💊🧠

> Graph Neural Networks and LoRA-enhanced ChemBERTa for hERG cardiotoxicity prediction and multi-target binding affinity estimation.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11%2B-orange?style=flat-square)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

This project provides a deep learning pipeline for two critical tasks in early-stage drug safety and discovery:

1. **hERG Cardiotoxicity Prediction** — Binary classification of whether a compound inhibits the hERG potassium channel, a major cause of drug-induced cardiac arrhythmia and drug withdrawal.
2. **Multi-Target Binding Affinity Estimation** — Drug-target interaction (DTI) prediction using the DAVIS kinase dataset.

Two complementary architectures are explored:
- **ChemBERTa** (RoBERTa pretrained on 10M PubChem SMILES) with optional **LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Graph Neural Networks (GNN/GCN/GAT)** operating directly on molecular graphs

---

## Repository Structure

```
Parameter-Efficient-Drug-Discovery/
├── src/
│   ├── cardiotoxicity_chemberta.py          # ChemBERTa fine-tuning for hERG
│   ├── cardiotoxicity_gnn.py                # GNN-based hERG classifier
│   ├── cardiotoxicity_molecular_descriptor.py  # Molecular descriptor baseline
│   ├── fine_tuned_chemberta.py              # LoRA-enhanced fine-tuning script
│   ├── belka_gnn.py                         # GNN for BELKA binding prediction
│   └── davis_test.py                        # Binding affinity on DAVIS dataset
├── notebooks/
│   ├── MS_CardioToxicity_Experiment.ipynb   # Main cardiotoxicity experiments
│   └── MS_CardioToxicity_GCN_GAT.ipynb      # GCN/GAT comparison notebook
├── data/
│   ├── herg_train.csv                       # hERG training set
│   ├── herg_val.csv                         # hERG validation set
│   ├── herg_test.csv                        # hERG test set
│   ├── hERG_IC50.csv                        # Raw IC50 data
│   ├── paper_valid_data.csv                 # External validation set
│   ├── DAVIS.csv                            # DAVIS kinase dataset
│   └── toxic.csv                            # Additional toxicity labels
├── requirements.txt
└── README.md
```

---

## Background

### hERG Cardiotoxicity

The **hERG (human Ether-à-go-go Related Gene)** potassium channel is a major anti-target in drug development. Unintended hERG inhibition causes Long QT syndrome — a life-threatening cardiac arrhythmia — and is a leading cause of drug withdrawal from the market. Early *in silico* prediction of hERG liability is therefore essential in drug pipelines.

This project frames hERG prediction as **binary classification** using a pChEMBL IC₅₀ threshold of **6.5**:

- **Label 1**: hERG blocker (IC₅₀ ≤ ~316 nM)
- **Label 0**: non-blocker

### DAVIS Binding Affinity

The [DAVIS dataset](https://tdcommons.ai/multi_pred_tasks/dti/) contains measured binding affinities (Kd) between 68 kinases and 442 inhibitors. It is used here to benchmark drug-target interaction (DTI) prediction.

---

## Model Architectures

### 1. ChemBERTa + LoRA

```
SMILES String
     │
     ▼
BPE Tokenizer (PubChem10M vocabulary)
     │
     ▼
ChemBERTa Encoder (RoBERTa base)
  [with optional LoRA low-rank adapters injected into Linear layers]
     │
     ▼
[CLS] token representation
     │
     ▼
Linear Classification Head (2-class)
     │
     ▼
Blocker / Non-Blocker
```

**Pretrained model:** `seyonec/PubChem10M_SMILES_BPE_396_250`  
**Fine-tuning objective:** CrossEntropyLoss  
**Optimizer:** AdamW (`lr=2e-5`, `eps=1e-8`)  
**Scheduler:** Linear warmup

### 2. Graph Neural Network (GNN / GCN / GAT)

Molecules are converted to graphs (atoms = nodes, bonds = edges) and processed by graph convolutional layers, allowing the model to learn directly from molecular topology.

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/FairuzShadmaniShishir/Parameter-Efficient-Deep-Learning-Models-for-Computational-Drug-Discovery.git
cd Parameter-Efficient-Deep-Learning-Models-for-Computational-Drug-Discovery
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Dataset

The hERG dataset CSV files require the following columns:

| Column               | Description                                    |
|----------------------|------------------------------------------------|
| `smiles_standarized` | Standardized SMILES string                     |
| `label`              | Binary label: `1` = blocker, `0` = non-blocker |

> **Note:** `paper_valid_data.csv` uses a column named `SMILES` (uppercase). The script handles this automatically.

**Building your own dataset from raw ChEMBL IC₅₀ data:**
```python
df['label'] = df['pChEMBL Value'].apply(lambda x: 1 if x >= 6.5 else 0)
```

---

## Usage

### hERG Cardiotoxicity — ChemBERTa
```bash
python src/cardiotoxicity_chemberta.py
```

### hERG Cardiotoxicity — GNN
```bash
python src/cardiotoxicity_gnn.py
```

### DAVIS Binding Affinity
```bash
python src/davis_test.py
```

### Key hyperparameters (top of each script)

| Parameter    | Default | Description                          |
|--------------|---------|--------------------------------------|
| `k_folds`    | 5       | Number of stratified CV folds        |
| `batch_size` | 4       | Reduce if GPU runs out of memory     |
| `epochs`     | 10      | Training epochs per fold             |

GPU is detected automatically:
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

---

## LoRA Fine-Tuning (Parameter-Efficient)

LoRA injects trainable low-rank matrices alongside frozen Linear layers, reducing trainable parameters from ~125M (full ChemBERTa) to as few as ~300K.

```
output = W·x  +  α · (A · B) · x
          ↑              ↑
     frozen original   trainable LoRA (rank r)
```

**Enable LoRA** in `src/fine_tuned_chemberta.py` by calling:
```python
replace_linear_with_lora(model, rank=16, alpha=16)
```

### LoRA Hyperparameters

| Parameter | Recommended | Description                                              |
|-----------|-------------|----------------------------------------------------------|
| `rank`    | 16          | Rank of matrices A and B. Try 4, 8, 16, 32.             |
| `alpha`   | 16          | Scaling factor. Setting `alpha = rank` keeps LR stable. |

**Tuning guide:**
- `rank=4` → most aggressive compression, fastest training
- `rank=16` → best balance of capacity and efficiency ✓
- `rank=32` → closer to full fine-tuning quality

---

## Evaluation Metrics

| Metric          | Description                                                               |
|-----------------|---------------------------------------------------------------------------|
| **MCC**         | Matthews Correlation Coefficient — robust for imbalanced classes          |
| **ROC-AUC**     | Area under the ROC curve                                                  |
| **Accuracy**    | Overall classification accuracy                                           |
| **F1 Score**    | Harmonic mean of precision and recall                                     |
| **Sensitivity** | True Positive Rate (recall for blockers)                                  |
| **Specificity** | True Negative Rate (recall for non-blockers)                              |

---

## Results

## Results
 
All results reported as mean ± std over cross-validation folds unless otherwise noted.
 
### hERG Cardiotoxicity
 
| Model | Accuracy | MCC | F1 | AUC |
|-------|----------|-----|----|-----|
| Transformer-Morgan | 0.84 ± 0.0092 | 0.67 ± 0.0303 | 0.87 ± 0.0098 | 0.93 ± 0.0143 |
| Transformer-MACCS | 0.83 ± 0.0167 | 0.66 ± 0.0498 | 0.86 ± 0.0123 | 0.90 ± 0.0171 |
| Transformer-FP2 | 0.83 ± 0.0181 | 0.66 ± 0.0407 | 0.86 ± 0.0072 | 0.90 ± 0.0071 |
| Transformer-AtomPairs | 0.84 ± 0.0123 | 0.66 ± 0.0359 | 0.87 ± 0.0073 | 0.90 ± 0.0098 |
| GCN | 0.84 ± 0.0151 | 0.66 ± 0.0280 | 0.87 ± 0.0076 | 0.90 ± 0.0096 |
| GAT | 0.79 ± 0.0088 | 0.56 ± 0.0152 | 0.85 ± 0.0056 | 0.86 ± 0.0125 |
| **Proposed Model** | **0.87 ± 0.0018** | **0.72 ± 0.0038** | **0.90 ± 0.0018** | **0.93 ± 0.0046** |
 
### Binding Affinity Prediction (BELKA — sEH, HSA, BRD4 targets)
 
#### AUCPR
 
| Model | sEH | HSA | BRD4 |
|-------|-----|-----|------|
| Gradient Boosting | 0.96 | 0.60 | 0.86 |
| MolTrans | 0.98 | 0.93 | 0.97 |
| DeepCDA | 0.99 | 0.95 | 0.97 |
| InceptionDTA | 0.99 | 0.95 | 0.98 |
| **Proposed** | **0.99** | **0.95** | **0.99** |
 
#### AUROC
 
| Model | sEH | HSA | BRD4 |
|-------|-----|-----|------|
| MolTrans | 0.98 | 0.94 | 0.96 |
| DeepCDA | 0.99 | 0.95 | 0.98 |
| InceptionDTA | 0.99 | 0.95 | 0.98 |
| **Proposed** | **0.99** | **0.96** | **0.98** |
 
#### Test Set Size
 
| Target | Positive Samples | Negative Samples | Total |
|--------|-----------------|-----------------|-------|
| sEH | 39,701 | 40,299 | 80,000 |
| HSA | 39,874 | 40,126 | 80,000 |
| BRD4 | 40,153 | 39,847 | 80,000 |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{shishir2024drugdiscovery,
  author    = {Fairuz Shadmani Shishir},
  title     = {Parameter-Efficient Deep Learning Models for Computational Drug Discovery},
  year      = {2024},
  url       = {https://github.com/FairuzShadmaniShishir/Parameter-Efficient-Deep-Learning-Models-for-Computational-Drug-Discovery}
}
```

---

## Contact

**Fairuz Shadmani Shishir**
- 📧 shishir@ku.edu
- 📧 fsshishir95@gmail.com

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
