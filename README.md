# Parameter-Efficient-Deep-Learning-Models-for-Computational-Drug-Discovery

# 🧬 ChemBERT-hERG: SMILES-Based hERG Cardiotoxicity Prediction

A deep learning pipeline using **ChemBERTa** (RoBERTa pretrained on PubChem SMILES) for binary classification of hERG channel blockers — a critical step in early-stage drug safety screening.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Citation](#citation)

---

## Overview

This project fine-tunes [ChemBERTa (`seyonec/PubChem10M_SMILES_BPE_396_250`)](https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_396_250) on hERG inhibition data to predict whether a given compound (represented as a SMILES string) is a hERG channel blocker. Classification is based on a pChEMBL IC₅₀ threshold of **6.5** (i.e., IC₅₀ ≤ 316 nM → blocker).

**Key features:**
- Tokenizes SMILES strings using a BPE tokenizer trained on 10M PubChem compounds
- Fine-tunes a RoBERTa-based sequence classifier
- Supports **stratified k-fold cross-validation**
- Reports MCC, ROC-AUC, Accuracy, F1, Sensitivity, and Specificity

---

## Background

The **hERG (human Ether-à-go-go Related Gene)** potassium channel is a major anti-target in drug development. Unintended hERG inhibition can cause life-threatening cardiac arrhythmias (Long QT syndrome), and is a leading cause of drug withdrawal from the market. Early *in silico* prediction of hERG liability is therefore essential.

This model treats hERG prediction as a **binary classification problem**:
- **Label 1**: hERG blocker (pChEMBL ≥ 6.5, IC₅₀ ≤ ~316 nM)
- **Label 0**: non-blocker

---

## Model Architecture

```
SMILES String
     │
     ▼
BPE Tokenizer (PubChem10M vocabulary)
     │
     ▼
ChemBERTa Encoder (RoBERTa base)
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
**Scheduler:** Linear warmup schedule

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/chembert-herg.git
cd chembert-herg
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The model expects three CSV files with the following columns:

| Column               | Description                              |
|----------------------|------------------------------------------|
| `smiles_standarized` | Standardized SMILES string               |
| `label`              | Binary label: `1` = blocker, `0` = non-blocker |

Place them in a `data/herg/` directory (or update paths in the script):

```
data/
└── herg/
    ├── herg_train.csv
    ├── herg_val.csv
    ├── herg_test.csv
    └── paper_valid_data.csv   # external validation set (uses 'SMILES' column)
```

> **Note:** `paper_valid_data.csv` uses a column named `SMILES` (uppercase) rather than `smiles_standarized`. The script handles this automatically.

### Labeling convention

If you are building your own dataset from raw ChEMBL IC₅₀ data:

```python
df['label'] = df['pChEMBL Value'].apply(lambda x: 1 if x >= 6.5 else 0)
```

---

## Usage

### Run the full pipeline (cross-validation + test evaluation)

```bash
python train_chembert_herg.py
```

### Customize key parameters

Edit these variables at the top of `train_chembert_herg.py`:

```python
k_folds   = 5      # Number of stratified CV folds
batch_size = 4     # Batch size (reduce if OOM)
epochs    = 10     # Epochs per fold
```

### GPU support

The script automatically detects and uses a CUDA GPU if available:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

---

## Training

Training uses **Stratified K-Fold Cross-Validation** to ensure balanced class distribution across folds. For each fold:

1. The model is **reset** (re-initialized from the pretrained checkpoint)
2. Fine-tuned for `epochs` epochs on the fold's training split
3. Evaluated on the held-out validation split

At the end, the **last fold's model** is used to evaluate on the external test set (`paper_valid_data.csv`).

```
Fold 1/5: Train → Validate → Record metrics
Fold 2/5: Train → Validate → Record metrics
...
Fold 5/5: Train → Validate → Evaluate on test set
```

**To use a simple train/val split instead**, comment out the k-fold block and uncomment the `random_split` section.

---

## Evaluation

The following metrics are reported per fold and on the final test set:

| Metric | Description |
|--------|-------------|
| **MCC** | Matthews Correlation Coefficient — balanced metric for imbalanced classes |
| **ROC-AUC** | Area under the ROC curve |
| **Accuracy** | Overall classification accuracy |
| **F1 Score** | Harmonic mean of precision and recall |
| **Sensitivity** | True Positive Rate (recall for positives) |
| **Specificity** | True Negative Rate (recall for negatives) |

Sample output:

```
==== Cross-validation Results ====
Avg Accuracy: 0.8712 ± 0.0143
Avg MCC:      0.7341 ± 0.0217
Avg AUC:      0.9105 ± 0.0089

Matthews Correlation Coefficient (MCC): 0.7512
ROC AUC: 0.9230
Accuracy: 0.8834
F1 Score: 0.8901
Sensitivity (Recall for Positive Class): 0.8762
Specificity (Recall for Negative Class): 0.8921
```

---

## Results

| Model | MCC | AUC | Accuracy |
|-------|-----|-----|----------|
| ChemBERTa (this work) | ~0.75 | ~0.92 | ~0.88 |

> Results will vary depending on dataset size, number of folds, and epochs. These are representative values.

---

## Project Structure

```
chembert-herg/
├── train_chembert_herg.py   # Main training & evaluation script
├── requirements.txt         # Python dependencies
├── README.md
└── data/
    └── herg/
        ├── herg_train.csv
        ├── herg_val.csv
        ├── herg_test.csv
        └── paper_valid_data.csv
```

---

## Dependencies

```
torch>=1.11.0
transformers>=4.18.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
```

Install all at once:

```bash
pip install torch transformers scikit-learn numpy pandas
```

> For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## LoRA Fine-Tuning (Parameter-Efficient)

The script includes a built-in **LoRA (Low-Rank Adaptation)** implementation that dramatically reduces the number of trainable parameters — useful when GPU memory is limited or you want faster fine-tuning.

### How it works

Instead of updating all model weights, LoRA injects low-rank trainable matrices `A` and `B` alongside each frozen Linear layer:

```
output = W·x  +  α · (A · B) · x
          ↑              ↑
     frozen original   trainable LoRA (rank r)
```

This reduces trainable parameters from ~125M (full ChemBERTa) to as few as ~300K depending on rank.

### Enabling LoRA

In `train_chembert_herg.py`, **uncomment** the LoRA block (currently at the bottom of the file):

```python
import math

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)
```

Then call it **after** loading the model and **before** training:

```python
model = RobertaForSequenceClassification.from_pretrained(
    "seyonec/PubChem10M_SMILES_BPE_396_250",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

# Inject LoRA into all Linear layers
replace_linear_with_lora(model, rank=16, alpha=16)
model = model.to(device)

# Verify parameter reduction
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")
```

The rest of the training loop remains **unchanged** — LoRA layers integrate transparently.

### LoRA Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rank` | 16 | Rank of the low-rank matrices A and B. Lower = fewer params. Try 4, 8, 16, 32. |
| `alpha` | 16 | Scaling factor for the LoRA update. Often set equal to `rank`. |

**Tuning tips:**
- `rank=4` → most aggressive compression, fastest training
- `rank=16` → good balance of capacity and efficiency (recommended starting point)
- `rank=32` → closer to full fine-tuning quality, more parameters
- Setting `alpha = rank` keeps the effective learning rate stable across different rank choices

### Parameter Count Comparison

| Mode | Approx. Trainable Params |
|------|--------------------------|
| Full fine-tuning | ~125,000,000 |
| LoRA (rank=4)   | ~75,000       |
| LoRA (rank=16)  | ~300,000      |
| LoRA (rank=32)  | ~600,000      |

> Exact counts depend on the number of Linear layers in the model. Run the `print` statement above to see the precise count for your configuration.

---



| Parameter | Value | Notes |
|-----------|-------|-------|
| Pretrained model | `seyonec/PubChem10M_SMILES_BPE_396_250` | ChemBERTa |
| Max sequence length | Dynamic (max in dataset) | Computed at runtime |
| Batch size | 4 | Increase with more VRAM |
| Learning rate | 2e-5 | AdamW |
| Epsilon | 1e-8 | AdamW |
| Warmup steps | 0 | Linear schedule |
| Folds | 5 | Stratified K-Fold |
| Epochs per fold | 10 | Adjust as needed |

---

## Citation

If you use this code, please cite the ChemBERTa paper:

```bibtex
@article{chithrananda2020chemberta,
  title={ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction},
  author={Chithrananda, Seyone and Grand, Gabriel and Ramsundar, Bharath},
  journal={arXiv preprint arXiv:2010.09885},
  year={2020}
}
```

---

## License

MIT License — see `LICENSE` for details.

---

*Built for computational drug discovery research. Contributions and issues welcome.*
