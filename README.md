# HiLoTab
Official implementation and benchmark pipeline for **HiLoTab** on HDLSS (High-Dimensional Low Sample Size) tabular datasets.
## Running Strategy
We intentionally did not design a single script that runs all models across all datasets automatically.
Since HDLSS datasets are often extremely high-dimensional, some experiments can take a significant amount of time depending on the available computational resources (GPU/CPU memory, runtime, and number of seeds).
To provide better flexibility and control over experiments, the repository is designed so that users can run models individually for each dataset.
Before running an experiment, navigate to the corresponding model directory.
Example:
```bash
cd BETA
```
Then execute the desired command.
Similarly:
```bash
cd Wide
```
for TabPFN-Wide experiments.
## Repository Structure

```text
HiLoTab/
│
├── BETA/
├── Wide/
├── datasets/
├── CatBoost/
├── XGboost/
└── README.md
```

# Installation

Clone the repository:

```bash
git clone https://github.com/annonym8098/HiLoTab
cd HiLoTab
```

Create a conda environment (recommended):
```bash
conda create -n [ENV_NAME] python=3.11
conda activate [ENV_NAME]
```

Install dependencies:
```bash
pip install -r requirements.txt
```

# Datasets
Place datasets inside the `datasets/` directory
Example:

```text
datasets/
└── arcene.csv/
```

# Running Experiments
Before running a model, navigate to the corresponding model directory.
All experiments support multiple random seeds. Example seeds:
```text
0,1,2,3,4
```
```Some models requires to download the checkpoints```

# BETA Models

## Download Pretrained Weights

Download pretrained weights from:
https://github.com/penfever/TuneTables/tree/main/models
Place the downloaded files inside:
```text
BETA/model/models/models_diff
```
## Run
```bash
CUDA_VISIBLE_DEVICES=0 python run_beta.py --dataset [DATA_NAME] --seeds [SEEDS] --target [TARGET_NAME]
```

---

# TabPFN-Wide
## Download Model Weights
Download pretrained weights from:
https://github.com/pfeiferAI/TabPFN-Wide/tree/main/models
Place the downloaded files inside:
```text
Wide/models
```
## Run
```bash
CUDA_VISIBLE_DEVICES=0 python run_wide.py --dataset [DATA_NAME] --seeds [SEEDS] --target [TARGET_NAME]
```

---

# Gradient Boosting Models

Supported models:
- CatBoost
- XGBoost
- LightGBM

## File Names

| Model | File Name |
|---|---|
| CatBoost | `cat.py` |
| XGBoost | `xgb.py` |
| LightGBM | `lgbm.py` |

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python [FILE_NAME].py --n_trials [OPTUNA_TRIALS] --dataset [DATASET_NAME] --seeds [SEEDS] --target [TARGET]
```

## Example

```bash
CUDA_VISIBLE_DEVICES=0 python xgb.py --n_trials 100 --dataset arcene --seeds 0,1,2,3,4 --target label
```
---
# Machine Learning Models

Supported models:
- KNN
- Ridge
- Lasso
- SVM
- Random Forest

## File Names

| Model | File Name |
|---|---|
| KNN | `knn.py` |
| SVM | `svm.py` |
| Lasso | `lasso.py` |
| Ridge | `ridge.py` |
| Random Forest | `rf.py` |

## Run

```bash
python [FILE_NAME].py --n_trials [OPTUNA_TRIALS] --dataset [DATASET_NAME] --seeds [SEEDS] --target [TARGET]
```

## Example

```bash
python lasso.py --n_trials 100 --dataset arcene --seeds 0,1,2,3,4 --target label
```
---
# Deep Learning Models

Supported models:
- MLP
- TabICL
- TabM
- TANDEM
- RealMLP

## File Names

| Model | File Name |
|---|---|
| MLP | `mlp.py` |
| TabICL | `tabicl.py` |
| TabM | `tabm.py` |
| TANDEM | `tandem.py` |
| RealMLP | `real_mlp.py` |

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python [FILE_NAME].py --n_trials [OPTUNA_TRIALS] --dataset [DATASET_NAME] --seeds [SEEDS] --target [TARGET]
```

## Example

```bash
CUDA_VISIBLE_DEVICES=0 python tabm.py --n_trials 100 --dataset arcene --seeds 0,1,2,3,4 --target label
```

---
# Feature Selection Models

Supported models:
- STG
- LSPIN
- LLSPIN

## File Names

| Model | File Name |
|---|---|
| STG | `stg.py` |
| LSPIN | `lspin.py` |
| LLSPIN | `llspin.py` |

## Run

These models use hyperparameter tuning via `--n_trials`.

```bash
python [FILE_NAME].py --n_trials [OPTUNA_TRIALS] --dataset [DATASET_NAME] --seeds [SEEDS] --target [TARGET]
```

## Example

```bash
python lspin.py --n_trials 100 --dataset arcene --seeds 0,1,2,3,4 --target label
```
# ProtoGate

## Setup

Before running ProtoGate:

1. Navigate to the source directory:
```bash
cd src
```

2. Place your dataset inside:
```text
Protogate/data/
```

---

## Run

ProtoGate uses Optuna for hyperparameter tuning.

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
--dataset csv \
--train_path /path/to/arcene.csv \
--model protogate \
--full_batch_training \
--enable_optuna \
--optuna_trial 100
```

---

# TabDPT

## Installation

Install TabDPT via pip:

```bash
pip install tabdpt
```

---

## Required Patch (Important)

After installation, locate the `models.py` file inside the installed site-packages directory of TabDPT.

### Step 1: Find this line
```python
attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
```

### Step 2: Replace it with

```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_mem_efficient=False,
    enable_math=True
):
    attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
```

This ensures stable attention computation across different CUDA setups.

---

## Running TabDPT

Use the provided pipeline script:

```bash
CUDA_VISIBLE_DEVICES=0 python tabdpt_pipeline.py \
--dataset [DATA_NAME] \
--seeds [SEEDS] \
--target [TARGET_NAME]
```

---

## Example

```bash
CUDA_VISIBLE_DEVICES=0 python tabdpt_pipeline.py \
--dataset arcene \
--seeds 0,1,2,3,4 \
--target label
```
