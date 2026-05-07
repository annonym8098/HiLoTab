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
└── arcene/
```

# Running Experiments
Before running a model, navigate to the corresponding model directory.
All experiments support multiple random seeds. Example seeds:
```text
0,1,2,3,4
```
```Some models requires you to download the checkpoints```

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

