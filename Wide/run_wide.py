import numpy as np
import pandas as pd
import os
import json
import time
import threading
import psutil
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetGraphicsRunningProcesses,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

import argparse
import random
import warnings
import logging
import torch

from tabpfn import TabPFNClassifier
from tabpfnwide.patches import fit
from tabpfn.model.loading import load_model_criterion_config

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_preprocessing.preprocessing import preprocess_fit_transform

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--target", default="label")
parser.add_argument("--seeds", default="0,1,2,3,4")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

dataset_name = args.dataset
model_name = "tabpfn_wide"

df = pd.read_csv(f'../dataset/{dataset_name}.csv')

target_col = args.target
X = df.drop(columns=[target_col]).values.astype(np.float32)
y = df[target_col].values

le = LabelEncoder()
y = le.fit_transform(y)

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

model, _, _ = load_model_criterion_config(
    model_path=None,
    check_bar_distribution_criterion=False,
    cache_trainset_representation=False,
    which='classifier',
    version='v2',
    download=True,
)

model.features_per_group = 1
checkpoint = torch.load("./models/TabPFN-Wide-8k_submission.pt", map_location=device)
model.load_state_dict(checkpoint)

setattr(TabPFNClassifier, 'fit', fit)

seeds = list(map(int, args.seeds.split(",")))

all_splits = []
for seed in seeds:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X, y):
        all_splits.append((train_idx, test_idx))


class ResourceMonitor:
    def __init__(self, gpu_index=0, interval=0.005):
        self.gpu_index = gpu_index
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_rss_mb = 0.0
        self.peak_gpu_mb = 0.0

    def _run(self):
        parent = psutil.Process(os.getpid())

        while not self.stop_event.is_set():
            rss_mb = self.process.memory_info().rss / (1024 ** 2)

            try:
                children = parent.children(recursive=True)
                pids = [parent.pid] + [c.pid for c in children]
            except:
                pids = [parent.pid]

            gpu_mb = 0.0

            try:
                compute_procs = nvmlDeviceGetComputeRunningProcesses(self.handle)
            except:
                compute_procs = []

            try:
                graphics_procs = nvmlDeviceGetGraphicsRunningProcesses(self.handle)
            except:
                graphics_procs = []

            for p in list(compute_procs) + list(graphics_procs):
                if p.pid in pids:
                    gpu_mb += p.usedGpuMemory / (1024 ** 2)

            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            self.peak_gpu_mb = max(self.peak_gpu_mb, gpu_mb)

            time.sleep(self.interval)

    def start(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


final_scores, auc_scores, f1_scores = [], [], []
train_times, infer_times = [], []
peak_gpu_memories, peak_cpu_rss_memories = [], []

for i, (train_idx, test_idx) in enumerate(all_splits):
    print(f"Split {i+1}/{len(all_splits)}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf = TabPFNClassifier(device=device, n_estimators=1, ignore_pretraining_limits=True)

    monitor = ResourceMonitor(gpu_index=args.gpu)

    start_train = time.perf_counter()
    monitor.start()
    clf.fit(X_train, y_train, model=model)
    end_train = time.perf_counter()
    monitor.stop()

    peak_cpu_rss_memories.append(monitor.peak_rss_mb)
    peak_gpu_memories.append(monitor.peak_gpu_mb)

    start_infer = time.perf_counter()
    preds = clf.predict(X_test)
    end_infer = time.perf_counter()

    probs = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, preds) * 100

    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y_test, probs[:, 1]) * 100
        f1 = f1_score(y_test, preds) * 100
    else:
        auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro") * 100
        f1 = f1_score(y_test, preds, average="macro") * 100

    final_scores.append(acc)
    auc_scores.append(auc)
    f1_scores.append(f1)

    train_times.append(end_train - start_train)
    infer_times.append((end_infer - start_infer) / len(test_idx))


Am, Sm = np.mean(final_scores), np.std(final_scores)
auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)

Tm = np.sum(train_times)
Lm = np.sum(infer_times)
GpuMm = np.max(peak_gpu_memories)
CpuMm = np.max(peak_cpu_rss_memories)

os.makedirs(f"results/{dataset_name}", exist_ok=True)

with open(f"results/{dataset_name}/{model_name}_results.json", "w") as f:
    json.dump(
        {
            "model": model_name,
            "dataset": dataset_name,
            "performance": {
                "accuracy_mean": Am,
                "accuracy_std": Sm,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
            },
            "resource_usage": {
                "wall_clock_train_time_sec": Tm,
                "inference_latency_sec_per_sample": Lm,
                "peak_gpu_mb": GpuMm,
                "peak_cpu_rss_mb": CpuMm,
            },
        },
        f,
        indent=4,
    )

print("========", dataset_name, "==========")
print(f"Accuracy: {Am:.2f} ± {Sm:.2f}")
print(f"AUC: {auc_mean:.2f} ± {auc_std:.2f}")
print(f"F1: {f1_mean:.2f} ± {f1_std:.2f}")
print(f"Train Time: {Tm:.2f}s | Inference: {Lm:.6f}")
print(f"GPU: {GpuMm:.2f} MB | CPU: {CpuMm:.2f} MB")