import numpy as np
import pandas as pd
import os
import json
import time
import threading
import psutil
import subprocess
import re
from pathlib import Path

from pynvml import *
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import argparse

BETA_ROOT = "."
TMP_DATA = Path(BETA_ROOT) / "tmp_beta_data"
TMP_DATA.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--target", default="label")
parser.add_argument("--seeds", default="0,1,2,3,4")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

dataset_name = args.dataset
model_name = "beta"

df = pd.read_csv(f"../dataset/{dataset_name}.csv")

target_col = args.target
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = X.values.astype(np.float32)
y = LabelEncoder().fit_transform(y.astype(str))

seeds = list(map(int, args.seeds.split(",")))
all_splits = []

for seed in seeds:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        all_splits.append((seed, fold, train_idx, test_idx))

class ResourceMonitor:
    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.process = psutil.Process(os.getpid())
        self.peak_rss_mb = 0.0
        self.peak_gpu_mb = 0.0
        self.stop_event = threading.Event()
        self.thread = None

    def run(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(self.gpu_index)

        while not self.stop_event.is_set():
            rss = self.process.memory_info().rss / (1024 ** 2)

            gpu_mem = 0.0
            try:
                procs = nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    if p.pid == self.process.pid:
                        gpu_mem += p.usedGpuMemory / (1024 ** 2)
            except:
                pass

            self.peak_rss_mb = max(self.peak_rss_mb, rss)
            self.peak_gpu_mb = max(self.peak_gpu_mb, gpu_mem)

            time.sleep(0.01)

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

ACC_RE = re.compile(r"\[Accuracy\]=([0-9.]+)")
AUC_RE = re.compile(r"\[AUC\]=([0-9.]+)")
F1_RE = re.compile(r"\[F1\]=([0-9.]+)")

def extract_last(pattern, text):
    m = pattern.findall(text)
    return float(m[-1]) if m else None

accs = []
aucs = []
f1s = []
train_times = []
infer_times = []
gpu_mem = []
cpu_mem = []
per_run_results = []

for i, (seed, fold, train_idx, test_idx) in enumerate(all_splits):
    print(f"Split {i+1}/{len(all_splits)}")

    X_trainval = X[train_idx]
    y_trainval = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=seed
    )

    name = f"{dataset_name}_s{seed}_f{fold}"
    folder = TMP_DATA / name
    folder.mkdir(exist_ok=True)

    np.save(folder / "N_train.npy", X_train)
    np.save(folder / "N_val.npy", X_val)
    np.save(folder / "N_test.npy", X_test)

    np.save(folder / "y_train.npy", y_train)
    np.save(folder / "y_val.npy", y_val)
    np.save(folder / "y_test.npy", y_test)

    info = {
        "task_type": "binclass" if len(np.unique(y)) == 2 else "multiclass",
        "n_num_features": X.shape[1],
        "n_cat_features": 0
    }

    with open(folder / "info.json", "w") as f:
        json.dump(info, f)

    log_file = folder / "log.txt"

    cmd = [
        "python", "main.py",
        "--model_type", "Beta",
        "--dataset", name,
        "--dataset_path", str(TMP_DATA.resolve()),
        "--seed_num", "1",
        "--gpu", str(args.gpu),
        "--batch_size", "128"
    ]

    monitor = ResourceMonitor(args.gpu)

    start = time.perf_counter()
    monitor.start()

    with open(log_file, "w") as f:
        subprocess.run(cmd, cwd=BETA_ROOT, stdout=f, stderr=f)

    monitor.stop()
    end = time.perf_counter()

    total_time = end - start
    text = open(log_file, "r").read()

    acc = extract_last(ACC_RE, text)
    auc = extract_last(AUC_RE, text)
    f1 = extract_last(F1_RE, text)

    if acc is None or auc is None or f1 is None:
        print("FAILED:", name)
        continue

    acc = acc * 100
    auc = auc * 100
    f1 = f1 * 100
    infer_time = total_time / len(y_test)

    accs.append(acc)
    aucs.append(auc)
    f1s.append(f1)
    train_times.append(total_time)
    infer_times.append(infer_time)
    gpu_mem.append(monitor.peak_gpu_mb)
    cpu_mem.append(monitor.peak_rss_mb)

    per_run_results.append(
        {
            "split_name": name,
            "seed": seed,
            "fold": fold,
            "accuracy": acc,
            "auc": auc,
            "f1": f1,
            "wall_clock_train_time_sec": total_time,
            "inference_latency_sec_per_sample": infer_time,
            "peak_gpu_mb": monitor.peak_gpu_mb,
            "peak_cpu_rss_mb": monitor.peak_rss_mb
        }
    )

Am = np.mean(accs)
Sm = np.std(accs)
auc_mean = np.mean(aucs)
auc_std = np.std(aucs)
f1_mean = np.mean(f1s)
f1_std = np.std(f1s)

Tm = np.sum(train_times)
Lm = np.sum(infer_times)
GpuMm = np.max(gpu_mem)
CpuMm = np.max(cpu_mem)

result = {
    "model": model_name,
    "dataset": dataset_name,
    "performance": {
        "accuracy_mean": Am,
        "accuracy_std": Sm,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std
    },
    "resource_usage": {
        "wall_clock_train_time_sec": Tm,
        "inference_latency_sec_per_sample": Lm,
        "peak_gpu_mb": GpuMm,
        "peak_cpu_rss_mb": CpuMm
    },
    "per_run": per_run_results
}

os.makedirs(f"results/{dataset_name}", exist_ok=True)

with open(f"results/{dataset_name}/{model_name}_results.json", "w") as f:
    json.dump(result, f, indent=4)

print("========", dataset_name, "==========")
print(f"Accuracy: {Am:.2f} ± {Sm:.2f}")
print(f"AUC: {auc_mean:.2f} ± {auc_std:.2f}")
print(f"F1: {f1_mean:.2f} ± {f1_std:.2f}")
print(f"Train Time: {Tm:.2f}s | Inference: {Lm:.6f}")
print(f"GPU: {GpuMm:.2f} MB | CPU: {CpuMm:.2f} MB")