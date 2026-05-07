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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import optuna
from xgboost import XGBClassifier
import argparse
import random
import warnings
import logging
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_preprocessing.preprocessing import preprocess_fit_transform

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger().setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--target", default="label")
parser.add_argument("--seeds", default="0,1,2,3,4")
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

dataset_name = args.dataset
model_name = "xgboost"

df = pd.read_csv(f'../dataset/{dataset_name}.csv')

target_col = args.target
X = df.drop(columns=[target_col]).values
y = df[target_col].values
le = LabelEncoder()
y = le.fit_transform(y)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

seeds = list(map(int, args.seeds.split(",")))

all_splits = []
for seed in seeds:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X, y):
        all_splits.append((train_idx, test_idx))

cache_dir = f"cache/{dataset_name}"
os.makedirs(cache_dir, exist_ok=True)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.1, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),
        "tree_method": "hist",
        "device": "cuda",
        "random_state": SEED,
        "seed": SEED,
    }

    scores = []

    for train_idx, _ in all_splits[:1]:
        X_train_full = X[train_idx]
        y_train_full = y[train_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
            stratify=y_train_full,
            random_state=SEED,
        )

        X_train, X_val = preprocess_fit_transform(X_train, X_val)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        score = accuracy_score(y_val, preds)
        scores.append(score)

    return np.mean(scores)


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
                compute_procs = nvmlDeviceGetComputeRunningProcesses(
                    self.handle
                )
            except:
                compute_procs = []

            try:
                graphics_procs = nvmlDeviceGetGraphicsRunningProcesses(
                    self.handle
                )
            except:
                graphics_procs = []

            all_procs = list(compute_procs) + list(graphics_procs)

            for p in all_procs:
                if p.pid in pids:
                    gpu_mb += p.usedGpuMemory / (1024 ** 2)

            if rss_mb > self.peak_rss_mb:
                self.peak_rss_mb = rss_mb
            if gpu_mb > self.peak_gpu_mb:
                self.peak_gpu_mb = gpu_mb

            time.sleep(self.interval)

    def start(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, daemon=True
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


t1 = time.perf_counter()

sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

t2 = time.perf_counter()

print("Tuning time ....................... (seconds)", t2 - t1)

best_params = study.best_params
best_params["tree_method"] = "hist"
best_params["device"] = "cuda"
best_params["random_state"] = SEED
best_params["seed"] = SEED

os.makedirs(f"params/{dataset_name}", exist_ok=True)

with open(
    f"params/{dataset_name}/{model_name}_params.json", "w"
) as f:
    json.dump(best_params, f, indent=4)

final_scores = []
auc_scores = []
f1_scores = []

train_times = []
infer_times = []
peak_gpu_memories = []
peak_cpu_rss_memories = []

for i, (train_idx, test_idx) in enumerate(all_splits):
    print(f"Final Split {i+1}/{len(all_splits)}")

    cache_file = f"{cache_dir}/split_{i}.npz"

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        X_train = data["X_train"]
        X_test = data["X_test"]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
        X_train, X_test = preprocess_fit_transform(
            X_train, X_test
        )
        np.savez(cache_file, X_train=X_train, X_test=X_test)

    y_train = y[train_idx]
    y_test = y[test_idx]

    model = XGBClassifier(**best_params)

    monitor = ResourceMonitor(gpu_index=args.gpu)

    start_train = time.perf_counter()
    monitor.start()
    model.fit(X_train, y_train)
    end_train = time.perf_counter()
    monitor.stop()

    peak_cpu_rss_mb = monitor.peak_rss_mb
    peak_gpu_mb = monitor.peak_gpu_mb

    start_infer = time.perf_counter()
    preds = model.predict(X_test)
    end_infer = time.perf_counter()

    proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds) * 100

    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y_test, proba[:, 1]) * 100
        f1 = f1_score(y_test, preds) * 100
    else:
        auc = roc_auc_score(
            y_test, proba, multi_class="ovr", average="macro"
        ) * 100
        f1 = f1_score(
            y_test, preds, average="macro"
        ) * 100

    final_scores.append(acc)
    auc_scores.append(auc)
    f1_scores.append(f1)

    train_times.append(end_train - start_train)
    infer_times.append((end_infer - start_infer) / len(test_idx))
    peak_gpu_memories.append(peak_gpu_mb)
    peak_cpu_rss_memories.append(peak_cpu_rss_mb)

Am = np.mean(final_scores)
Sm = np.std(final_scores)

Tm = np.sum(train_times)
Lm = np.sum(infer_times)
GpuMm = np.max(peak_gpu_memories)
CpuMm = np.max(peak_cpu_rss_memories)

auc_mean = np.mean(auc_scores)
auc_std = np.std(auc_scores)
f1_mean = np.mean(f1_scores)
f1_std = np.std(f1_scores)

os.makedirs(f"results/{dataset_name}", exist_ok=True)

with open(
    f"results/{dataset_name}/{model_name}_results.json", "w"
) as f:
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
            "per_run": {
                "accuracy": final_scores,
                "auc": auc_scores,
                "f1": f1_scores,
                "train_time": train_times,
                "infer_time": infer_times,
                "peak_gpu_mb": peak_gpu_memories,
                "peak_cpu_rss_mb": peak_cpu_rss_memories,
            },
            "meta": {
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "seeds": seeds,
            },
        },
        f,
        indent=4,
    )

print("========" , dataset_name, "==========")
print(f"Accuracy: {Am:.2f} ± {Sm:.2f}")
print(f"AUC: {auc_mean:.2f} ± {auc_std:.2f}")
print(f"F1: {f1_mean:.2f} ± {f1_std:.2f}")
print(f"Total Train Time: {Tm:.2f} seconds")
print(f"Total Inference Time: {Lm:.6f}")
print(f"Peak GPU Memory: {GpuMm:.2f} MB")
print(f"Peak CPU RSS: {CpuMm:.2f} MB")