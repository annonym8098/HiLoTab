import numpy as np
import pandas as pd
import os
import json
import time
import threading
import psutil
import argparse
import random
import warnings
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetComputeRunningProcesses, nvmlDeviceGetGraphicsRunningProcesses
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import optuna

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_preprocessing.preprocessing import preprocess_fit_transform

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger().setLevel(logging.CRITICAL)


class LSPINClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        pred_hidden_dims,
        gate_hidden_dims,
        a=1.0,
        sigma=0.5,
        lam=0.5,
        gamma1=0.0,
        gamma2=0.0,
        activation_pred="relu",
        activation_gate="relu",
        use_batchnorm=True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = a
        self.sigma = sigma
        self.lam = lam
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.use_batchnorm = use_batchnorm

        self.act_pred = self._get_activation(activation_pred)
        self.act_gate = self._get_activation(activation_gate)

        layers = []
        prev_dim = input_dim

        for h in gate_hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(self._get_activation(activation_gate))
            prev_dim = h

        self.gate_net = nn.Sequential(*layers)
        self.alpha_layer = nn.Linear(prev_dim, input_dim)

        pred_layers = []
        prev_dim = input_dim

        for h in pred_hidden_dims:
            pred_layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                pred_layers.append(nn.BatchNorm1d(h))
            pred_layers.append(self._get_activation(activation_pred))
            prev_dim = h

        self.pred_body = nn.Sequential(*pred_layers)
        self.pred_head = nn.Linear(prev_dim, output_dim)

    def _get_activation(self, name):
        return {
            "relu": nn.ReLU(),
            "l_relu": nn.LeakyReLU(0.2),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "none": nn.Identity()
        }[name]

    def forward(self, x, z=None, train_gates=True, compute_sim=False):
        alpha = self.alpha_layer(self.gate_net(x))

        if train_gates:
            noise = torch.randn_like(alpha)
            z_sample = alpha + self.sigma * noise
            gate = torch.clamp(self.a * z_sample + 0.5, 0, 1)
        else:
            gate = torch.clamp(self.a * alpha + 0.5, 0, 1)

        x_gated = x * gate
        features = self.pred_body(x_gated)
        logits = self.pred_head(features)

        return logits, gate, alpha

    def compute_loss(self, logits, labels, gate, alpha, z=None, compute_sim=False):
        task_loss = F.cross_entropy(logits, labels)

        reg = 0.5 - 0.5 * torch.erf((-1 / (2 * self.a) - alpha) / (self.sigma * (2 ** 0.5)))
        reg_gates = self.lam * reg.mean()

        if compute_sim and z is not None:
            K = 1.0 - self._squared_dist(z) / 2.0
            D = self._squared_dist(gate)
            D_neg = -D
            sim_loss = self.gamma1 * (K * D).mean() + self.gamma2 * ((1 - K) * D_neg).mean()
        else:
            sim_loss = 0.0

        total_loss = task_loss + reg_gates + sim_loss

        return total_loss, task_loss.item(), reg_gates.item(), sim_loss if isinstance(sim_loss, float) else sim_loss.item()

    def _squared_dist(self, x):
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        dist = x_norm + x_norm.T - 2.0 * torch.matmul(x, x.T)
        return dist

    def get_prob_alpha(self, x):
        with torch.no_grad():
            alpha = self.alpha_layer(self.gate_net(x))
            return torch.clamp(self.a * alpha + 0.5, 0, 1)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResourceMonitor:
    def __init__(self, gpu_index=0, interval=0.005):
        self.gpu_index = gpu_index
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_rss_mb = 0.0
        self.peak_gpu_mb = 0.0
        self.handle = None

    def _run(self):
        parent = psutil.Process(os.getpid())

        while not self.stop_event.is_set():
            try:
                rss_mb = self.process.memory_info().rss / (1024 ** 2)
            except:
                rss_mb = 0.0

            try:
                children = parent.children(recursive=True)
                pids = [parent.pid] + [c.pid for c in children]
            except:
                pids = [parent.pid]

            gpu_mb = 0.0

            if self.handle is not None:
                try:
                    compute_procs = nvmlDeviceGetComputeRunningProcesses(self.handle)
                except:
                    compute_procs = []

                try:
                    graphics_procs = nvmlDeviceGetGraphicsRunningProcesses(self.handle)
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
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        except:
            self.handle = None

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


def build_model(input_dim, output_dim, params):
    return LSPINClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
        pred_hidden_dims=params["pred_hidden_dims"],
        gate_hidden_dims=params["gate_hidden_dims"],
        a=params["a"],
        sigma=params["sigma"],
        lam=params["lam"],
        gamma1=params["gamma1"],
        gamma2=params["gamma2"],
        activation_pred=params["activation_pred"],
        activation_gate=params["activation_gate"],
        use_batchnorm=params["use_batchnorm"]
    )


def train_model(model, X_train, y_train, params, device, seed):
    set_seed(seed)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=params["batch_size"],
        shuffle=True,
        generator=generator,
        drop_last=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    model.train()

    for epoch in range(params["epochs"]):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, gate, alpha = model(xb, train_gates=True)
            loss, _, _, _ = model.compute_loss(logits, yb, gate, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def predict_model(model, X_test, device, batch_size=1024):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size, shuffle=False)

    all_logits = []

    model.eval()

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits, _, _ = model(xb, train_gates=False)
            all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    proba = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(proba, axis=1)

    return preds, proba


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--target", default="label")
parser.add_argument("--seeds", default="0,1,2,3,4")
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

SEED = 42
set_seed(SEED)

dataset_name = args.dataset
model_name = "lspin"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(f"../dataset/{dataset_name}.csv")

target_col = args.target
X = df.drop(columns=[target_col]).values
y = df[target_col].values

le = LabelEncoder()
y = le.fit_transform(y)

num_classes = len(np.unique(y))

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
        "pred_hidden_dims": [
            trial.suggest_categorical("pred_h1", [64, 128, 200, 256, 512]),
            trial.suggest_categorical("pred_h2", [64, 128, 200, 256, 512])
        ],
        "gate_hidden_dims": [
            trial.suggest_categorical("gate_h1", [32, 64, 100, 128, 256]),
            trial.suggest_categorical("gate_h2", [32, 64, 100, 128, 256])
        ],
        "a": trial.suggest_float("a", 0.5, 2.0),
        "sigma": trial.suggest_float("sigma", 0.1, 1.0),
        "lam": trial.suggest_float("lam", 1e-4, 1.0, log=True),
        "gamma1": trial.suggest_float("gamma1", 0.0, 1.0),
        "gamma2": trial.suggest_float("gamma2", 0.0, 1.0),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "epochs": trial.suggest_int("epochs", 10, 100),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "activation_pred": trial.suggest_categorical("activation_pred", ["relu", "l_relu", "tanh"]),
        "activation_gate": trial.suggest_categorical("activation_gate", ["relu", "l_relu", "tanh"]),
        "use_batchnorm": trial.suggest_categorical("use_batchnorm", [True, False])
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
            random_state=SEED
        )

        X_train, X_val = preprocess_fit_transform(X_train, X_val)

        set_seed(SEED + trial.number)

        model = build_model(
            input_dim=X_train.shape[1],
            output_dim=num_classes,
            params=params
        ).to(device)

        model = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            params=params,
            device=device,
            seed=SEED + trial.number
        )

        preds, _ = predict_model(model, X_val, device)
        score = accuracy_score(y_val, preds)
        scores.append(score)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.mean(scores)


t1 = time.perf_counter()

sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

t2 = time.perf_counter()

print("Tuning time ....................... (seconds)", t2 - t1)

best_params = {
    "pred_hidden_dims": [
        study.best_params["pred_h1"],
        study.best_params["pred_h2"]
    ],
    "gate_hidden_dims": [
        study.best_params["gate_h1"],
        study.best_params["gate_h2"]
    ],
    "a": study.best_params["a"],
    "sigma": study.best_params["sigma"],
    "lam": study.best_params["lam"],
    "gamma1": study.best_params["gamma1"],
    "gamma2": study.best_params["gamma2"],
    "lr": study.best_params["lr"],
    "weight_decay": study.best_params["weight_decay"],
    "epochs": study.best_params["epochs"],
    "batch_size": study.best_params["batch_size"],
    "activation_pred": study.best_params["activation_pred"],
    "activation_gate": study.best_params["activation_gate"],
    "use_batchnorm": study.best_params["use_batchnorm"]
}

os.makedirs(f"params/{dataset_name}", exist_ok=True)

with open(f"params/{dataset_name}/{model_name}_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

final_scores = []
auc_scores = []
f1_scores = []

train_times = []
infer_times = []
peak_gpu_memories = []
peak_cpu_rss_memories = []

for i, (train_idx, test_idx) in enumerate(all_splits):
    print(f"Final Split {i + 1}/{len(all_splits)}")

    split_seed = SEED + i
    set_seed(split_seed)

    cache_file = f"{cache_dir}/split_{i}.npz"

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        X_train = data["X_train"]
        X_test = data["X_test"]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
        X_train, X_test = preprocess_fit_transform(X_train, X_test)
        np.savez(cache_file, X_train=X_train, X_test=X_test)

    y_train = y[train_idx]
    y_test = y[test_idx]

    model = build_model(
        input_dim=X_train.shape[1],
        output_dim=num_classes,
        params=best_params
    ).to(device)

    monitor = ResourceMonitor(gpu_index=args.gpu)

    start_train = time.perf_counter()
    monitor.start()

    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        params=best_params,
        device=device,
        seed=split_seed
    )

    end_train = time.perf_counter()
    monitor.stop()

    peak_cpu_rss_mb = monitor.peak_rss_mb
    peak_gpu_mb = monitor.peak_gpu_mb

    start_infer = time.perf_counter()
    preds, proba = predict_model(model, X_test, device)
    end_infer = time.perf_counter()

    acc = accuracy_score(y_test, preds) * 100

    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y_test, proba[:, 1]) * 100
        f1 = f1_score(y_test, preds) * 100
    else:
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro") * 100
        f1 = f1_score(y_test, preds, average="macro") * 100

    final_scores.append(acc)
    auc_scores.append(auc)
    f1_scores.append(f1)

    train_times.append(end_train - start_train)
    infer_times.append((end_infer - start_infer) / len(test_idx))
    peak_gpu_memories.append(peak_gpu_mb)
    peak_cpu_rss_memories.append(peak_cpu_rss_mb)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

with open(f"results/{dataset_name}/{model_name}_results.json", "w") as f:
    json.dump(
        {
            "model": model_name,
            "dataset": dataset_name,
            "best_params": best_params,
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
            "per_run": {
                "accuracy": final_scores,
                "auc": auc_scores,
                "f1": f1_scores,
                "train_time": train_times,
                "infer_time": infer_times,
                "peak_gpu_mb": peak_gpu_memories,
                "peak_cpu_rss_mb": peak_cpu_rss_memories
            },
            "meta": {
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_classes": int(num_classes),
                "seeds": seeds,
                "n_splits_total": len(all_splits),
                "tuning_time_sec": t2 - t1
            }
        },
        f,
        indent=4
    )

print("========", dataset_name, "==========")
print(f"Accuracy: {Am:.2f} ± {Sm:.2f}")
print(f"AUC: {auc_mean:.2f} ± {auc_std:.2f}")
print(f"F1: {f1_mean:.2f} ± {f1_std:.2f}")
print(f"Total Train Time: {Tm:.2f} seconds")
print(f"Total Inference Time: {Lm:.6f}")
print(f"Peak GPU Memory: {GpuMm:.2f} MB")
print(f"Peak CPU RSS: {CpuMm:.2f} MB")