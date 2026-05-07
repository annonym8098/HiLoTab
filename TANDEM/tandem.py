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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import optuna
import argparse
import random
import warnings
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except:
    pass

dataset_name = args.dataset
model_name = "tandem"

df = pd.read_csv(f"../dataset/{dataset_name}.csv")

target_col = args.target
X = df.drop(columns=[target_col]).values
y = df[target_col].values
le = LabelEncoder()
y = le.fit_transform(y)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

seeds = list(map(int, args.seeds.split(",")))

all_splits = []
for seed in seeds:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X, y):
        all_splits.append((train_idx, test_idx))

cache_dir = f"cache/{dataset_name}"
os.makedirs(cache_dir, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_fit_transform(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train.astype(np.float32), X_test.astype(np.float32)


class GatingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def forward(self, x):
        if self.training:
            noise = torch.normal(0, self.sigma, size=x.size(), device=x.device)
        else:
            noise = 0
        mu = self.net(x)
        z = mu + 0.1 * noise * float(self.training)
        gates = torch.clamp(z + 0.5, 0.0, 1.0)
        return mu, x * gates, gates


class OSDT(nn.Module):
    def __init__(self, input_dim, depth, gating_net=None):
        super().__init__()
        self.depth = depth
        self.gating_net = gating_net
        self.split_weights = nn.Parameter(torch.randn(depth, input_dim))
        self.feature_thresholds = nn.Parameter(torch.randn(depth))
        self.log_temperatures = nn.Parameter(torch.zeros(depth))
        indices = torch.arange(2 ** depth)
        bits = ((indices[:, None] & (1 << torch.arange(depth))) > 0).float()
        self.register_buffer("bin_codes", bits.t())

    def forward(self, x):
        selected = []
        for i in range(self.depth):
            x_g = self.gating_net(x)[1] if self.gating_net is not None else x
            s = torch.einsum("bi,i->b", x_g, self.split_weights[i])
            selected.append(s)
        selected = torch.stack(selected, dim=1)
        logits = (selected - self.feature_thresholds) * torch.exp(-self.log_temperatures)
        logits = torch.stack([-logits, logits], dim=-1)
        bins = torch.sigmoid(logits)
        left, right = bins[..., 0], bins[..., 1]
        codes = self.bin_codes.unsqueeze(0)
        match = left.unsqueeze(-1) * (1 - codes) + right.unsqueeze(-1) * codes
        return match.prod(dim=-2)


class OSDTEncoder(nn.Module):
    def __init__(self, input_dim, num_trees, depth, gating_net=None):
        super().__init__()
        self.trees = nn.ModuleList(
            [OSDT(input_dim, depth, gating_net) for _ in range(num_trees)]
        )

    def forward(self, x):
        return torch.stack([t(x) for t in self.trees], dim=0).mean(dim=0)


class ModularEncoder(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU()])
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ModularDecoder(nn.Module):
    def __init__(self, output_dim, hidden):
        super().__init__()
        rev_hidden = list(reversed(hidden))
        layers = []
        prev = rev_hidden[0]
        for h in rev_hidden[1:]:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Tandem(nn.Module):
    def __init__(self, input_dim, hidden, depth, trees, gating):
        super().__init__()
        self.nn_enc = ModularEncoder(input_dim, hidden)
        self.osdt_enc = OSDTEncoder(input_dim, trees, depth, gating)
        self.dec = ModularDecoder(input_dim, hidden)

    def forward(self, x):
        z1 = self.nn_enc(x)
        z2 = self.osdt_enc(x)
        return self.dec(z1), self.dec(z2), z1, z2


class TandemModel(nn.Module):
    def __init__(self, input_dim, n_classes, depth=7, trees=1, gate_hidden=128, sigma=0.5):
        super().__init__()
        last = 2 ** depth
        hidden = [last * 2, last]
        gating = GatingNet(input_dim, hidden_dim=gate_hidden, sigma=sigma)
        self.tandem = Tandem(input_dim, hidden, depth, trees, gating)
        self.clf = nn.Linear(last, n_classes)

    def forward(self, x):
        r1, r2, z1, z2 = self.tandem(x)
        return self.clf(z1), r1, r2, z1, z2


class TandemClassifier:
    def __init__(
        self,
        lr=1e-3,
        epochs=100,
        batch=64,
        seed=42,
        depth=7,
        trees=1,
        gate_hidden=128,
        sigma=0.5,
        weight_decay=0.0,
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.seed = seed
        self.depth = depth
        self.trees = trees
        self.gate_hidden = gate_hidden
        self.sigma = sigma
        self.weight_decay = weight_decay
        self.model = None
        self.classes_ = None

    def fit(self, X, y, Xv, yv):
        set_seed(self.seed)

        X = torch.tensor(X, dtype=torch.float32)
        Xv = torch.tensor(Xv, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        yv = torch.tensor(yv, dtype=torch.long)

        self.classes_ = np.unique(y.cpu().numpy())
        y_map = {c: i for i, c in enumerate(self.classes_)}
        y = torch.tensor([y_map[int(v.item())] for v in y], dtype=torch.long)
        yv = torch.tensor([y_map[int(v.item())] for v in yv], dtype=torch.long)

        self.model = TandemModel(
            input_dim=X.shape[1],
            n_classes=len(self.classes_),
            depth=self.depth,
            trees=self.trees,
            gate_hidden=self.gate_hidden,
            sigma=self.sigma,
        ).to(device)

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        dl = DataLoader(
            TensorDataset(X, y),
            batch_size=self.batch,
            shuffle=True,
            drop_last=True,
        )
        vl = DataLoader(
            TensorDataset(Xv, yv),
            batch_size=self.batch,
            shuffle=False,
            drop_last=True,
        )

        best = float("inf")
        best_w = None

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad()
                logit, r1, r2, z1, z2 = self.model(xb)
                loss = (
                    loss_fn(logit, yb)
                    + F.mse_loss(r1, xb)
                    + F.mse_loss(r2, xb)
                    + 0.1 * F.mse_loss(z1, z2)
                )
                loss.backward()
                opt.step()

            self.model.eval()
            vloss = 0.0
            with torch.no_grad():
                for xb, yb in vl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logit, r1, r2, z1, z2 = self.model(xb)
                    loss = (
                        loss_fn(logit, yb)
                        + F.mse_loss(r1, xb)
                        + F.mse_loss(r2, xb)
                        + 0.1 * F.mse_loss(z1, z2)
                    )
                    vloss += loss.item()

            if vloss < best:
                best = vloss
                best_w = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

        self.model.load_state_dict(best_w)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch):
                xb = X[start:start + self.batch]
                logit, _, _, _, _ = self.model(xb)
                pred_idx = torch.argmax(logit, dim=1).cpu().numpy()
                preds.append(pred_idx)
        preds = np.concatenate(preds, axis=0)
        return self.classes_[preds]

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch):
                xb = X[start:start + self.batch]
                logit, _, _, _, _ = self.model(xb)
                prob = torch.softmax(logit, dim=1).cpu().numpy()
                probs.append(prob)
        return np.concatenate(probs, axis=0)


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
        if torch.cuda.is_available():
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 50, 300),
        "batch": trial.suggest_categorical("batch", [16, 32, 64, 128]),
        "depth": trial.suggest_int("depth", 4, 8),
        "trees": trial.suggest_int("trees", 1, 3),
        "gate_hidden": trial.suggest_categorical("gate_hidden", [64, 128, 256]),
        "sigma": trial.suggest_float("sigma", 0.1, 1.0),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
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

        model = TandemClassifier(**params)
        model.fit(X_train, y_train, X_val, y_val)

        preds = model.predict(X_val)
        score = accuracy_score(y_val, preds)
        scores.append(score)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.mean(scores)


t1 = time.perf_counter()

sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

t2 = time.perf_counter()

print("Tuning time ....................... (seconds)", t2 - t1)

best_params = study.best_params
best_params["seed"] = SEED

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
    print(f"Final Split {i+1}/{len(all_splits)}")

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

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=SEED,
    )

    model = TandemClassifier(**best_params)

    monitor = ResourceMonitor(gpu_index=args.gpu)

    start_train = time.perf_counter()
    monitor.start()
    model.fit(X_tr, y_tr, X_val, y_val)
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

    if device.type == "cuda":
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

print("========", dataset_name, "==========")
print(f"Accuracy: {Am:.2f} ± {Sm:.2f}")
print(f"AUC: {auc_mean:.2f} ± {auc_std:.2f}")
print(f"F1: {f1_mean:.2f} ± {f1_std:.2f}")
print(f"Total Train Time: {Tm:.2f} seconds")
print(f"Total Inference Time: {Lm:.6f}")
print(f"Peak GPU Memory: {GpuMm:.2f} MB")
print(f"Peak CPU RSS: {CpuMm:.2f} MB")