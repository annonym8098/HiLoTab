import os
import sys
import json
import time
import numpy as np
import optuna
import threading
import psutil

from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetGraphicsRunningProcesses,
)

import argparse

sys.path.append(os.path.abspath("src"))

from run_experiment import run_experiment, parse_arguments



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--train_path", type=str, default=None)
parser.add_argument("--valid_percentage", type=float, default=0.2)

args_cli = parser.parse_args()
if args_cli.dataset == "csv":
    dataset_name = os.path.splitext(os.path.basename(args_cli.train_path))[0]
else:
    dataset_name = args_cli.dataset
train_path = args_cli.train_path
model_name = "protogate"
seeds = [0,1,2,3,4]


# ======================
# RESOURCE MONITOR
# ======================
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


# ======================
# ARG BUILDER
# ======================
def get_args(dataset, seed, fold, params):
    args = parse_arguments([
        "--dataset", "csv",
        "--train_path", train_path,
        "--model", "protogate",
        "--repeat_id", str(seed),
        "--test_split", str(fold),
        "--cv_folds", "5",
        "--metric_model_selection", "balanced_accuracy",
        "--lr", str(params["lr"]),
        "--protogate_lam_global", str(params["lam_global"]),
        "--protogate_lam_local", str(params["lam_local"]),
        "--pred_k", str(params["pred_k"]),
        "--max_steps", str(params["max_steps"]),
        "--protogate_gating_hidden_layer_list", "200",
        "--disable_wandb",
        "--valid_percentage", str(args_cli.valid_percentage),
    ])
    return args


# ======================
# METRIC EXTRACTION
# ======================
def extract_metric(result):
    if isinstance(result, list):
        result = result[0]

    acc = None
    auc = None
    f1 = None

    for k, v in result.items():
        key = k.lower()

        # 🔥 only use bestmodel test metrics
        if "bestmodel_test" not in key:
            continue

        # ✅ plain accuracy (YOU added this earlier)
        if "accuracy" in key and "balanced" not in key:
            acc = float(v)

        # ✅ AUROC
        elif "auroc" in key:
            auc = float(v)

        # ✅ F1 (your macro but named weighted)
        elif "f1_weighted" in key:
            f1 = float(v)

    return acc, auc, f1


def extract_accuracy(result):
    acc, _, _ = extract_metric(result)
    return acc if acc is not None else 0.0


# ======================
# OPTUNA TUNING
# ======================
def objective(trial):

    params = {
        "lr": trial.suggest_float("lr", 1e-3, 0.3, log=True),
        "lam_global": trial.suggest_float("lam_global", 1e-5, 1e-2, log=True),
        "lam_local": trial.suggest_float("lam_local", 1e-5, 1e-2, log=True),
        "pred_k": trial.suggest_int("pred_k", 1, 10),
        "max_steps": trial.suggest_int("max_steps", 500, 8000),
    }
    
    args = get_args(dataset_name, seed=0, fold=0, params=params)

    output = run_experiment(args)

    return extract_accuracy(output["test_result"])


print("=== TUNING START ===")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params

print("Best Params:", best_params)


# ======================
# FULL EVALUATION
# ======================
final_scores = []
auc_scores = []
f1_scores = []

train_times = []
infer_times = []
peak_gpu_memories = []
peak_cpu_rss_memories = []

for seed in seeds:
    for fold in range(5):

        print(f"Seed {seed}, Fold {fold}")

        args = get_args(dataset_name, seed, fold, best_params)

        monitor = ResourceMonitor(gpu_index=0)

        start = time.perf_counter()
        monitor.start()

        output = run_experiment(args)

        monitor.stop()
        end = time.perf_counter()

        acc, auc, f1 = extract_metric(output["test_result"])

        final_scores.append(acc)
        auc_scores.append(auc)
        f1_scores.append(f1)

        train_times.append(end - start)
        infer_times.append(output["infer_time"] / output["test_size"])
        peak_gpu_memories.append(monitor.peak_gpu_mb)
        peak_cpu_rss_memories.append(monitor.peak_rss_mb)


# ======================
# AGGREGATION
# ======================
Am = float(np.mean(final_scores))
Sm = float(np.std(final_scores))

auc_mean = float(np.mean(auc_scores))
auc_std = float(np.std(auc_scores))

f1_mean = float(np.mean(f1_scores))
f1_std = float(np.std(f1_scores))

Tm = float(np.sum(train_times))
Lm = float(np.sum(infer_times))
GpuMm = float(np.max(peak_gpu_memories))
CpuMm = float(np.max(peak_cpu_rss_memories))


# ======================
# SAVE PARAMS
# ======================
os.makedirs(f"params/{dataset_name}", exist_ok=True)

with open(f"params/{dataset_name}/{model_name}_params.json", "w") as f:
    json.dump(best_params, f, indent=4, default=float)


# ======================
# SAVE RESULTS
# ======================
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
                "seeds": seeds,
                "best_params": best_params
            },
        },
        f,
        indent=4,
        default=float
    )

print("=== DONE ===")



# import os
# import sys
# import json
# import time
# import numpy as np
# import optuna

# sys.path.append(os.path.abspath("src"))

# from run_experiment import run_experiment, parse_arguments


# dataset_name = "metabric-pam50__200"
# model_name = "protogate"
# seeds = [0,1,2,3,4]


# # ======================
# # ARG BUILDER
# # ======================
# def get_args(dataset, seed, fold, params):
#     args = parse_arguments([
#         "--dataset", dataset,
#         "--model", "protogate",
#         "--repeat_id", str(seed),
#         "--test_split", str(fold),
#         "--cv_folds", "5",
#         "--metric_model_selection", "balanced_accuracy",
#         "--lr", str(params["lr"]),
#         "--protogate_lam_global", str(params["lam_global"]),
#         "--protogate_lam_local", str(params["lam_local"]),
#         "--pred_k", str(params["pred_k"]),
#         "--max_steps", str(params["max_steps"]),
#         "--protogate_gating_hidden_layer_list", "200",
#         "--disable_wandb"
#     ])
#     return args


# # ======================
# # METRIC EXTRACTION
# # ======================
# def extract_metric(result):
#     if isinstance(result, list):
#         result = result[0]

#     acc = None
#     auc = None
#     f1 = None

#     for k, v in result.items():
#         key = k.lower()

#         if "balanced_accuracy" in key:
#             acc = float(v)
#         elif "auc" in key or "auroc" in key:
#             auc = float(v)
#         elif "f1" in key:
#             f1 = float(v)

#     return acc, auc, f1


# def extract_accuracy(result):
#     acc, _, _ = extract_metric(result)
#     return acc if acc is not None else 0.0


# # ======================
# # OPTUNA TUNING
# # ======================
# def objective(trial):

#     params = {
#         "lr": trial.suggest_float("lr", 1e-3, 0.3, log=True),
#         "lam_global": trial.suggest_float("lam_global", 1e-5, 1e-2, log=True),
#         "lam_local": trial.suggest_float("lam_local", 1e-5, 1e-2, log=True),
#         "pred_k": trial.suggest_int("pred_k", 1, 10),
#         "max_steps": trial.suggest_int("max_steps", 500, 3000),
#     }

#     # only ONE split (fast)
#     args = get_args(dataset_name, seed=0, fold=0, params=params)

#     output = run_experiment(args)

#     return extract_accuracy(output["test_result"])


# print("=== TUNING START ===")

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=10)

# best_params = study.best_params

# print("Best Params:", best_params)


# # ======================
# # FULL EVALUATION
# # ======================
# final_scores = []
# auc_scores = []
# f1_scores = []
# train_times = []

# for seed in seeds:
#     for fold in range(5):

#         print(f"Seed {seed}, Fold {fold}")

#         args = get_args(dataset_name, seed, fold, best_params)

#         start = time.perf_counter()
#         output = run_experiment(args)
#         end = time.perf_counter()

#         acc, auc, f1 = extract_metric(output["test_result"])

#         final_scores.append(acc)
#         auc_scores.append(auc)
#         f1_scores.append(f1)

#         train_times.append(end - start)


# # ======================
# # AGGREGATION
# # ======================
# Am = float(np.mean(final_scores))
# Sm = float(np.std(final_scores))

# auc_mean = float(np.mean(auc_scores))
# auc_std = float(np.std(auc_scores))

# f1_mean = float(np.mean(f1_scores))
# f1_std = float(np.std(f1_scores))

# Tm = float(np.sum(train_times))


# # ======================
# # SAVE PARAMS
# # ======================
# os.makedirs(f"params/{dataset_name}", exist_ok=True)

# with open(f"params/{dataset_name}/{model_name}_params.json", "w") as f:
#     json.dump(best_params, f, indent=4)


# # ======================
# # SAVE RESULTS
# # ======================
# os.makedirs(f"results/{dataset_name}", exist_ok=True)

# with open(f"results/{dataset_name}/{model_name}_results.json", "w") as f:
#     json.dump(
#         {
#             "model": model_name,
#             "dataset": dataset_name,
#             "performance": {
#                 "accuracy_mean": Am,
#                 "accuracy_std": Sm,
#                 "auc_mean": auc_mean,
#                 "auc_std": auc_std,
#                 "f1_mean": f1_mean,
#                 "f1_std": f1_std,
#             },
#             "resource_usage": {
#                 "wall_clock_train_time_sec": Tm,
#             },
#             "per_run": {
#                 "accuracy": final_scores,
#                 "auc": auc_scores,
#                 "f1": f1_scores,
#                 "train_time": train_times,
#             },
#             "meta": {
#                 "seeds": seeds,
#                 "best_params": best_params
#             },
#         },
#         f,
#         indent=4,
#     )

# print("=== DONE ===")