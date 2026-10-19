"""Seeds, environment capture and resource monitoring."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import random
import sys

PACKAGES = [
    "torch", "transformers", "tokenizers", "datasets", "scikit-learn", "numpy",
    "pandas", "scipy", "shap", "matplotlib", "seaborn", "streamlit", "accelerate",
]


def set_seed(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def package_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def memory_snapshot() -> dict:
    """Process RSS and system memory in GB (plus MPS/CUDA allocations if any)."""
    import psutil

    vm = psutil.virtual_memory()
    snap = {
        "rss_gb": round(psutil.Process().memory_info().rss / 2**30, 3),
        "sys_total_gb": round(vm.total / 2**30, 2),
        "sys_available_gb": round(vm.available / 2**30, 2),
    }
    try:
        import torch

        if torch.cuda.is_available():
            snap["cuda_alloc_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
            snap["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        elif torch.backends.mps.is_available():
            snap["mps_alloc_gb"] = round(torch.mps.current_allocated_memory() / 2**30, 3)
            snap["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 3)
    except Exception:
        pass
    return snap


def hardware_summary() -> dict:
    import psutil

    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
    }
