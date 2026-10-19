"""Runtime device selection with an actual smoke test.

GPU *hardware* being present, a driver being installed and PyTorch being able
to run kernels on it are three different things, so we probe each backend by
running a small matmul + backward pass before selecting it.
"""
from __future__ import annotations

import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DeviceReport:
    selected: str
    cuda_available: bool = False
    cuda_device_name: Optional[str] = None
    cuda_total_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    mps_built: bool = False
    mps_available: bool = False
    amp_dtype: Optional[str] = None
    smoke_test_ms: Optional[float] = None
    torch_version: str = ""
    machine: str = field(default_factory=platform.machine)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _smoke(device: str) -> float:
    import torch

    t0 = time.perf_counter()
    x = torch.randn(256, 256, device=device, requires_grad=True)
    y = (x @ x.T).tanh().sum()
    y.backward()
    _ = float(x.grad.abs().sum().item())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000


def select_device(preferred: str = "auto") -> DeviceReport:
    """Return the best *working* device: cuda > mps > cpu (or `preferred`)."""
    import torch

    rep = DeviceReport(selected="cpu", torch_version=torch.__version__)
    rep.cuda_available = torch.cuda.is_available()
    rep.mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    rep.mps_available = bool(rep.mps_built and torch.backends.mps.is_available())
    if rep.cuda_available:
        try:
            rep.cuda_device_name = torch.cuda.get_device_name(0)
            rep.cuda_total_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            rep.cuda_version = torch.version.cuda
        except Exception as e:  # pragma: no cover - hardware specific
            rep.errors.append(f"cuda query: {e}")

    candidates = ["cuda", "mps", "cpu"] if preferred == "auto" else [preferred, "cpu"]
    for dev in candidates:
        if dev == "cuda" and not rep.cuda_available:
            continue
        if dev == "mps" and not rep.mps_available:
            continue
        try:
            rep.smoke_test_ms = round(_smoke(dev), 2)
            rep.selected = dev
            break
        except Exception as e:  # fall back gracefully
            rep.errors.append(f"{dev} smoke test failed: {e}")
            logger.warning("Device %s failed smoke test: %s", dev, e)

    # Mixed precision: bf16/fp16 autocast on CUDA. On MPS, autocast is opt-in
    # (`amp: on`): on the M1 Pro used here fp16 autocast was *slower* than fp32
    # (0.52 vs 0.46 s/step, BERT-base-sized stack, bs16, len64; numerically finite),
    # see results/runs/_device/device_report.json.
    if rep.selected == "cuda":
        rep.amp_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    logger.info("Selected device: %s (%s)", rep.selected, rep.cuda_device_name or rep.machine)
    return rep
