"""Where the models run: an NVIDIA graphics card, an Apple chip, or the processor.

detect_device() is called when the panel opens and its result is shown at the top of the panel and written into
every results file, because outlines from the graphics card and the processor differ slightly. A whole batch therefore
runs on one device. If the card runs out of memory on a large field, run_with_memory_fallback() first retries the field
on the card with smaller batches, and only then (when allowed) on the processor, recording which device was used.

The choice can be forced with the environment variable COCHLEA_PHENOTYPING_DEVICE (auto, cuda, mps or cpu) or from the
panel's settings.
"""
from __future__ import annotations

import inspect
import os
import platform
from dataclasses import asdict, dataclass
from typing import Callable, Optional

ENV_VAR = "COCHLEA_PHENOTYPING_DEVICE"
CHOICES = ("auto", "cuda", "mps", "cpu")


@dataclass(frozen=True)
class Device:
    kind: str                       # "cuda", "mps" or "cpu"
    name: str                       # card or processor name
    memory_gb: Optional[float]      # card memory, when known
    torch_version: Optional[str]
    note: str                       # why this device was chosen, in plain words

    @property
    def gpu(self) -> bool:
        return self.kind in ("cuda", "mps")

    def label(self) -> str:
        if self.kind == "cuda":
            mem = f", {self.memory_gb:.0f} GB" if self.memory_gb else ""
            return f"Graphics card: {self.name}{mem}"
        if self.kind == "mps":
            return f"Apple graphics: {self.name}"
        return f"Processor: {self.name}"

    def torch_device(self):
        import torch
        return torch.device("cuda:0" if self.kind == "cuda" else self.kind)

    def as_dict(self) -> dict:
        return asdict(self)


def _cpu_name() -> str:
    return platform.processor() or platform.machine() or "processor"


def detect_device(prefer: str = "auto", _torch=None) -> Device:
    """Find the best device PyTorch can use. prefer: auto, cuda, mps or cpu (the argument wins over the environment
    variable unless it is auto). _torch lets tests pass a stand-in for the torch module."""
    prefer = (prefer or "auto").lower()
    if prefer == "auto":
        prefer = os.environ.get(ENV_VAR, "auto").lower() or "auto"
    if prefer not in CHOICES:
        raise ValueError(f"device must be one of {CHOICES}, got {prefer!r}")
    torch = _torch
    if torch is None:
        try:
            import torch  # noqa: F811
        except Exception as e:  # torch missing or broken install
            return Device("cpu", _cpu_name(), None, None, f"PyTorch could not be loaded ({type(e).__name__}); models run on the processor")
    tv = getattr(torch, "__version__", None)
    if prefer == "cpu":
        return Device("cpu", _cpu_name(), None, tv, "processor chosen in settings")
    cuda_ok = bool(torch.cuda.is_available())
    if cuda_ok and prefer in ("auto", "cuda"):
        mem = None
        try:
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            pass
        return Device("cuda", torch.cuda.get_device_name(0), mem, tv, "NVIDIA graphics card found")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    mps_ok = bool(mps is not None and mps.is_available())
    if mps_ok and prefer in ("auto", "mps"):
        return Device("mps", "Apple silicon", None, tv, "Apple graphics found")
    if getattr(getattr(torch, "version", None), "cuda", None) is None and platform.system() != "Darwin":
        why = ("this PyTorch was installed without graphics card support; to use an NVIDIA card, reinstall PyTorch "
               "with CUDA from pytorch.org")
    else:
        why = "PyTorch cannot see a graphics card (check the NVIDIA driver)"
    if prefer in ("cuda", "mps"):
        why = f"{prefer} was asked for, but {why}"
    return Device("cpu", _cpu_name(), None, tv, why + "; models run on the processor")


def cellpose_model_kwargs(device: Device, model_class=None) -> dict:
    """Keyword arguments that put a Cellpose model on this device: gpu=..., plus device=... when the installed
    Cellpose accepts it (needed for Apple graphics and for choosing a card)."""
    kw = {"gpu": device.gpu}
    if model_class is None:
        try:
            from cellpose import models
            model_class = models.CellposeModel
        except Exception:
            return kw
    try:
        accepts = "device" in inspect.signature(model_class.__init__).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts and device.gpu:
        kw["device"] = device.torch_device()
    return kw


def _is_out_of_memory(err: BaseException) -> bool:
    name = type(err).__name__
    return name == "OutOfMemoryError" or "out of memory" in str(err).lower()


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_with_memory_fallback(fn: Callable, device: Device, batch_sizes=(8, 4, 2, 1), allow_cpu: bool = False,
                             cpu_device: Optional[Device] = None):
    """Call fn(device, batch_size). On a graphics-card out-of-memory error, retry with the next smaller batch size;
    if all fail and allow_cpu, run once on the processor. Returns (result, record) where record says which device and
    batch size produced the result and whether it fell back, so the results table and QC report can show it."""
    tried = []
    sizes = batch_sizes if device.gpu else batch_sizes[:1]
    for bs in sizes:
        try:
            out = fn(device, bs)
            return out, dict(device=device.kind, device_name=device.name, batch_size=bs,
                             fell_back_to_processor=False, out_of_memory_retries=len(tried))
        except Exception as e:
            if not (device.gpu and _is_out_of_memory(e)):
                raise
            tried.append(bs)
            _empty_cache()
    if not allow_cpu:
        raise MemoryError(f"graphics card ran out of memory at batch sizes {tried}; allow the processor or use a "
                          "smaller field")
    cpu = cpu_device or Device("cpu", _cpu_name(), None, device.torch_version, "fallback after card ran out of memory")
    out = fn(cpu, batch_sizes[-1])
    return out, dict(device="cpu", device_name=cpu.name, batch_size=batch_sizes[-1], fell_back_to_processor=True,
                     out_of_memory_retries=len(tried))
