import types

import pytest

from napari_cochlea_phenotyping._device import (Device, cellpose_model_kwargs, detect_device,
                                                run_with_memory_fallback)


def fake_torch(cuda=False, mps=False, cuda_build="12.1"):
    t = types.SimpleNamespace(__version__="2.5.0")
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i: "NVIDIA Test Card",
                                   get_device_properties=lambda i: types.SimpleNamespace(total_memory=8e9))
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    t.version = types.SimpleNamespace(cuda=cuda_build)
    return t


def test_card_found(monkeypatch):
    monkeypatch.delenv("COCHLEA_PHENOTYPING_DEVICE", raising=False)
    d = detect_device(_torch=fake_torch(cuda=True))
    assert d.kind == "cuda" and d.gpu and d.name == "NVIDIA Test Card" and round(d.memory_gb) == 8
    assert d.label().startswith("Graphics card: NVIDIA Test Card")


def test_processor_forced(monkeypatch):
    monkeypatch.delenv("COCHLEA_PHENOTYPING_DEVICE", raising=False)
    assert detect_device("cpu", _torch=fake_torch(cuda=True)).kind == "cpu"
    monkeypatch.setenv("COCHLEA_PHENOTYPING_DEVICE", "cpu")
    assert detect_device(_torch=fake_torch(cuda=True)).kind == "cpu"
    assert detect_device("cuda", _torch=fake_torch(cuda=True)).kind == "cuda"   # the argument wins


def test_cpu_build_explained(monkeypatch):
    monkeypatch.delenv("COCHLEA_PHENOTYPING_DEVICE", raising=False)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    d = detect_device(_torch=fake_torch(cuda=False, cuda_build=None))
    assert d.kind == "cpu" and "without graphics card support" in d.note


def test_apple(monkeypatch):
    monkeypatch.delenv("COCHLEA_PHENOTYPING_DEVICE", raising=False)
    assert detect_device(_torch=fake_torch(mps=True)).kind == "mps"


def test_bad_choice():
    with pytest.raises(ValueError):
        detect_device("gpu", _torch=fake_torch())


def test_cellpose_kwargs():
    class WithDevice:
        def __init__(self, gpu=False, device=None): pass

    class WithoutDevice:
        def __init__(self, gpu=False): pass

    cpu = Device("cpu", "x", None, None, "")
    assert cellpose_model_kwargs(cpu, WithDevice) == {"gpu": False}
    assert cellpose_model_kwargs(cpu, WithoutDevice) == {"gpu": False}


class OutOfMemoryError(RuntimeError):
    pass


def test_memory_fallback_smaller_batches():
    gpu = Device("cuda", "card", 8.0, "2.5", "")
    seen = []

    def fn(dev, bs):
        seen.append((dev.kind, bs))
        if bs > 2: raise OutOfMemoryError("CUDA out of memory")
        return "ok"

    out, rec = run_with_memory_fallback(fn, gpu)
    assert out == "ok" and rec["batch_size"] == 2 and rec["out_of_memory_retries"] == 2
    assert not rec["fell_back_to_processor"] and seen == [("cuda", 8), ("cuda", 4), ("cuda", 2)]


def test_memory_fallback_processor_only_when_allowed():
    gpu = Device("cuda", "card", 8.0, "2.5", "")

    def fn(dev, bs):
        if dev.kind == "cuda": raise OutOfMemoryError("out of memory")
        return dev.kind

    with pytest.raises(MemoryError):
        run_with_memory_fallback(fn, gpu)
    out, rec = run_with_memory_fallback(fn, gpu, allow_cpu=True)
    assert out == "cpu" and rec["fell_back_to_processor"]


def test_other_errors_not_hidden():
    gpu = Device("cuda", "card", 8.0, "2.5", "")

    def fn(dev, bs):
        raise ValueError("bad field")

    with pytest.raises(ValueError):
        run_with_memory_fallback(fn, gpu, allow_cpu=True)
