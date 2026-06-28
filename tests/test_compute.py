"""Unit tests for the ComputeConfig runtime helpers (v2 Phase H1)."""
import os

from evoliez.utils.compute import (
    apply_cpu_budget, gpu_for_index, gpu_pool_list, max_workers,
)


def test_apply_cpu_budget_sets_thread_vars(monkeypatch):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    n = apply_cpu_budget(16)
    assert n == 4 and os.environ["OMP_NUM_THREADS"] == "4"   # per-proc default 4 (watchdog-safe)
    assert os.environ["MKL_NUM_THREADS"] == "4"


def test_apply_cpu_budget_respects_smaller_existing(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    for v in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    n = apply_cpu_budget(16)
    assert n == 2   # never raise an already-lower launcher pin


def test_gpu_pool_list_config_and_env(monkeypatch):
    assert gpu_pool_list([0, 2, 3]) == [0, 2, 3]            # explicit config wins
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3")
    assert gpu_pool_list(None) == [0, 2, 3]                 # else parse env
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gpu_pool_list(None) == []                        # else empty (single/CPU)


def test_max_workers_and_roundrobin(monkeypatch):
    assert max_workers(16, per_task_cores=4) == 4
    assert max_workers(1, per_task_cores=4) == 1            # never 0
    assert gpu_for_index(0, [0, 2, 3]) == 0
    assert gpu_for_index(4, [0, 2, 3]) == 2                 # round-robin
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gpu_for_index(0, []) is None                    # empty pool -> no pin
