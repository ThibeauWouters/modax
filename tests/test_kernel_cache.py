"""Kernels are cached on disk across processes, and the lowered program no longer depends on the process.

The FFI call used to carry the raw ``CUfunction`` pointer as an attribute, so the HLO (and JAX's persistent-cache key)
differed in every process and the numba build of the kernel (seconds) was repeated in every process. Now the attribute is
a stable kernel id, resolved by the launcher in a per-process registry, and the compiled cubin is stored under a key that
fingerprints everything the kernel depends on.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from modax._kernel_cache import KernelKeyError, fingerprint

ROOT = Path(__file__).resolve().parents[1]


def _decay_factory(k):
    def rhs(y, t, p):
        return (-k * p[0] * y[0],)

    return rhs


def test_fingerprint_sees_closure_values():
    assert fingerprint(_decay_factory(2.0)) == fingerprint(_decay_factory(2.0))
    assert fingerprint(_decay_factory(2.0)) != fingerprint(_decay_factory(3.0))


def test_fingerprint_sees_code():
    def a(y, t, p):
        return (-p[0] * y[0],)

    def b(y, t, p):
        return (p[0] * y[0],)

    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_sees_arrays_and_rejects_unknown_objects():
    table = np.arange(4.0)

    def with_table(y, t, p):
        return (table[0] * y[0],)

    key1 = fingerprint(with_table)
    table[0] = 5.0
    assert fingerprint(with_table) != key1

    unknown = object()

    def with_unknown(y, t, p):
        return (unknown, y[0])

    with pytest.raises(KernelKeyError):
        fingerprint(with_unknown)


def _cuda_available() -> bool:
    try:
        from numba_cuda_mlir import cuda
    except ImportError:
        return False
    return bool(cuda.is_available())


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA required")

_SCRIPT = textwrap.dedent(
    """
    import hashlib, json, sys
    import jax, jax.numpy as jnp, numpy as np
    import modax._jax_numba_custom_call as C

    if sys.argv[1] == "forbid-compile":
        def no_compile(*args, **kwargs):
            raise RuntimeError("the kernel was compiled although it is cached")
        C.target_kernel = no_compile

    from modax.tsit5 import solve

    def rhs(y, t, p):
        return (-p[0] * y[0], p[0] * y[0] - p[1] * y[1])

    y0 = jnp.array([1.0, 0.0])
    params = jnp.stack([jnp.linspace(0.5, 2.0, 64), jnp.ones(64)], axis=1)
    f = jax.jit(lambda p: solve(rhs, y0, jnp.array([0.0, 1.0]), p, rtol=1e-8, atol=1e-10))
    text = f.lower(params).as_text()
    out = np.asarray(f(params))
    print(json.dumps({"hlo": hashlib.sha256(text.encode()).hexdigest(), "out": out[:, -1, :].tolist()}))
    """
)


def _run(tmp_path, mode, cache_dir):
    env = dict(os.environ, MODAX_KERNEL_CACHE_DIR=str(cache_dir), PYTHONPATH=str(ROOT))
    script = tmp_path / "solve_once.py"
    script.write_text(_SCRIPT)
    res = subprocess.run(
        [sys.executable, str(script), mode],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


@requires_cuda
def test_cached_kernel_is_reused_in_a_new_process(tmp_path):
    cache_dir = tmp_path / "kernels"
    first = _run(tmp_path, "compile", cache_dir)
    assert any(cache_dir.glob("*.cubin")), "no kernel was written to the cache"
    second = _run(tmp_path, "forbid-compile", cache_dir)
    assert second["out"] == first["out"]


@requires_cuda
def test_lowered_program_is_the_same_in_every_process(tmp_path):
    """No process-specific pointer in the HLO, so JAX's persistent compilation cache can hit."""
    cache_dir = tmp_path / "kernels"
    a = _run(tmp_path, "compile", cache_dir)
    b = _run(tmp_path, "compile", cache_dir)
    assert a["hlo"] == b["hlo"]


@requires_cuda
def test_cache_can_be_switched_off(tmp_path):
    """MODAX_KERNEL_CACHE_DIR=off disables the cache instead of naming a directory."""
    res = _run(tmp_path, "compile", "off")
    assert not (tmp_path / "off").exists()
    assert len(res["out"]) == 64
