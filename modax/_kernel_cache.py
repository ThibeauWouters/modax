"""Keys and storage for compiled kernels that survive the process.

numba builds a solver kernel in seconds, every process again, and the FFI call used to carry the ``CUfunction``
pointer, which made the lowered program -- and so JAX's persistent-cache key -- different in every process. A kernel is
now stored on disk as its linked cubin under a key that fingerprints everything it is built from: the right-hand side
(its code, closure values, defaults and the globals it reads), the solver options, modax's own sources, the numba-cuda-mlir
version and the GPU target. The key also gives the kernel a stable id for the FFI call.

A right-hand side that closes over something the fingerprint cannot read raises `KernelKeyError`; that kernel is then
built as before and given a process-local id, so nothing is cached wrongly.

``MODAX_KERNEL_CACHE_DIR`` sets the directory; ``off`` (or ``0``) disables the disk cache. The default is
``$XDG_CACHE_HOME/modax/kernels`` or ``~/.cache/modax/kernels``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import os
import types
from pathlib import Path
from typing import Any

import numpy as np

_ENV_VAR = "MODAX_KERNEL_CACHE_DIR"
_OFF = {"off", "0", "false", "no"}


class KernelKeyError(TypeError):
    """The fingerprint cannot see everything a kernel depends on."""


def cache_dir() -> Path | None:
    """The directory kernels are stored in, or ``None`` when the cache is off."""
    value = os.environ.get(_ENV_VAR)
    if value is not None:
        if value.strip().lower() in _OFF:
            return None
        return Path(value)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "modax" / "kernels"


def _update(h: "hashlib._Hash", obj: Any, seen: set[int]) -> None:
    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes)):
        h.update(f"{type(obj).__name__}:{obj!r};".encode())
        return
    if isinstance(obj, np.generic):
        h.update(f"np:{obj.dtype}:{obj!r};".encode())
        return
    if isinstance(obj, np.ndarray):
        h.update(f"nd:{obj.dtype}:{obj.shape};".encode())
        h.update(np.ascontiguousarray(obj).tobytes())
        return
    if isinstance(obj, (tuple, list, frozenset, set)):
        items = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else obj
        h.update(f"{type(obj).__name__}[{len(items)}];".encode())
        for item in items:
            _update(h, item, seen)
        return
    if isinstance(obj, dict):
        h.update(f"dict[{len(obj)}];".encode())
        for key in sorted(obj, key=repr):
            _update(h, key, seen)
            _update(h, obj[key], seen)
        return
    if isinstance(obj, types.ModuleType):
        h.update(f"module:{obj.__name__};".encode())
        return
    if isinstance(obj, type):
        h.update(f"type:{obj.__module__}.{obj.__qualname__};".encode())
        return
    if isinstance(obj, types.CodeType):
        h.update(b"code;")
        h.update(obj.co_code)
        _update(h, obj.co_names, seen)
        _update(h, obj.co_varnames, seen)
        _update(h, obj.co_freevars, seen)
        for const in obj.co_consts:
            _update(h, const, seen)
        return
    if isinstance(obj, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        h.update(f"builtin:{getattr(obj, '__module__', '')}.{obj.__qualname__};".encode())
        return
    if isinstance(obj, functools.partial):
        h.update(b"partial;")
        _update(h, obj.func, seen)
        _update(h, obj.args, seen)
        _update(h, obj.keywords, seen)
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(f"dataclass:{type(obj).__qualname__};".encode())
        _update(h, dataclasses.astuple(obj), seen)
        return
    py_func = getattr(obj, "py_func", None)
    if py_func is not None:  # a numba dispatcher (cuda.jit device function or kernel)
        h.update(f"dispatcher:{type(obj).__name__};".encode())
        _update(h, py_func, seen)
        _update(h, dict(getattr(obj, "targetoptions", {})), seen)
        return
    if isinstance(obj, types.FunctionType):
        if id(obj) in seen:  # recursion through globals
            h.update(f"function-ref:{obj.__qualname__};".encode())
            return
        seen.add(id(obj))
        h.update(f"function:{obj.__module__}.{obj.__qualname__};".encode())
        _update(h, obj.__code__, seen)
        _update(h, obj.__defaults__, seen)
        _update(h, obj.__kwdefaults__, seen)
        cells = obj.__closure__ or ()
        _update(h, len(cells), seen)
        for cell in cells:
            try:
                value = cell.cell_contents
            except ValueError:  # empty cell
                value = None
            _update(h, value, seen)
        # Globals the code reads: functions it calls and constants it uses.
        for name in sorted(_global_names(obj.__code__)):
            if name in obj.__globals__:
                _update(h, name, seen)
                _update(h, obj.__globals__[name], seen)
        return
    if hasattr(obj, "_fields") and isinstance(obj, tuple):  # NamedTuple
        _update(h, tuple(obj), seen)
        return
    raise KernelKeyError(f"cannot fingerprint {type(obj).__module__}.{type(obj).__qualname__}")


def _global_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_names(const)
    return names


def fingerprint(obj: Any) -> str:
    """SHA-256 of everything ``obj`` (typically a right-hand side) is built from.

    Raises:
        KernelKeyError: ``obj`` refers to something whose content cannot be read, such as an arbitrary object.
    """
    h = hashlib.sha256()
    _update(h, obj, set())
    return h.hexdigest()


@functools.cache
def _environment_digest() -> str:
    """modax's own sources, the numba-cuda-mlir version and the GPU target: a change in any invalidates every kernel."""
    import numba_cuda_mlir

    from modax._jax_numba_custom_call import compile_target

    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(str(getattr(numba_cuda_mlir, "__version__", "unknown")).encode())
    h.update(repr(compile_target()).encode())
    return h.hexdigest()


def kernel_key(*parts: Any) -> str | None:
    """Cache key of a kernel built from ``parts``, or ``None`` if one of them cannot be fingerprinted."""
    try:
        return fingerprint((_environment_digest(),) + parts)
    except KernelKeyError:
        return None


def load(key: str) -> tuple[bytes, str] | None:
    """The stored cubin and entry name of ``key``, if the cache is on and holds it."""
    directory = cache_dir()
    if directory is None:
        return None
    cubin, name = directory / f"{key}.cubin", directory / f"{key}.name"
    if not (cubin.exists() and name.exists()):
        return None
    return cubin.read_bytes(), name.read_text()


def store(key: str, cubin: bytes, func_name: str) -> None:
    """Store a compiled kernel under ``key`` (atomically, so concurrent processes never read half a file)."""
    directory = cache_dir()
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, data in ((".cubin", cubin), (".name", func_name.encode())):
        target = directory / f"{key}{suffix}"
        tmp = target.with_suffix(f"{suffix}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
