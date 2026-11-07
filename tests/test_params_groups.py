"""Tsit5 with ``params_group_size``: consecutive trajectories share one ``params`` row.

Trajectory ``i`` reads row ``i // params_group_size``. With every row written out once per trajectory instead, the
kernel does the same arithmetic on the same numbers, so the two must agree bit for bit.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")

from modax.tsit5 import solve  # noqa: E402

N_TABLE = 32


def tabulated_rate(y, t, p):
    """dy/dt = -k(t) y with k linear on a table held in the row; y[1] scales the rate per trajectory."""
    lo = 0
    hi = N_TABLE - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p[mid] <= t:
            lo = mid
        else:
            hi = mid
    frac = (t - p[lo]) / (p[hi] - p[lo])
    k = p[N_TABLE + lo] + frac * (p[N_TABLE + hi] - p[N_TABLE + lo])
    return (-k * y[1] * y[0], 0.0)


def _inputs(n_groups, group):
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 1.0, N_TABLE)
    slopes = rng.uniform(0.5, 2.0, n_groups)
    rows = np.concatenate([np.broadcast_to(grid, (n_groups, N_TABLE)), slopes[:, None] * (1 + grid[None, :])], axis=1)
    scale = rng.uniform(0.5, 1.5, n_groups * group)
    y0 = np.stack([np.ones_like(scale), scale], axis=1)
    return jnp.asarray(y0), jnp.asarray(rows), slopes, scale


def test_grouped_rows_match_duplicated_rows_bit_for_bit():
    group = 8
    y0, rows, _, _ = _inputs(16, group)
    t_span = jnp.array([0.0, 0.5, 1.0])
    kw = dict(rtol=1e-8, atol=1e-10, error_weights=jnp.array([1.0, 0.0]))
    grouped, stats_g = solve(tabulated_rate, y0, t_span, rows, params_group_size=group, return_stats=True, **kw)
    full, stats_f = solve(tabulated_rate, y0, t_span, jnp.repeat(rows, group, axis=0), return_stats=True, **kw)
    assert grouped.shape == full.shape == (16 * group, 3, 2)
    assert np.array_equal(np.asarray(grouped), np.asarray(full))
    assert np.array_equal(np.asarray(stats_g["accepted_steps"]), np.asarray(stats_f["accepted_steps"]))


def test_grouped_solution_is_correct():
    group = 4
    y0, rows, slopes, scale = _inputs(8, group)
    out = solve(tabulated_rate, y0, jnp.array([0.0, 1.0]), rows, params_group_size=group, rtol=1e-10, atol=1e-12)
    # k(t) = slope (1 + t) exactly on the table, so y(1) = exp(-scale * slope * 1.5)
    exact = np.exp(-scale * np.repeat(slopes, group) * 1.5)
    np.testing.assert_allclose(np.asarray(out[:, -1, 0]), exact, rtol=1e-8)


def test_both_backends_with_groups():
    """Shared memory (small ensembles) and thread-local (large) kernels both index by group."""
    group = 100
    for n_groups in [4, 200]:  # 400 and 20000 trajectories: below and above the shared-memory cap
        y0, rows, _, _ = _inputs(n_groups, group)
        kw = dict(rtol=1e-8, atol=1e-10)
        grouped = solve(tabulated_rate, y0, jnp.array([0.0, 1.0]), rows, params_group_size=group, **kw)
        full = solve(tabulated_rate, y0, jnp.array([0.0, 1.0]), jnp.repeat(rows, group, axis=0), **kw)
        assert np.array_equal(np.asarray(grouped), np.asarray(full))


def test_row_count_must_match():
    y0, rows, _, _ = _inputs(4, 8)
    with pytest.raises(ValueError, match="params_group_size"):
        solve(tabulated_rate, y0, jnp.array([0.0, 1.0]), rows[:3], params_group_size=8)


def test_groups_are_not_differentiable():
    y0, rows, _, _ = _inputs(2, 4)
    with pytest.raises(NotImplementedError, match="params_group_size"):
        jax.grad(lambda r: solve(tabulated_rate, y0, jnp.array([0.0, 1.0]), r, params_group_size=4)[:, -1, 0].sum())(rows)


def test_default_group_size_is_unchanged():
    """The default (1) is today's kernel: one row per trajectory."""
    y0, rows, _, _ = _inputs(8, 1)
    a = solve(tabulated_rate, y0, jnp.array([0.0, 1.0]), rows)
    b = solve(tabulated_rate, y0, jnp.array([0.0, 1.0]), rows, params_group_size=1)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert math.isfinite(float(a[0, -1, 0]))
