"""Time the three subset-drawing strategies compared in the README.

``permutation_test_mean_diff`` draws each permutation's treatment-sized
subset by taking ``argpartition`` of random keys. The README compares
that against ``argsort`` of the same keys and against a per-row
Fisher-Yates shuffle, and cites this script for the measured ratios.
All three run at the report's scale of 10,000 permutations over 120,000
pooled users, in the same 256-row chunks as the shipped code.

The argpartition timing calls the shipped implementation directly. The
other two mirror its chunked loop and change only how the subset is
drawn. Given the same seed, argsort of the keys selects the same subset
as argpartition (the smallest ``n_t`` keys), so the argsort variant must
reproduce the shipped p-value exactly; the script checks this and fails
if the mirror has drifted from the implementation. The draw cost does
not depend on the outcome values, so the arms are plain standard normal
draws rather than the report's revenue mixture.

Run with: python scripts/run_bench_permutation.py (from the repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datetime
import os
import platform
import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from abkit.resampling import permutation_test_mean_diff

N_PERMUTATIONS = 10_000
N_PER_ARM = 60_000
CHUNK = 256  # matches _CHUNK in abkit.resampling
SEED_DATA = 7
SEED_DRAW = 2
REPEATS = 3

ChunkSums = Callable[[np.random.Generator, NDArray[np.float64], int, int], NDArray[np.float64]]


def argsort_sums(
    rng: np.random.Generator, pooled: NDArray[np.float64], size: int, n_t: int
) -> NDArray[np.float64]:
    keys = rng.random((size, pooled.size), dtype=np.float32)
    take = np.argsort(keys, axis=1)[:, :n_t]
    return pooled[take].sum(axis=1)


def shuffle_sums(
    rng: np.random.Generator, pooled: NDArray[np.float64], size: int, n_t: int
) -> NDArray[np.float64]:
    block = np.tile(pooled, (size, 1))
    rng.permuted(block, axis=1, out=block)
    return block[:, :n_t].sum(axis=1)


def time_variant(
    control: NDArray[np.float64], treatment: NDArray[np.float64], chunk_sums: ChunkSums
) -> tuple[float, float]:
    """Run one full permutation test with the given draw and return (seconds, p)."""
    rng = np.random.default_rng(SEED_DRAW)
    start = time.perf_counter()
    pooled = np.concatenate([control, treatment])
    n_t = treatment.size
    n_c = control.size
    total = pooled.sum()
    observed = treatment.mean() - control.mean()
    hits = 0
    done = 0
    while done < N_PERMUTATIONS:
        size = min(CHUNK, N_PERMUTATIONS - done)
        sums_t = chunk_sums(rng, pooled, size, n_t)
        perm_diffs = sums_t / n_t - (total - sums_t) / n_c
        hits += int(np.sum(np.abs(perm_diffs) >= abs(observed)))
        done += size
    elapsed = time.perf_counter() - start
    return elapsed, (1 + hits) / (N_PERMUTATIONS + 1)


def time_shipped(
    control: NDArray[np.float64], treatment: NDArray[np.float64]
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED_DRAW)
    start = time.perf_counter()
    result = permutation_test_mean_diff(control, treatment, rng=rng, n_permutations=N_PERMUTATIONS)
    return time.perf_counter() - start, result.p_value


def main() -> None:
    rng = np.random.default_rng(SEED_DATA)
    control = rng.standard_normal(N_PER_ARM)
    treatment = rng.standard_normal(N_PER_ARM)

    variants: list[tuple[str, Callable[[], tuple[float, float]]]] = [
        ("argpartition (shipped)", lambda: time_shipped(control, treatment)),
        ("argsort", lambda: time_variant(control, treatment, argsort_sums)),
        ("Fisher-Yates shuffle", lambda: time_variant(control, treatment, shuffle_sums)),
    ]

    print("Permutation subset-draw benchmark")
    print(f"date: {datetime.date.today().isoformat()}")
    print(f"machine: {platform.platform()}, {platform.processor()}, {os.cpu_count()} logical cores")
    print(f"python {platform.python_version()}, numpy {np.__version__}")
    print(
        f"scale: {N_PERMUTATIONS:,} permutations over {2 * N_PER_ARM:,} pooled users "
        f"({N_PER_ARM:,} per arm), chunk size {CHUNK}"
    )
    print()

    best: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for name, run in variants:
        times: list[float] = []
        for _ in range(REPEATS):
            elapsed, p_value = run()
            times.append(elapsed)
            p_values[name] = p_value
        best[name] = min(times)
        runs = ", ".join(f"{seconds:.2f}" for seconds in times)
        print(
            f"{name}: best {best[name]:.2f} s of {REPEATS} runs ({runs} s); p = {p_values[name]:.4f}"
        )

    if p_values["argsort"] != p_values["argpartition (shipped)"]:
        raise RuntimeError(
            "the argsort mirror no longer reproduces the shipped p-value, so the "
            "loop in this script has drifted from permutation_test_mean_diff"
        )

    ref = best["argpartition (shipped)"]
    print()
    print(f"argsort / argpartition: {best['argsort'] / ref:.2f}x")
    print(f"Fisher-Yates shuffle / argpartition: {best['Fisher-Yates shuffle'] / ref:.2f}x")


if __name__ == "__main__":
    main()
