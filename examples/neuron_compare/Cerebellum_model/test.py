#!/usr/bin/env python3
r"""Cerebellar micro-network — real cells + distance-based connectivity.

Builds a small number of biophysical multi-compartment cells, assigns each
a 3-D soma position, and wires them with distance-dependent connection
probability kernels (step, Gaussian, exponential).

This is an **integration smoke test** — it verifies that the
``distance_connectivity`` module works end-to-end with real ``braincell.Cell``
objects.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ---- resolve repo root -------------------------------------------------------
_CWD = Path(__file__).resolve().parent
_REPO_ROOT = _CWD
for _c in (_CWD, *_CWD.parents):
    if (_c / "braincell").exists() and (_c / "examples").exists():
        _REPO_ROOT = _c
        break
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CELL_DIR = _REPO_ROOT / "examples" / "neuron_compare" / "cell"
if str(_CELL_DIR) not in sys.path:
    sys.path.insert(0, str(_CELL_DIR))

import brainstate
import brainunit as u
import numpy as np

import braincell
from braincell import mech
from braincell.filter import at

# ---- cell-model imports ------------------------------------------------------
from grc_ma2020.grc_braincell import GrC as GrCBuilder
from grc_ma2020.parameters import DEFAULT_MORPH_PATH as GRC_MORPH, load_grc20_params

from goc_ma2020.goc_braincell import GoC as GoCBuilder
from goc_ma2020.parameters import DEFAULT_MORPH_PATH as GOC_MORPH, load_goc20_params

from pc_ma2024.pc_braincell import PC as PCBuilder
from pc_ma2024.parameters import DEFAULT_MORPH_PATH as PC_MORPH, load_pc24_params

from sc_ma2021.sc_braincell import SC as SCBuilder
from sc_ma2021.parameters import DEFAULT_MORPH_PATH as SC_MORPH, load_sc21_params

from bc_ma2025.bc_braincell import BC as BCBuilder
from bc_ma2025.parameters import DEFAULT_MORPH_PATH as BC_MORPH, load_bc25_params

# ---- distance-connectivity imports -------------------------------------------
from distance_connectivity import (  # noqa: E402
    ExponentialConnection,
    GaussianConnection,
    SpatialPopulation,
    StepConnection,
    create_random_positions,
    make_kernel,
    pairwise_euclidean_distance,
)

brainstate.environ.set(precision=64)

# ═══════════════════════════════════════════════════════════════════════════════
# Cell factory — one builder call → one Cell
# ═══════════════════════════════════════════════════════════════════════════════

DT = 0.1 * u.ms
V_INIT_MV = -65.0
V_THRESHOLD = 0.0 * u.mV
DEFAULT_TAU = 2.0 * u.ms

_CELL_SPECS: dict[str, dict[str, Any]] = {
    "GrC": {"builder": GrCBuilder, "params": load_grc20_params,
            "morph": GRC_MORPH, "temp": 25.0},
    "GoC": {"builder": GoCBuilder, "params": load_goc20_params,
            "morph": GOC_MORPH, "temp": 34.0},
    "PC":  {"builder": PCBuilder,  "params": lambda: load_pc24_params(indiv=138),
            "morph": PC_MORPH,  "temp": 36.0},
    "SC":  {"builder": SCBuilder,  "params": load_sc21_params,
            "morph": SC_MORPH,  "temp": 32.0},
    "BC":  {"builder": BCBuilder,  "params": load_bc25_params,
            "morph": BC_MORPH,  "temp": 36.0},
}

_POP_SIZES: dict[str, int] = {"GrC": 3, "GoC": 2, "PC": 2, "SC": 2, "BC": 2}

# ═══════════════════════════════════════════════════════════════════════════════
# Projection table — each entry specifies pre → post + distance kernel
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECTIONS: list[dict[str, Any]] = [
    {"pre": "GrC", "post": "GoC", "kind": "step",
     "kw": {"d_threshold": 200.0 * u.um, "p_max": 0.8},
     "weight": 0.002 * u.uS, "e": 0.0 * u.mV},

    {"pre": "GoC", "post": "GrC", "kind": "exponential",
     "kw": {"lambda_": 200.0 * u.um, "p_max": 0.6},
     "weight": 0.00014 * u.uS, "e": -75.0 * u.mV},

    {"pre": "GrC", "post": "PC", "kind": "gaussian",
     "kw": {"sigma": 150.0 * u.um, "p_max": 0.7},
     "weight": 0.006 * u.uS, "e": 0.0 * u.mV},

    {"pre": "GrC", "post": "SC", "kind": "exponential",
     "kw": {"lambda_": 180.0 * u.um, "p_max": 0.7},
     "weight": 0.00045 * u.uS, "e": 0.0 * u.mV},

    {"pre": "GrC", "post": "BC", "kind": "gaussian",
     "kw": {"sigma": 120.0 * u.um, "p_max": 0.7},
     "weight": 0.00035 * u.uS, "e": 0.0 * u.mV},

    {"pre": "SC", "post": "PC", "kind": "step",
     "kw": {"d_threshold": 200.0 * u.um, "p_max": 0.8},
     "weight": 0.025 * u.uS, "e": -75.0 * u.mV},

    {"pre": "BC", "post": "PC", "kind": "gaussian",
     "kw": {"sigma": 150.0 * u.um, "p_max": 0.7},
     "weight": 0.02 * u.uS, "e": -75.0 * u.mV},
]

# ═══════════════════════════════════════════════════════════════════════════════


def _syn_name(pre: str, post: str) -> str:
    return f"syn_{pre}_to_{post}"


def _build_one_cell(name: str) -> braincell.Cell:
    """Build a single multi-compartment cell of type *name*."""
    spec = _CELL_SPECS[name]
    params = spec["params"]()
    builder = spec["builder"](
        spec["morph"],
        params=params,
        temperature_celsius=spec["temp"],
        v_init_mV=V_INIT_MV,
    )
    assembly = builder.build()
    cell = assembly.cell
    cell.V_th = V_THRESHOLD
    return cell


def run() -> int:
    brainstate.environ.set(dt=DT)
    rng = np.random.default_rng(7)

    print("=" * 64)
    print("Cerebellar micro-network — real cells + distance connectivity")
    print("=" * 64)
    print(f"braincell: {braincell.__version__}    dt: {DT}")
    print(f"cell types: {len(_CELL_SPECS)}   total cells: {sum(_POP_SIZES.values())}")
    print(f"projections: {len(_PROJECTIONS)}")
    print()

    # ---- 1. Build cells ------------------------------------------------------
    print("─" * 64)
    print("Building cells …")
    print("─" * 64)

    total_start = time.perf_counter()
    cells: dict[str, list[braincell.Cell]] = {}
    build_rows: list[dict] = []

    for name, n in _POP_SIZES.items():
        cells[name] = []
        t0 = time.perf_counter()
        for i in range(n):
            try:
                c = _build_one_cell(name)
                cells[name].append(c)
            except Exception as exc:
                why = str(exc).split("\n")[0]
                build_rows.append(dict(pop=name, idx=i, ok=False, err=why))
                break
        dt_build = time.perf_counter() - t0
        ok_count = len(cells[name])
        build_rows.append(dict(pop=name, n=ok_count, t=round(dt_build, 2), ok=True))
        print(f"  {name:<6}  {ok_count}/{n} built  ({dt_build:.1f}s)")

    n_built = sum(len(c) for c in cells.values())
    n_total = sum(_POP_SIZES.values())
    if n_built == 0:
        print("  ✗ No cells built — aborting")
        return 1
    print(f"  → {n_built}/{n_total} cells")

    # ---- 2. Assign spatial positions → SpatialPopulation ---------------------
    print()
    print("─" * 64)
    print("Assigning spatial positions …")
    print("─" * 64)

    pops: dict[str, SpatialPopulation] = {}
    cell_coords: list[tuple[str, int, float, float, float]] = []
    for name in cells:
        n = len(cells[name])
        pos = create_random_positions(n, bounds=(-100.0 * u.um, 100.0 * u.um), rng=rng)
        pops[name] = SpatialPopulation(name, pos)
        # attach cell list to the population for synapse placement
        pops[name].cells = cells[name]
        for i in range(n):
            cell_coords.append((name, i, float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
        print(f"  {name:<6}  N={n}  pos∈[{pos.min():.0f}, {pos.max():.0f}] µm")

    # print a few individual cell coordinates
    print()
    print("  Individual soma coordinates (random sample):")
    import random
    for name, idx, x, y, z in random.sample(cell_coords, k=min(8, len(cell_coords))):
        print(f"    {name}[{idx}]  ({x:7.1f}, {y:7.1f}, {z:7.1f}) µm")

    # ---- 3. Build kernels + generate connections -----------------------------
    print()
    print("─" * 64)
    print("Generating distance-based connections …")
    print("─" * 64)

    kernels: dict[str, Any] = {}
    edge_results: list[dict] = []
    errors: list[str] = []

    for proj in _PROJECTIONS:
        pre_n, post_n = proj["pre"], proj["post"]
        if pre_n not in pops or post_n not in pops:
            continue
        key = f"{pre_n}→{post_n}"

        kernel = make_kernel(proj["kind"], **proj["kw"])
        kernels[key] = kernel

        pre_pos = pops[pre_n].positions
        post_pos = pops[post_n].positions
        n_possible = pops[pre_n].size * pops[post_n].size

        pre_idx, post_idx = kernel.generate_connections(pre_pos, post_pos, rng=rng)
        n_edge = len(pre_idx)

        # ---- place ExpSyn on post-synaptic cells -------------------------
        post_cells = pops[post_n].cells
        for post_i in np.unique(post_idx):
            post_cells[post_i].place(
                at("soma", 0.5),
                mech.Synapse(
                    synapse_type="ExpSyn",
                    params=dict(
                        tau=DEFAULT_TAU,
                        e=proj["e"],
                        weight=proj["weight"],
                    ),
                    name=_syn_name(pre_n, post_n),
                ),
            )

        # ---- statistics --------------------------------------------------
        if n_edge > 0:
            d_real = pairwise_euclidean_distance(pre_pos, post_pos)[pre_idx, post_idx]
            d_mean, d_min, d_max = float(d_real.mean()), float(d_real.min()), float(d_real.max())
        else:
            d_mean = d_min = d_max = float("nan")

        p_mean = float(kernel.probability_matrix(pre_pos, post_pos).mean())

        # ---- validations -------------------------------------------------
        if n_edge > 0:
            if pre_idx.min() < 0 or pre_idx.max() >= pops[pre_n].size:
                errors.append(f"{key}: pre_idx out of range")
            if post_idx.min() < 0 or post_idx.max() >= pops[post_n].size:
                errors.append(f"{key}: post_idx out of range")

        if proj["kind"] == "step" and n_edge > 0:
            thresh_um = proj["kw"]["d_threshold"].to_decimal(u.um)
            if d_max > thresh_um + 1e-9:
                errors.append(f"{key}: step d_max={d_max:.0f} > threshold={thresh_um:.0f}")

        edge_results.append(dict(
            proj=key, kind=proj["kind"], edges=n_edge,
            possible=n_possible, p_theory=round(p_mean, 4),
            d_mean=round(d_mean, 1),
            d_range=(round(d_min, 0), round(d_max, 0)),
        ))

        print(f"  {key:<10}  {proj['kind']:<12}  "
              f"edges={n_edge:2d}/{n_possible:3d}  "
              f"〈p〉={p_mean:.4f}  〈d〉={d_mean:5.0f} µm  "
              f"[{d_min:4.0f}, {d_max:4.0f}] µm")

    # ---- 4. p(d) formula check -----------------------------------------------
    print()
    print("─" * 64)
    print("Kernel formula validation …")
    print("─" * 64)

    d_check = np.array([0.0, 50.0, 100.0, 200.0, 400.0]) * u.um
    d_vals = d_check.to_decimal(u.um)

    for key, kernel in kernels.items():
        p_vals = kernel.p_distance(d_check)
        if kernel.kernel_name == "step":
            expected = np.where(d_vals <= kernel.d_threshold_um, kernel.p_max, 0.0)
        elif kernel.kernel_name == "gaussian":
            expected = kernel.p_max * np.exp(-d_vals**2 / (2 * kernel.sigma_um**2))
        else:
            expected = kernel.p_max * np.exp(-d_vals / kernel.lambda_um)

        ok = np.allclose(p_vals, expected)
        if not ok:
            errors.append(f"{key}: p_distance formula mismatch")

    if errors:
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print(f"  ✓ all {len(kernels)} kernels pass formula check")

    # ---- 5. Init + reset cells -----------------------------------------------
    print()
    print("─" * 64)
    print("Initialising cells …")
    print("─" * 64)

    t_init = time.perf_counter()
    for clist in cells.values():
        for c in clist:
            c.init_state()
            c.reset_state()
    t_init = time.perf_counter() - t_init
    print(f"  {n_built} cells init + reset  ({t_init:.1f}s)")

    # ---- 6. Report -----------------------------------------------------------
    total_t = time.perf_counter() - total_start
    total_edges = sum(r["edges"] for r in edge_results)

    print()
    print("=" * 64)
    if errors:
        print(f"FAILED — {len(errors)} error(s)")
        return 1
    else:
        print(f"OK — {n_built} cells, {len(edge_results)} projections, "
              f"{total_edges} edges, 0 errors")
        print(f"total time: {total_t:.1f}s")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
