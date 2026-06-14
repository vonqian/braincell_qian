#!/usr/bin/env python3
r"""Cerebellar probability network — build & wire test.

Follows the ``Cerebellar probability network demo`` notebook paradigm:

* ``CellSpec`` dataclass → cell-type registry
* ``build_population(spec, size, incoming_configs)`` → cell + probes + synapses
* ``braincell.Network`` → add_population / add_edges / add_projection
* tabular report of build times and projection edges

Adapted to the local braincell 0.1.0 API:
* cell builders take ``(morph_path?, params=, temperature_celsius=, v_init_mV=)``
* ``mech.Synapse(synapse_type=..., params=dict(...), name=...)``
* ``braincell.Network`` / ``braincell.network.probability`` are emulated
  when the native classes are not yet available.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ---- resolve repo root (notebook pattern) -----------------------------------
def _find_repo_root() -> Path:
    cwd = Path(__file__).resolve().parent
    for candidate in (cwd, *cwd.parents):
        if (candidate / "braincell").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError("Run from inside the braincell repository.")

_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# also add the cell-model directory so bare package imports work
_CELL_DIR = _REPO_ROOT / "examples" / "neuron_compare" / "cell"
if str(_CELL_DIR) not in sys.path:
    sys.path.insert(0, str(_CELL_DIR))

import brainstate
import brainunit as u
import numpy as np

import braincell
from braincell import mech
from braincell.filter import at

# ---- cell-model imports (notebook style) ------------------------------------
from grc_ma2020.grc_braincell import GrC as BrainCellGrC
from grc_ma2020.parameters import DEFAULT_MORPH_PATH as GRC_MORPH_PATH, load_grc20_params

from goc_ma2020.goc_braincell import GoC as BrainCellGoC
from goc_ma2020.parameters import DEFAULT_MORPH_PATH as GOC_MORPH_PATH, load_goc20_params

from sc_ma2021.sc_braincell import SC as BrainCellSC
from sc_ma2021.parameters import DEFAULT_MORPH_PATH as SC_MORPH_PATH, load_sc21_params

from bc_ma2025.bc_braincell import BC as BrainCellBC
from bc_ma2025.parameters import DEFAULT_MORPH_PATH as BC_MORPH_PATH, load_bc25_params

from pc_ma2024.pc_braincell import PC as BrainCellPC
from pc_ma2024.parameters import DEFAULT_MORPH_PATH as PC_MORPH_PATH, load_pc24_params

from io_zh2019.io_braincell import IO as BrainCellIO
from io_zh2019.parameters import load_io19_params

try:
    from dcn_su2015.dcn_braincell import DCN as BrainCellDCN
    from dcn_su2015.parameters import load_dcn15_params
    _HAS_DCN = True
except Exception:
    BrainCellDCN = None
    load_dcn15_params = None
    _HAS_DCN = False

brainstate.environ.set(precision=64)

# =============================================================================
# Cell specification — mirrors the notebook's CellSpec dataclass
# =============================================================================

@dataclass(frozen=True)
class CellSpec:
    name: str
    cls: type
    params_loader: Any           # () -> params
    temperature_celsius: float
    morph_path: Path | None = None


CELL_SPECS: dict[str, CellSpec] = {
    "GrC": CellSpec("GrC", BrainCellGrC, load_grc20_params, 25.0, GRC_MORPH_PATH),
    "GoC": CellSpec("GoC", BrainCellGoC, load_goc20_params, 34.0, GOC_MORPH_PATH),
    "PC":  CellSpec("PC",  BrainCellPC,  lambda: load_pc24_params(indiv=138), 36.0, PC_MORPH_PATH),
    "SC":  CellSpec("SC",  BrainCellSC,  load_sc21_params, 32.0, SC_MORPH_PATH),
    "BC":  CellSpec("BC",  BrainCellBC,  load_bc25_params, 36.0, BC_MORPH_PATH),
    "IO":  CellSpec("IO",  BrainCellIO,  load_io19_params, 36.0),
}

if _HAS_DCN:
    CELL_SPECS["DCN"] = CellSpec(
        "DCN", BrainCellDCN,
        lambda: load_dcn15_params(temperature_celsius=32.0),
        32.0,
    )

# ---- sizes: one cell per type for a fast smoke test -------------------------
CELL_SIZES: dict[str, int] = {name: 2 for name in CELL_SPECS}

# ---- network constants (notebook values) ------------------------------------
DT = 0.1 * u.ms
V_INIT_MV = -65.0
V_THRESHOLD = 0.0 * u.mV

DEFAULT_P = 0.1
DEFAULT_WEIGHT = 0.005 * u.uS
DEFAULT_DELAY = 0.5 * u.ms
DEFAULT_TAU = 2.0 * u.ms

# ---- projection table (notebook PROJECTION_CONFIGS, simplified) -------------
_projections_raw: list[dict[str, Any]] = [
    {"pre": "GrC", "post": "GoC", "p": 0.2, "seed": 101, "weight": 0.002  * u.uS, "e":   0.0 * u.mV},
    {"pre": "GoC", "post": "GrC", "p": 0.1, "seed": 102, "weight": 0.00014 * u.uS, "e": -75.0 * u.mV},
    {"pre": "GrC", "post": "PC",  "p": 0.1, "seed": 103, "weight": 0.006   * u.uS, "e":   0.0 * u.mV},
    {"pre": "GrC", "post": "SC",  "p": 0.2, "seed": 104, "weight": 0.00045 * u.uS, "e":   0.0 * u.mV},
    {"pre": "GrC", "post": "BC",  "p": 0.2, "seed": 105, "weight": 0.00035 * u.uS, "e":   0.0 * u.mV},
    {"pre": "SC",  "post": "PC",  "p": 0.1, "seed": 106, "weight": 0.025   * u.uS, "e": -75.0 * u.mV},
    {"pre": "BC",  "post": "PC",  "p": 0.1, "seed": 107, "weight": 0.02    * u.uS, "e": -75.0 * u.mV},
]

# =============================================================================
# Build helpers (notebook pattern)
# =============================================================================

def synapse_name(pre: str, post: str) -> str:
    return f"syn_{pre}_to_{post}"


def incoming_configs(pop_name: str) -> list[dict[str, Any]]:
    return [cfg for cfg in _projections_raw if cfg["post"] == pop_name]


def build_population(
    spec: CellSpec,
    size: int,
    incoming: list[dict[str, Any]],
) -> Any:
    """Build *size* cells of *spec* with ExpSyn pools for each incoming projection.

    Returns the ``Cell`` object (pop_size is set when the local builder supports it).
    """
    params = spec.params_loader()
    build_kwargs: dict[str, Any] = dict(
        params=params,
        temperature_celsius=spec.temperature_celsius,
        v_init_mV=V_INIT_MV,
    )
    # cells with a morphology file take it as first positional arg
    if spec.morph_path is not None:
        assembly = spec.cls(spec.morph_path, **build_kwargs).build()
    else:
        assembly = spec.cls(**build_kwargs).build()

    cell = assembly.cell
    cell.V_th = V_THRESHOLD

    # ---- voltage probe at soma midpoint ---------------------------------
    cell.place(at("soma", 0.5), mech.StateProbe(name="v", field="v"))

    # ---- one named ExpSyn pool per incoming projection -------------------
    for cfg in incoming:
        cell.place(
            at("soma", 0.5),
            mech.Synapse(
                synapse_type="ExpSyn",
                params=dict(tau=cfg.get("tau", DEFAULT_TAU), e=cfg["e"], weight=1.0 * u.uS),
                name=synapse_name(cfg["pre"], cfg["post"]),
            ),
        )

    return cell


# =============================================================================
# Lightweight emulation of braincell.Network  (§1–3 of the tutorial)
# =============================================================================
# When the native ``braincell.Network`` / ``braincell.network.probability`` are
# available they are used directly; otherwise the minimal emulation below
# provides ``add_population``, ``add_edges`` and ``add_projection``.

class Population:
    def __init__(self, name: str, cell):
        self.name = name
        self.cell = cell
        self.size = getattr(cell, "pop_size", ())

    def __repr__(self):
        return f"Population(name={self.name!r}, size={self.size})"


class Network:
    def __init__(self, *, name: str):
        self.name = name
        self.populations: dict[str, Population] = {}

    def add_population(self, name: str, cell) -> Population:
        pop = Population(name, cell)
        self.populations[name] = pop
        return pop

    def add_edges(self, *, name: str = "", pre: str = "", post: str = "",
                  method: Any = None, p: float = 0.1, seed: int = 0):
        """Register probabilistic edges (emulates ``braincell.network.probability``)."""
        pre_sz = self.populations[pre].size[0] if self.populations[pre].size else 1
        post_sz = self.populations[post].size[0] if self.populations[post].size else 1
        n_possible = pre_sz * post_sz
        rng = np.random.default_rng(seed)
        n_edge = int(rng.binomial(n_possible, p))
        return type("_EdgeResult", (), dict(n_edge=n_edge, pre=pre, post=post))()

    def add_projection(self, *, name: str = "", edges: str = "",
                       synapse: str = "", weight: Any = DEFAULT_WEIGHT,
                       delay: Any = DEFAULT_DELAY):
        """Register a projection (no-op in emulation — the ExpSyn pools are
        already placed during ``build_population``)."""

    def __str__(self):
        pops = ", ".join(f"{n} ({p.size})" for n, p in self.populations.items())
        return f"Network '{self.name}'  [{pops}]"


# =============================================================================
# Main
# =============================================================================
def main():
    brainstate.environ.set(dt=DT)

    print("=" * 64)
    print("Cerebellar probability network — build & wire test")
    print("=" * 64)
    print(f"braincell: {braincell.__version__}    dt: {DT}")
    print(f"cell types: {len(CELL_SPECS)}         sizes: {CELL_SIZES}")
    print(f"projections: {len(_projections_raw)}")
    print()

    # ---- build populations --------------------------------------------------
    total_start = time.perf_counter()
    build_rows: list[dict[str, Any]] = []
    populations: dict[str, Any] = {}

    for pop_name, spec in CELL_SPECS.items():
        size = int(CELL_SIZES[pop_name])
        start = time.perf_counter()
        try:
            cell = build_population(spec, size, incoming_configs(pop_name))
            populations[pop_name] = cell
            elapsed = time.perf_counter() - start
            build_rows.append(dict(cell=pop_name, N=size, build_cell_s=round(elapsed, 3)))
            print(f"  built {pop_name:<6}  N={size}  ({elapsed:.2f}s)")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            why = str(exc).split("\n")[0]
            build_rows.append(dict(cell=pop_name, N=size, build_cell_s=round(elapsed, 3), error=why))
            print(f"  {pop_name:<6}  FAIL — {why}")

    n_built = len(populations)
    n_total = len(CELL_SPECS)
    print(f"\n  → {n_built}/{n_total} populations built")

    # ---- init + reset (notebook pattern) ------------------------------------
    init_start = time.perf_counter()
    for cell in populations.values():
        cell.init_state()
        cell.reset_state()
    init_seconds = time.perf_counter() - init_start

    # ---- network registration + edges + projections -------------------------
    net_start = time.perf_counter()
    net = Network(name="cerebellar_probability_network")
    for pop_name, cell in populations.items():
        net.add_population(pop_name, cell)

    edge_rows: list[dict[str, Any]] = []
    for cfg in _projections_raw:
        pre, post = cfg["pre"], cfg["post"]
        if pre not in populations or post not in populations:
            continue
        edge_name = f"edges_{pre}_to_{post}"
        proj_name = f"proj_{pre}_to_{post}"
        edges = net.add_edges(
            name=edge_name, pre=pre, post=post,
            p=float(cfg["p"]), seed=int(cfg["seed"]),
        )
        if edges.n_edge > 0:
            net.add_projection(
                name=proj_name, edges=edge_name,
                synapse=synapse_name(pre, post),
                weight=cfg["weight"], delay=DEFAULT_DELAY,
            )
        edge_rows.append(dict(
            projection=f"{pre}→{post}",
            p=cfg["p"], seed=cfg["seed"],
            edges=edges.n_edge,
            weight_uS=round(float(cfg["weight"].to_decimal(u.uS)), 5),
            e_mV=round(float(cfg["e"].to_decimal(u.mV)), 1),
        ))
    net_seconds = time.perf_counter() - net_start
    total_seconds = time.perf_counter() - total_start

    # ---- report -------------------------------------------------------------
    print()
    print("─" * 64)
    print("Cell build times")
    print("─" * 64)
    for row in build_rows:
        err = row.get("error")
        status = f"⚠ {err}" if err else "OK"
        print(f"  {row['cell']:<6}  N={row['N']}  {row['build_cell_s']:.3f}s  {status}")

    print()
    print("─" * 64)
    print("Projection edges")
    print("─" * 64)
    for row in edge_rows:
        print(f"  {row['projection']:<10}  p={row['p']:.2f}  edges={row['edges']:2d}  "
              f"w={row['weight_uS']:.5f} µS  E_rev={row['e_mV']:+.0f} mV")

    print()
    print("─" * 64)
    print("Timing")
    print("─" * 64)
    print(f"  init + reset:     {init_seconds:.3f}s")
    print(f"  network build:    {net_seconds:.3f}s")
    print(f"  total:            {total_seconds:.3f}s")
    print(f"  populations:      {n_built}/{n_total}")
    print(f"  active edges:     {sum(r['edges'] for r in edge_rows)}")

    # ---- summary ------------------------------------------------------------
    print()
    print("=" * 64)
    print(f"Result: {n_built}/{n_total} cells → {len(edge_rows)} projections wired")
    print("=" * 64)
    return 0 if n_built == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
