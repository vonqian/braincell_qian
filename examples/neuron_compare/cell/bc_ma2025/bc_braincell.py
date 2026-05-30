from __future__ import annotations

"""BrainCell version of the Basket Cell assembly.

This file mirrors ``bc_neuron.py``.  See ``pc_braincell.py`` for a detailed
explanation of the translation conventions used here.

Key differences from the PC model:

- The BC has four morphologically distinct compartments: **soma**, **dendrites**,
  **AIS** (the first axon branch), and **axon** (all remaining axon branches).
- All cable properties (Ra = 122 Ω·cm, cm = 1 µF/cm²) are uniform across regions.
- Leak conductance differs per region: soma = 4×10⁻⁵, dend/AIS = 1×10⁻⁵,
  axon = 1×10⁻⁶ S/cm².
- Calcium dynamics use ``CdpStC_MA2025_BC`` (the BC-specific pump/concentration
  mechanism) instead of the PC ``CdpCAM_MA2024_PC``.
- Cav2.1 is a permeability-based (GHK) channel; all other channels are
  conductance-based.

Channel → region mapping (from the BC literature / ``bc_neuron.py``):

========= ====== ====== ===== =====
Channel   Soma   Dend   AIS   Axon
========= ====== ====== ===== =====
Leak      ✔      ✔      ✔     ✔
Nav1.1    ✔      ✘      ✘     ✘
Nav1.6    ✘      ✘      ✔     ✔
Cav3.2    ✔      ✔      ✘     ✘
Cav1.2    ✔      ✔      ✘     ✘
Cav1.3    ✔      ✔      ✘     ✘
Cav2.1    ✘      ✘      ✔     ✔
Kir2.3    ✔      ✘      ✘     ✘
Kv3.4     ✔      ✘      ✔     ✔
Kv4.3     ✔      ✔      ✘     ✘
Kv1.1     ✘      ✘      ✘     ✔
Kca3.1    ✔      ✘      ✘     ✘
Kca2.2    ✘      ✔      ✘     ✘
Kca1.1    ✘      ✘      ✔     ✔
HCN1      ✔      ✘      ✔     ✔
Ca Pump   ✔      ✔      ✔     ✔
========= ====== ====== ===== =====
"""

from pathlib import Path
from typing import Any

import brainunit as u
from braincell import Cell, Morphology, mech
from braincell._discretization.policy import MaxCVLen
from braincell.filter import AllRegion, BranchSlice, branch_in

from .parameters import (
    CAV12_DEND,
    CAV12_SOMA,
    CAV13_DEND,
    CAV13_SOMA,
    CAV21_AIS_PERM,
    CAV21_AXON_PERM,
    CAV32_DEND,
    CAV32_SOMA,
    CDP_PUMP,
    CM_UF_CM2,
    CV_MAX_LEN_UM,
    DEFAULT_MORPH_PATH,
    H_E_MV,
    HCN1_AIS,
    HCN1_AXON,
    HCN1_SOMA,
    K_E_MV,
    KCA11_AIS,
    KCA11_AXON,
    KCA22_DEND,
    KCA31_SOMA,
    KIR23_SOMA,
    KV11_AXON,
    KV34_AIS,
    KV34_AXON,
    KV34_SOMA,
    KV43_DEND,
    KV43_SOMA,
    LEAK_E_MV,
    LEAK_G_AIS_MS_CM2,
    LEAK_G_AXON_MS_CM2,
    LEAK_G_DEND_MS_CM2,
    LEAK_G_SOMA_MS_CM2,
    NA_E_MV,
    NAV11_SOMA,
    NAV16_AIS,
    NAV16_AXON,
    RA_OHM_CM,
)


def _find_first_axon_branch(morpho: Morphology) -> int | None:
    for i, branch in enumerate(morpho.branches):
        if branch.type == "axon":
            return i
    return None


class BC:
    """Basket Cell model translated from ``bc_neuron.py`` into BrainCell."""

    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        *,
        temperature_celsius: float = 36.0,
        v_init_mV: float = -65.0,
    ):
        self.morph_path = Path(morph_path)
        self.temperature_celsius = float(temperature_celsius)
        self.v_init_mV = float(v_init_mV)
        self.morpho: Morphology | None = None
        self.cell: Cell | None = None
        self.regions: dict[str, Any] = {}


    def build(self) -> "BC":
        self.morpho = Morphology.from_asc(self.morph_path)

        # ``MaxCVLen(..., keep_odd=True)`` matches the NEURON rule:
        #
        #   sec.nseg = 1 + 2 * int(sec.L / CV_MAX_LEN_UM)
        self.cell = Cell(
            self.morpho,
            cv_policy=MaxCVLen(CV_MAX_LEN_UM * u.um, keep_odd=True),
            V_init=self.v_init_mV * u.mV,
            solver="staggered",
        )
        self._define_regions()
        self._paint_cable()
        self._paint_ions()
        self._paint_channels()
        return self

    # ------------------------------------------------------------------
    # Region definitions
    # ------------------------------------------------------------------

    def _define_regions(self) -> None:
        """Build the four compartment regions used by BC.

        In ``bc_neuron.py``:
        - ``self.soma[0]`` is the soma,
        - ``self.dend`` lists all dendrite sections,
        - ``self.axon[0]`` is the AIS,
        - ``self.axon[1:]`` is the rest of the axon.
        """
        # Soma and dendrite regions (same as PC)
        dend_region = branch_in("type", "dendrite")

        # AIS = first axon branch
        ais_idx = _find_first_axon_branch(self.morpho)
        if ais_idx is None:
            raise ValueError("No axon branches found in morphology — cannot define AIS.")
        ais_region = BranchSlice(ais_idx, 0.0, 1.0)

        axon_region = branch_in("type", "axon") - ais_region

        self.regions = {
            "soma": branch_in("type", "soma"),
            "dend": dend_region,
            "ais": ais_region,
            "axon": axon_region,
        }



    def _paint_cable(self) -> None:
        if self.cell is None or self.morpho is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        self.cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=LEAK_E_MV * u.mV,
                membrane_capacitance=CM_UF_CM2 * (u.uF / u.cm**2),
                axial_resistivity=RA_OHM_CM * (u.ohm * u.cm),
            ),
        )



    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        temp = u.celsius2kelvin(self.temperature_celsius)
        self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=NA_E_MV * u.mV))
        self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=K_E_MV * u.mV))

        
        for region_name in ("soma", "dend", "ais", "axon"):
            self.cell.paint(
                self.regions[region_name],
                mech.Ion(
                    "CdpStC_MA2025_BC",
                    name="ca",
                    temp=temp,
                    Co=2.0 * u.mM,
                    Ci_initializer=45e-6 * u.mM,
                    TotalPump=CDP_PUMP * (u.mol / u.cm**2),
                ),
            )

    # ------------------------------------------------------------------
    # Channels
    # ------------------------------------------------------------------

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        temp = u.celsius2kelvin(self.temperature_celsius)
        mS_cm2 = u.mS / u.cm**2
        cm_s = u.cm / u.second

        
        self.cell.paint(
            self.regions["soma"],
            mech.Channel("IL", g_max=LEAK_G_SOMA_MS_CM2 * mS_cm2, E=LEAK_E_MV * u.mV),
            mech.Channel("Nav1p1_MA2025_BC", g_max=NAV11_SOMA * mS_cm2, temp=temp),
            mech.Channel("Cav3p2_MA2025_BC", g_max=CAV32_SOMA * mS_cm2, temp=temp),
            mech.Channel("Cav1p2_MA2025_BC", g_max=CAV12_SOMA * mS_cm2, temp=temp),
            mech.Channel("Cav1p3_MA2025_BC", g_max=CAV13_SOMA * mS_cm2, temp=temp),
            mech.Channel("Kir2p3_MA2025_BC", g_max=KIR23_SOMA * mS_cm2, temp=temp),
            mech.Channel("Kv3p4_MA2025_BC", g_max=KV34_SOMA * mS_cm2, temp=temp),
            mech.Channel("Kv4p3_MA2025_BC", g_max=KV43_SOMA * mS_cm2, temp=temp),
            mech.Channel("Kca3p1_MA2025_BC", g_max=KCA31_SOMA * mS_cm2, temp=temp),
            mech.Channel("HCN1_MA2025_BC", g_max=HCN1_SOMA * mS_cm2, E=H_E_MV * u.mV, temp=temp),
        )

       
        self.cell.paint(
            self.regions["dend"],
            mech.Channel("IL", g_max=LEAK_G_DEND_MS_CM2 * mS_cm2, E=LEAK_E_MV * u.mV),
            mech.Channel("Cav3p2_MA2025_BC", g_max=CAV32_DEND * mS_cm2, temp=temp),
            mech.Channel("Cav1p2_MA2025_BC", g_max=CAV12_DEND * mS_cm2, temp=temp),
            mech.Channel("Cav1p3_MA2025_BC", g_max=CAV13_DEND * mS_cm2, temp=temp),
            mech.Channel("Kv4p3_MA2025_BC", g_max=KV43_DEND * mS_cm2, temp=temp),
            mech.Channel("Kca2p2_MA2025_BC", g_max=KCA22_DEND * mS_cm2, temp=temp),
        )

       
        self.cell.paint(
            self.regions["ais"],
            mech.Channel("IL", g_max=LEAK_G_AIS_MS_CM2 * mS_cm2, E=LEAK_E_MV * u.mV),
            mech.Channel("Nav1p6_MA2025_BC", g_max=NAV16_AIS * mS_cm2, temp=temp),
            mech.Channel("Kv3p4_MA2025_BC", g_max=KV34_AIS * mS_cm2, temp=temp),
            mech.Channel("HCN1_MA2025_BC", g_max=HCN1_AIS * mS_cm2, E=H_E_MV * u.mV, temp=temp),
            mech.Channel("Kca1p1_MA2025_BC", g_max=KCA11_AIS * mS_cm2, temp=temp),
            mech.Channel("Cav2p1_MA2025_BC", g_max=CAV21_AIS_PERM * cm_s, temp=temp),
        )

        
        self.cell.paint(
            self.regions["axon"],
            mech.Channel("IL", g_max=LEAK_G_AXON_MS_CM2 * mS_cm2, E=LEAK_E_MV * u.mV),
            mech.Channel("Nav1p6_MA2025_BC", g_max=NAV16_AXON * mS_cm2, temp=temp),
            mech.Channel("Kv3p4_MA2025_BC", g_max=KV34_AXON * mS_cm2, temp=temp),
            mech.Channel("Kv1p1_MA2025_BC", g_max=KV11_AXON * mS_cm2, temp=temp),
            mech.Channel("HCN1_MA2025_BC", g_max=HCN1_AXON * mS_cm2, E=H_E_MV * u.mV, temp=temp),
            mech.Channel("Kca1p1_MA2025_BC", g_max=KCA11_AXON * mS_cm2, temp=temp),
            mech.Channel("Cav2p1_MA2025_BC", g_max=CAV21_AXON_PERM * cm_s, temp=temp),
        )
