#!/usr/bin/env python3
r"""Distance-dependent connection probability for neuronal networks.

Provides three kernels that map soma-to-soma Euclidean distance to connection
probability — all **negatively correlated**: greater separation → lower (or
zero) probability.  Also provides a :class:`SpatialPopulation` abstraction
where every neuron carries a 3-D soma coordinate.

Kernels
-------
.. list-table::
   :header-rows: 1

   * - Kernel
     - Formula :math:`p(d)`
     - Parameter
   * - :class:`StepConnection`
     - :math:`p_{\max}` if :math:`d \le d_{\text{thresh}}`, else 0
     - ``d_threshold``
   * - :class:`GaussianConnection`
     - :math:`p_{\max} \cdot \exp\bigl(-d^2 / (2\sigma^2)\bigr)`
     - ``sigma``
   * - :class:`ExponentialConnection`
     - :math:`p_{\max} \cdot \exp(-d / \lambda)`
     - ``lambda_``

All kernels inherit from :class:`BaseConnection` and expose an identical API:

* ``p_distance(d)`` — evaluate :math:`p` at one or more distances.
* ``probability_matrix(pre_pos, post_pos)`` — (M, N) pairwise matrix.
* ``generate_connections(pre_pos, post_pos, rng=)`` — stochastic draw
  returning ``(pre_idx, post_idx)`` index arrays.

All distance parameters **require** ``brainunit.Quantity`` units
(e.g. ``80.0 * u.um``).  The dimensionless probability ``p_max`` is a
plain ``float`` in (0, 1].

Examples
--------

.. code-block:: python

    import brainunit as u
    from distance_connectivity import (
        SpatialPopulation,
        GaussianConnection,
        ExponentialConnection,
        StepConnection,
        create_random_positions,
        make_kernel,
    )

    # two populations with random 3-D soma positions
    grc = SpatialPopulation("GrC", create_random_positions(50))
    goc = SpatialPopulation("GoC", create_random_positions(20))

    # build a kernel and draw connections  (units required)
    kernel = GaussianConnection(sigma=80.0 * u.um, p_max=0.3)
    pre_idx, post_idx = kernel.generate_connections(
        grc.positions, goc.positions
    )

    # or use the factory
    kernel = make_kernel("exponential", lambda_=120.0 * u.um, p_max=0.25)

    # evaluate p(d) directly
    import numpy as np
    d = np.linspace(0, 300, 100) * u.um
    p = kernel.p_distance(d)              # shape (100,)
    p_single = kernel.p_distance(50.0 * u.um)  # scalar
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import brainunit as u
import numpy as np
from brainunit import Quantity


__all__ = [
    # base
    "BaseConnection",
    "DistanceConnectionRule",
    # kernels
    "StepConnection",
    "GaussianConnection",
    "ExponentialConnection",
    # spatial
    "SpatialPopulation",
    "create_random_positions",
    "pairwise_euclidean_distance",
    # convenience
    "make_kernel",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _to_um_float(value: Quantity) -> float:
    """Convert a ``brainunit.Quantity`` length to a plain ``float`` in µm."""
    return float(value.to_decimal(u.um))


def _to_um_array(value: Quantity) -> np.ndarray:
    """Convert a ``brainunit.Quantity`` length (scalar or array) to µm floats."""
    return np.asarray(value.to_decimal(u.um), dtype=float)


def _resolve_rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    """Return *rng* if given, otherwise a fresh ``default_rng``."""
    return rng if rng is not None else np.random.default_rng()


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol (for duck-typed kernels)
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class DistanceConnectionRule(Protocol):
    """Structural protocol for distance-dependent connection rules.

    Any object implementing ``p_distance``, ``probability_matrix``, and
    ``generate_connections`` satisfies this interface.
    """

    def p_distance(self, d: Quantity) -> np.ndarray: ...

    def probability_matrix(
        self, pre_pos: np.ndarray, post_pos: np.ndarray
    ) -> np.ndarray: ...

    def generate_connections(
        self,
        pre_pos: np.ndarray,
        post_pos: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...


# ═══════════════════════════════════════════════════════════════════════════════
# Distance utilities
# ═══════════════════════════════════════════════════════════════════════════════


def pairwise_euclidean_distance(
    pre_pos: np.ndarray,
    post_pos: np.ndarray,
) -> np.ndarray:
    """All-pairs Euclidean (L₂) distance between two sets of soma positions.
    Parameters
    pre_pos : (M, 3) array
        Pre-synaptic soma coordinates in µm.
    post_pos : (N, 3) array
        Post-synaptic soma coordinates in µm.

    Returns
    (M, N) array
        ``d[i, j] = ‖pre_pos[i] − post_pos[j]‖₂``
    """
    diff = pre_pos[:, None, :] - post_pos[None, :, :]  # (M, N, 3)
    return np.linalg.norm(diff, axis=-1)


def create_random_positions(
    size: int,
    bounds: (
        tuple[Quantity, Quantity]
        | tuple[
            tuple[Quantity, Quantity],
            tuple[Quantity, Quantity],
            tuple[Quantity, Quantity],
        ]
    ) = (-200.0 * u.um, 200.0 * u.um),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate random 3-D soma positions uniformly within *bounds*.
    (size, 3) array of (x, y, z) soma coordinates in µm.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # detect flat (lo, hi) vs per-axis ((x_lo,x_hi), (y_lo,y_hi), (z_lo,z_hi))
    if isinstance(bounds[0], Quantity):
        # single (lo, hi) → cube
        lo = _to_um_float(bounds[0])
        hi = _to_um_float(bounds[1])
        return rng.uniform(lo, hi, size=(size, 3))

    # per-axis — each element must be a (lo, hi) Quantity pair
    if not (len(bounds) == 3 and all(isinstance(b, tuple) for b in bounds)):
        raise TypeError(
            f"bounds must be (lo, hi) or ((x_lo,x_hi), (y_lo,y_hi), (z_lo,z_hi)) "
            f"with brainunit.Quantity values, got {bounds!r}"
        )

    positions = np.empty((size, 3), dtype=float)
    for axis, (lo_q, hi_q) in enumerate(bounds):
        positions[:, axis] = rng.uniform(
            _to_um_float(lo_q), _to_um_float(hi_q), size=size
        )
    return positions





@dataclass
class SpatialPopulation:
    """A neuronal population where every soma has a 3-D spatial coordinate.

    Parameters
    ----------
    name:
        Population label (e.g. ``"GrC"``, ``"PC"``).
    positions:
        Array of shape ``(size, 3)`` — soma (x, y, z) coordinates in µm.
    """

    name: str
    positions: np.ndarray  # (size, 3) in µm

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (size, 3), got {self.positions.shape}"
            )

    # -- convenience properties ------------------------------------------------

    @property
    def size(self) -> int:
        """Number of neurons in the population."""
        return self.positions.shape[0]

    @property
    def x(self) -> np.ndarray:
        """X-coordinates of all somata (µm)."""
        return self.positions[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Y-coordinates of all somata (µm)."""
        return self.positions[:, 1]

    @property
    def z(self) -> np.ndarray:
        """Z-coordinates of all somata (µm)."""
        return self.positions[:, 2]

    def __repr__(self) -> str:
        return f"SpatialPopulation(name={self.name!r}, size={self.size})"


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════


class BaseConnection(ABC):
    """Abstract base for distance-dependent connection probability kernels.
    Subclasses implement only :meth:`_p_distance_impl`, which receives raw
    µm float arrays. 
    """

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _p_distance_impl(self, d_um: np.ndarray) -> np.ndarray:
        """Kernel-specific formula operating on raw µm float arrays.

        Parameters
        ----------
        d_um : np.ndarray
            Distance(s) in µm.  May be 0-D (scalar), 1-D, or 2-D.

        Returns
        -------
        np.ndarray
            Connection probability at each distance, same shape as *d_um*.
        """
        ...

    @property
    @abstractmethod
    def kernel_name(self) -> str:
        """Short string identifier for this kernel (e.g. ``"gaussian"``)."""
        ...

    
    def p_distance(
        self,
        d: Quantity,
    ) -> np.ndarray:
        """Connection probability at one or more soma-to-soma distances.

            Connection probability at each distance.  Shape matches the
            input, or is a 0-D array for scalar input.
        """
        d_um = _to_um_array(d)
        return self._p_distance_impl(d_um)

    def probability_matrix(
        self,
        pre_pos: np.ndarray,
        post_pos: np.ndarray,
    ) -> np.ndarray:
        """(M, N) matrix of pairwise connection probabilities.

        Parameters
        ----------
        pre_pos : (M, 3) array
            Pre-synaptic soma coordinates in µm.
        post_pos : (N, 3) array
            Post-synaptic soma coordinates in µm.

        Returns
        -------
        (M, N) np.ndarray
            ``P[i, j]`` = probability that pre-neuron *i* connects to
            post-neuron *j*.
        """
        d = pairwise_euclidean_distance(pre_pos, post_pos)
        return self._p_distance_impl(d)

    def generate_connections(
        self,
        pre_pos: np.ndarray,
        post_pos: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stochastically draw connections from the distance-dependent rule.

        Parameters
        ----------
        pre_pos : (M, 3) array
            Pre-synaptic soma coordinates in µm.
        post_pos : (N, 3) array
            Post-synaptic soma coordinates in µm.
        rng : np.random.Generator, optional
            Random state for reproducibility.

        Returns
        -------
        (pre_idx, post_idx) : tuple of 1-D np.ndarray
            Indices of connected pre- and post-synaptic neurons.  Each
            pair ``(pre_idx[k], post_idx[k])`` is a realised connection.
        """
        p = self.probability_matrix(pre_pos, post_pos)
        rand = _resolve_rng(rng).random(p.shape)
        mask = rand < p
        return np.where(mask)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(...)"  # pragma: no cover




class StepConnection(BaseConnection):
    r"""Step-function (hard-threshold) connection probability.

    .. math::

        p(d) = \begin{cases}
            p_{\max} & \text{if } d \le d_{\text{thresh}} \\
            0        & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    d_threshold:
        Distance cutoff as a ``brainunit.Quantity`` length
        (e.g. ``100.0 * u.um``).  Pairs with
        :math:`d > d_{\text{thresh}}` have zero connection probability.
    p_max:
        Connection probability for pairs within the threshold.
        Must be in [0, 1].
    """

    def __init__(
        self,
        d_threshold: Quantity = 100.0 * u.um,
        p_max: float = 0.3,
    ) -> None:
        self._d_threshold = d_threshold
        self.d_threshold_um = _to_um_float(d_threshold)
        if self.d_threshold_um < 0:
            raise ValueError(f"d_threshold must be >= 0, got {self.d_threshold_um}")
        if not (0 <= p_max <= 1):
            raise ValueError(f"p_max must be in [0, 1], got {p_max}")
        self.p_max = float(p_max)

    @property
    def kernel_name(self) -> str:
        return "step"

    @property
    def d_threshold(self) -> Quantity:
        """Distance cutoff."""
        return self._d_threshold

    def _p_distance_impl(self, d_um: np.ndarray) -> np.ndarray:
        return np.where(d_um <= self.d_threshold_um, self.p_max, 0.0)

    def __repr__(self) -> str:
        return (
            f"StepConnection(d_threshold={self.d_threshold}, "
            f"p_max={self.p_max:.3f})"
        )


class GaussianConnection(BaseConnection):
    r"""Gaussian (squared-exponential) distance-decay connection probability.

    .. math::

        p(d) = p_{\max} \cdot \exp\!\left(-\frac{d^2}{2\sigma^2}\right)

    The probability equals :math:`p_{\max}` at zero distance and decays
    smoothly with increasing soma separation.  At :math:`d = \sigma` the
    probability drops to :math:`p_{\max} \cdot e^{-1/2} \approx
    0.607 \, p_{\max}`.

    Parameters
    ----------
    sigma:
        Width (standard deviation) of the Gaussian kernel as a
        ``brainunit.Quantity`` length (e.g. ``50.0 * u.um``).
    p_max:
        Maximum connection probability at zero distance.
        Must be in [0, 1].
    """

    def __init__(
        self,
        sigma: Quantity = 50.0 * u.um,
        p_max: float = 0.3,
    ) -> None:
        self._sigma = sigma
        self.sigma_um = _to_um_float(sigma)
        if self.sigma_um < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma_um}")
        if not (0 <= p_max <= 1):
            raise ValueError(f"p_max must be in [0, 1], got {p_max}")
        self.p_max = float(p_max)

    @property
    def kernel_name(self) -> str:
        return "gaussian"

    @property
    def sigma(self) -> Quantity:
        """Kernel width."""
        return self._sigma

    def _p_distance_impl(self, d_um: np.ndarray) -> np.ndarray:
        return self.p_max * np.exp(-(d_um**2) / (2.0 * self.sigma_um**2))

    def __repr__(self) -> str:
        return (
            f"GaussianConnection(sigma={self.sigma}, "
            f"p_max={self.p_max:.3f})"
        )


class ExponentialConnection(BaseConnection):
    r"""Exponential distance-decay connection probability.

    .. math::

        p(d) = p_{\max} \cdot \exp\!\left(-\frac{d}{\lambda}\right)

    At :math:`d = \lambda` the probability drops to :math:`p_{\max} \cdot
    e^{-1} \approx 0.368 \, p_{\max}`.
    Parameters
    ----------
    lambda_:
        Length-scale constant as a ``brainunit.Quantity`` length
        (e.g. ``80.0 * u.um``).  Larger values produce longer-range
        connectivity.
    p_max:
        Maximum connection probability at zero distance.
        Must be in [0, 1].
    """

    def __init__(
        self,
        lambda_: Quantity = 80.0 * u.um,
        p_max: float = 0.3,
    ) -> None:
        self._lambda = lambda_
        self.lambda_um = _to_um_float(lambda_)
        if self.lambda_um < 0:
            raise ValueError(f"lambda_ must be >= 0, got {self.lambda_um}")
        if not (0 <= p_max <= 1):
            raise ValueError(f"p_max must be in [0, 1], got {p_max}")
        self.p_max = float(p_max)

    @property
    def kernel_name(self) -> str:
        return "exponential"

    @property
    def lambda_(self) -> Quantity:
        """Length-scale constant."""
        return self._lambda

    def _p_distance_impl(self, d_um: np.ndarray) -> np.ndarray:
        return self.p_max * np.exp(-d_um / self.lambda_um)

    def __repr__(self) -> str:
        return (
            f"ExponentialConnection(lambda_={self.lambda_}, "
            f"p_max={self.p_max:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel registry / factory
# ═══════════════════════════════════════════════════════════════════════════════

_KERNEL_REGISTRY: dict[str, type[BaseConnection]] = {
    "step": StepConnection,
    "gaussian": GaussianConnection,
    "exponential": ExponentialConnection,
}


def make_kernel(kind: str = "gaussian", **kwargs: Any) -> BaseConnection:
    """Factory for distance-dependent connection probability kernels.

    Parameters
    ----------
    kind:
        One of ``"step"``, ``"gaussian"``, or ``"exponential"``.
    **kwargs:
        Forwarded to the kernel constructor (e.g. ``sigma=80.0 * u.um``,
        ``p_max=0.3`` for ``"gaussian"``).

    Returns
    -------
    BaseConnection
        An instance of :class:`StepConnection`, :class:`GaussianConnection`,
        or :class:`ExponentialConnection`.

    Raises
    ------
    ValueError
        If *kind* is not a recognised kernel name.
    """
    cls = _KERNEL_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown kernel kind {kind!r}; choose from {list(_KERNEL_REGISTRY)}"
        )
    return cls(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Smoke-test: kernel curves + stochastic connection statistics."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    # ---- toy populations --------------------------------------------------
    grc = SpatialPopulation("GrC", create_random_positions(60, rng=rng))
    goc = SpatialPopulation("GoC", create_random_positions(25, rng=rng))
    pc = SpatialPopulation("PC", create_random_positions(10, rng=rng))

    print("Populations:")
    for pop in (grc, goc, pc):
        print(
            f"  {pop}  — x∈[{pop.x.min():.0f}, {pop.x.max():.0f}]  "
            f"y∈[{pop.y.min():.0f}, {pop.y.max():.0f}]  "
            f"z∈[{pop.z.min():.0f}, {pop.z.max():.0f}]"
        )

    # ---- kernels ----------------------------------------------------------
    kernels: list[BaseConnection] = [
        StepConnection(d_threshold=100.0 * u.um, p_max=0.4),
        GaussianConnection(sigma=80.0 * u.um, p_max=0.4),
        ExponentialConnection(lambda_=120.0 * u.um, p_max=0.4),
    ]

    # ---- brainunit round-trip check ---------------------------------------
    k_bu = GaussianConnection(sigma=80.0 * u.um, p_max=0.4)
    assert abs(k_bu.sigma_um - 80.0) < 1e-9, "sigma conversion"
    p_bu = k_bu.p_distance(50.0 * u.um)
    assert abs(float(p_bu) - 0.4 * np.exp(-50.0**2 / (2 * 80.0**2))) < 1e-9, "p_distance check"
    print("\n  ✓ brainunit Quantity round-trip OK")

    # ---- per-kernel statistics --------------------------------------------
    print("\n" + "=" * 64)
    print("GrC → GoC  (60 × 25 = 1500 possible pairs)")
    print("=" * 64)

    for kernel in kernels:
        pre_idx, post_idx = kernel.generate_connections(
            grc.positions, goc.positions, rng=rng
        )
        n_conn = len(pre_idx)
        p_avg = n_conn / (grc.size * goc.size)

        if n_conn > 0:
            d_realised = pairwise_euclidean_distance(grc.positions, goc.positions)[
                pre_idx, post_idx
            ]
            d_mean = float(np.mean(d_realised))
            d_min = float(np.min(d_realised))
            d_max = float(np.max(d_realised))
        else:
            d_mean = d_min = d_max = float("nan")

        print(
            f"  {kernel.kernel_name:<12}  edges={n_conn:4d}  "
            f"p_avg={p_avg:.4f}  "
            f"d_mean={d_mean:.1f}µm  d_range=[{d_min:.0f}, {d_max:.0f}]µm"
        )

    # ---- p_distance scalar check ------------------------------------------
    print("\n  p_distance(0) values:")
    for kernel in kernels:
        print(f"    {kernel.kernel_name:<12}  {float(kernel.p_distance(0.0 * u.um)):.3f}")

    # ---- plot: p(d) curve + histogram of realised distances ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    d_span = np.linspace(0, 400, 500)

    for ax, kernel in zip(axes, kernels):
        # theoretical p(d) — now trivially clean with p_distance()
        p_theory = kernel.p_distance(d_span * u.um)

        ax.plot(d_span, p_theory, "k-", lw=1.5, label=r"$p(d)$ theory")

        # histogram of realised distances
        pre_idx, post_idx = kernel.generate_connections(
            grc.positions, goc.positions, rng=rng
        )
        if len(pre_idx) > 0:
            d_real = pairwise_euclidean_distance(grc.positions, goc.positions)[
                pre_idx, post_idx
            ]
            ax.hist(
                d_real,
                bins=40,
                density=True,
                alpha=0.4,
                color="C0",
                label="realised distances",
            )

        ax.set_title(kernel.kernel_name.capitalize())
        ax.set_xlabel("Distance  d  (µm)")
        ax.set_ylabel(r"Connection probability  $p(d)$")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 400)
        ax.set_ylim(-0.02, None)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Distance-dependent connection probability — GrC → GoC",
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
