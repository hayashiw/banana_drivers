"""
Self-intersection penalty (``CurveSelfIntersect``).

Motivation
----------
Banana coils are parameterised as ``CurveCWSFourier`` curves on an
axisymmetric winding surface and optimised with Fourier order as a DOF.
At ``order=2`` the shape space is narrow enough that self-intersecting
("figure-8") geometry is uncommon; from ``order>=3`` onward these
topologies appear routinely and the post-hoc ``is_self_intersecting``
detector in ``banana_coil_solver.py`` flags a non-trivial fraction of
optimizer outputs as invalid. Higher order is desired for a better fit
to the plasma surface; the blocker is a differentiable penalty that
discourages self-intersection during optimisation instead of only
catching it after the fact.

This module provides one: a quadratic-hinge self-distance penalty
with the same mathematical structure as
:class:`simsopt.geo.curveobjectives.CurveCurveDistance`, but with a
single curve and a periodic neighbour-exclusion mask.

Mathematical form
-----------------
Given a closed curve :math:`\\gamma : [0, 1) \\to \\mathbb{R}^3`
discretised on ``N`` quadpoints, the objective is

.. math::
    J = \\frac{C}{2} \\sum_{i, j}
        M_{ij}\\,
        \\bigl\\lVert \\gamma'_i \\bigr\\rVert\\,
        \\bigl\\lVert \\gamma'_j \\bigr\\rVert\\,
        \\max\\!\\bigl(d_{\\min} - \\lVert \\gamma_i - \\gamma_j \\rVert,\\; 0\\bigr)^2,

where :math:`M_{ij} \\in \\{0, 1\\}` is a static index mask that
zeroes out the diagonal and the quadpoints within ``neighbor_skip``
steps of it, wrapping periodically. The factor ``1/2`` compensates
for symmetric double counting. The prefactor :math:`C` is selected
by the ``normalize`` kwarg: ``normalize=True`` sets
:math:`C = 1/N^2` (quadpoint-count invariant but scales the penalty
by :math:`1/N^2` at fixed threshold/violation, which at typical
``N ~ 500`` crushes the term by :math:`\\sim 10^{-6}` relative to
comparable CurveCurveDistance-style penalties), while the default
``normalize=False`` uses :math:`C = 1` so the raw pairwise sum is
returned.

Design notes
------------
* The mask is built once at construction from quadpoint indices and
  held as a constant JAX array. This keeps the JIT trace simple and
  avoids spurious gradient contributions.
* Quadpoint spacing is uniform in curve parameter ``s``, not arc
  length. At high Fourier order, a fixed ``neighbor_skip`` therefore
  corresponds to a variable arc-length window. For the banana-coil
  application this is acceptable because ``neighbor_skip`` is meant
  to suppress adjacency artifacts, not to define a physical minimum
  separation; ``minimum_distance`` carries the physical meaning.
* The ``max(.,0)^2`` hinge is :math:`C^1` at the threshold, so
  :math:`J` is a smooth function of the DOFs (this is what
  ``CurveCurveDistance`` does).
"""

import numpy as np
from jax import grad
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit


def _self_distance_pure(gamma, gammadash, minimum_distance, mask, normalize):
    """Self curve-curve distance penalty, JAX-pure.

    Parameters
    ----------
    gamma : (N, 3) array
        Curve points in Cartesian coordinates.
    gammadash : (N, 3) array
        Curve tangents d(gamma)/ds.
    minimum_distance : float
        Activation threshold :math:`d_{\\min}` on the pairwise
        distance :math:`\\lVert \\gamma_i - \\gamma_j \\rVert`.
    mask : (N, N) array of {0, 1}
        Precomputed constant mask zeroing out the diagonal and the
        within-``neighbor_skip`` band (with periodic wrap).
    normalize : bool
        When ``True``, divide by :math:`N^2` (quadpoint-count invariant
        but scales the penalty down by :math:`1/N^2` at fixed
        threshold/violation). When ``False``, return the raw pairwise
        sum.
    """
    # Squared pairwise distance. The diagonal is identically 0 in exact
    # arithmetic; jnp.sqrt(0) has an infinite subgradient, and even
    # though the mask multiplies those entries by 0 in the forward pass,
    # the VJP would still propagate NaN through the 0*inf product. The
    # standard "double-where" pattern below protects the backward pass
    # on any pair where d^2 = 0 without biasing the forward value.
    dist_sq = jnp.sum((gamma[:, None, :] - gamma[None, :, :]) ** 2, axis=2)
    safe = jnp.where(dist_sq > 0.0, dist_sq, 1.0)
    dists = jnp.where(dist_sq > 0.0, jnp.sqrt(safe), 0.0)
    alen = (jnp.linalg.norm(gammadash, axis=1)[:, None]
            * jnp.linalg.norm(gammadash, axis=1)[None, :])
    viol = jnp.maximum(minimum_distance - dists, 0.0) ** 2
    # 0.5 removes the symmetric double count.
    total = 0.5 * jnp.sum(mask * alen * viol)
    if normalize:
        return total / (gamma.shape[0] ** 2)
    return total


class CurveSelfIntersect(Optimizable):
    r"""
    Penalty that steers a curve away from self-intersecting ("figure-8")
    topology by penalising non-neighbouring quadpoints that come within
    ``minimum_distance`` of each other.

    The implementation mechanism is a self curve-curve distance hinge
    (the single-curve analogue of
    :class:`simsopt.geo.curveobjectives.CurveCurveDistance`), but the
    design intent is self-intersection prevention: the penalty fires
    well before a true crossing forms, giving the optimiser a smooth
    gradient pushing distant-in-parameter points apart. Without it,
    self-intersecting geometry at higher-order banana coils is caught
    only post hoc by ``banana_coil_solver.is_self_intersecting``.

    .. math::
        J = \frac{C}{2} \sum_{i, j}
            M_{ij}\,
            \lVert \gamma'_i \rVert\,
            \lVert \gamma'_j \rVert\,
            \max\!\bigl(d_{\min} - \lVert \gamma_i - \gamma_j \rVert,\; 0\bigr)^2

    with :math:`M_{ij}` the periodic neighbour-exclusion mask
    described below and :math:`C = 1/N^2` when ``normalize=True`` or
    :math:`C = 1` otherwise.

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        Curve to penalise. Must expose ``gamma()``, ``gammadash()``,
        ``dgamma_by_dcoeff_vjp`` and ``dgammadash_by_dcoeff_vjp``
        (e.g. ``CurveXYZFourier`` or ``CurveCWSFourierCPP``).
    minimum_distance : float
        Activation threshold :math:`d_{\min}`. Pairs of quadpoints
        separated by less than this distance contribute a penalty.
    neighbor_skip : int, optional
        Number of nearest-index-neighbour quadpoints to exclude on
        each side (wrapping periodically). Must satisfy
        ``0 <= neighbor_skip < N/2``. Default 3 matches
        ``banana_coil_solver.is_self_intersecting``.
    normalize : bool, optional
        When ``True``, include the :math:`1/N^2` prefactor, making the
        penalty magnitude approximately invariant to the quadpoint
        count but also shrinking it by :math:`1/N^2` relative to the
        raw pairwise sum. When ``False`` (default), return the raw
        sum, so the penalty has the same dimensional scaling as other
        SIMSOPT pairwise distance objectives (e.g.,
        ``CurveCurveDistance``, whose normalization comes from the
        integral measure, not an explicit :math:`1/N^2`).

    Notes
    -----
    * The mask is static (constructed once from quadpoint indices).
    * ``neighbor_skip`` counts indices, not arc length; with
      non-uniform arc-length spacing at higher Fourier order this
      means a variable arc-length exclusion window. The activation
      threshold ``minimum_distance`` is what carries physical
      meaning.
    """

    def __init__(self, curve, minimum_distance, neighbor_skip=3,
                 normalize=False):
        self.curve = curve
        self.minimum_distance = minimum_distance
        self.neighbor_skip = neighbor_skip
        self.normalize = normalize

        N = len(curve.quadpoints)
        if not (0 <= neighbor_skip < N // 2):
            raise ValueError(
                f"neighbor_skip={neighbor_skip} must satisfy "
                f"0 <= neighbor_skip < N/2 = {N // 2}.")
        idx = np.arange(N)
        d = np.abs(idx[:, None] - idx[None, :])
        d = np.minimum(d, N - d)                    # periodic wrap
        mask_np = (d > neighbor_skip).astype(np.float64)
        self._mask_np = mask_np                     # kept for diagnostics
        self._mask = jnp.asarray(mask_np)

        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda g, gd: _self_distance_pure(
            g, gd, minimum_distance, self._mask, normalize))
        self.dJ_dgamma = jit(lambda g, gd: grad(self.J_jax, argnums=0)(g, gd))
        self.dJ_dgammadash = jit(
            lambda g, gd: grad(self.J_jax, argnums=1)(g, gd))

    # ── Diagnostics ────────────────────────────────────────────────────
    def shortest_self_distance(self):
        """Return the minimum pairwise distance over non-masked pairs.

        Pure NumPy; cheap to call from a driver's diagnostics CSV.
        Returns ``+inf`` if the mask is all zeros (degenerate).
        """
        g = self.curve.gamma()
        diff = g[:, None, :] - g[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        d = np.where(self._mask_np > 0, d, np.inf)
        return float(np.min(d))

    # ── Optimizable API ────────────────────────────────────────────────
    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        return (self.curve.dgamma_by_dcoeff_vjp(
                    np.asarray(self.dJ_dgamma(g, gd)))
                + self.curve.dgammadash_by_dcoeff_vjp(
                    np.asarray(self.dJ_dgammadash(g, gd))))

    return_fn_map = {'J': J, 'dJ': dJ}
