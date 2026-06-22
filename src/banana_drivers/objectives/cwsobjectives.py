import numpy as np
from jax import grad
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit

# =============================================================================
# Poloidal extent objective ===================================================

@jit
def _poloidal_extent_pure(gamma, gammadash, R_winding, Z_winding,
                          theta_target, p):
    R = jnp.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]
    theta_in = jnp.arctan2(Z - Z_winding, -(R - R_winding))
    arc_length = jnp.linalg.norm(gammadash, axis=-1)
    excess = jnp.maximum(jnp.abs(theta_in) - theta_target, 0.0)
    return (1.0 / p) * jnp.mean(excess ** p * arc_length)

class PoloidalExtent(Optimizable):
    r"""
    Lp penalty on the poloidal extent of a curve on an axisymmetric winding
    surface, measured from the inboard midplane.

    The curve is projected onto a winding torus of major radius
    :math:`R_\Sigma` and vertical position :math:`Z_\Sigma`. The inboard
    poloidal angle of each quadpoint :math:`i` is

    .. math::
        \theta_{\mathrm{in},i} = \mathrm{atan2}(Z_i - Z_\Sigma,\, -(R_i - R_\Sigma)),

    so that :math:`\theta_{\mathrm{in}} = 0` at the inboard midplane and the
    branch cut of ``atan2`` falls at the outboard midplane. The objective is

    .. math::
        J = \frac{1}{p}\, \frac{1}{N} \sum_{i=1}^{N}
            \max\!\bigl(|\theta_{\mathrm{in},i}| - \theta_{\mathrm{target}},\; 0\bigr)^{p}\,
            \bigl\lVert \gamma'_i \bigr\rVert,

    i.e. an arclength-weighted Lp norm of the threshold violation. This is the
    same per-quadpoint penalty pattern used by
    :class:`simsopt.geo.curveobjectives.LpCurveCurvature`. For ``p >= 2`` the
    integrand is :math:`C^1` at the threshold, so :math:`J` is a smooth
    function of the curve DOFs.

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        The curve to penalise. Any Curve subclass that exposes ``gamma()``,
        ``gammadash()``, ``dgamma_by_dcoeff_vjp`` and
        ``dgammadash_by_dcoeff_vjp`` is supported (e.g. ``CurveXYZFourier``,
        ``CurveCWSFourierCPP``).
    R_winding : float
        Major radius of the winding-surface axis :math:`R_\Sigma`.
    theta_target : float
        Allowed poloidal half-extent :math:`\theta_{\mathrm{target}}`
        (radians). Quadpoints with :math:`|\theta_{\mathrm{in}}| \le
        \theta_{\mathrm{target}}` contribute zero.
    p : int, optional
        Lp exponent, default 4 (matches the project's curvature exponent).
    Z_winding : float, optional
        Vertical position of the winding-surface axis :math:`Z_\Sigma`,
        default 0.0.
    """

    def __init__(self, curve, R_winding, theta_target, p=4, Z_winding=0.0):
        self.curve = curve
        self.R_winding = R_winding
        self.Z_winding = Z_winding
        self.theta_target = theta_target
        self.p = p
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda g, gd: _poloidal_extent_pure(
            g, gd, R_winding, Z_winding, theta_target, p))
        self.dJ_dgamma = jit(lambda g, gd: grad(self.J_jax, argnums=0)(g, gd))
        self.dJ_dgammadash = jit(
            lambda g, gd: grad(self.J_jax, argnums=1)(g, gd))

    def poloidal_half_width(self):
        g = self.curve.gamma()
        R = np.linalg.norm(g[:, :2], axis=-1)
        Z = g[:, 2]
        theta_in = np.arctan2(Z - self.Z_winding, -(R - self.R_winding))
        return float(np.max(np.abs(theta_in)))

    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        return (self.curve.dgamma_by_dcoeff_vjp(np.asarray(self.dJ_dgamma(g, gd)))
                + self.curve.dgammadash_by_dcoeff_vjp(np.asarray(self.dJ_dgammadash(g, gd))))

    return_fn_map = {'J': J, 'dJ': dJ}

# =============================================================================
# Ellipse width objective =====================================================

_DEFAULT_SCALE = 2.0 * np.sqrt(2.0)
_DEFAULT_EPSILON = 1.0e-20

@jit
def _ellipse_width_pure(gamma, gammadash, scale, epsilon):
    dl = jnp.linalg.norm(gammadash, axis=-1)
    w = dl / jnp.sum(dl)

    mu = jnp.sum(w[:, None] * gamma, axis=0)
    centered = gamma - mu
    cov = (w[:, None] * centered).T @ centered  # 3x3 weighted covariance

    # Ascending: eigvals[0] = out-of-plane scatter, eigvals[1] = narrow
    # in-plane direction, eigvals[2] = wide in-plane direction. We want
    # the middle eigenvalue; the smallest collapses to zero for a planar
    # curve and the optimizer would game it.
    eigvals = jnp.linalg.eigvalsh(cov)
    lam_narrow = eigvals[1]

    return scale * jnp.sqrt(jnp.maximum(lam_narrow, epsilon))

class EllipseWidth(Optimizable):
    r"""
    Smooth scalar measure of the narrow in-plane width of a curve in
    physical (meter) units.

    The arclength-weighted 3x3 sample covariance :math:`\Sigma` of the
    curve points :math:`\gamma_i \in \mathbb{R}^3` has eigenvalues
    :math:`\lambda_+ \ge \lambda_\text{mid} \ge \lambda_- \ge 0`. For a
    near-planar curve, :math:`\lambda_-` captures only the out-of-plane
    scatter (which goes to zero for an exactly planar curve), while
    :math:`\lambda_\text{mid}` and :math:`\lambda_+` are the narrow and
    wide in-plane principal moments. The objective is

    .. math::
        W \;=\; s \, \sqrt{\max(\lambda_\text{mid},\, \varepsilon)},

    where :math:`s` is a user-configurable scale factor (default
    :math:`2\sqrt{2}`, chosen so that for a planar ellipse-boundary
    curve :math:`W` equals the literal narrow diameter), and
    :math:`\varepsilon` is a numerical floor that keeps the square
    root defined at exact degeneracy.

    The objective is computed directly from the 3D Cartesian
    coordinates of the curve (in meters), so hardware port-fit
    constraints are naturally expressed in physical units and
    :class:`QuadraticPenalty` wrapping this objective bounds the
    actual narrow diameter.

    Wrap with :class:`simsopt.objectives.QuadraticPenalty` to enforce
    a minimum (anti-collapse) or maximum (port-fit) bound:

    .. code-block:: python

        from simsopt.objectives import QuadraticPenalty
        Jw = EllipseWidth(curve)
        Jmin = QuadraticPenalty(Jw, 0.05, "min")  # don't let it collapse
        Jmax = QuadraticPenalty(Jw, 0.30, "max")  # fits through 30 cm port

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        Curve to evaluate. Must expose ``gamma()``, ``gammadash()``,
        ``dgamma_by_dcoeff_vjp`` and ``dgammadash_by_dcoeff_vjp``.
    scale : float, optional
        Scale factor :math:`s` in
        :math:`W = s\sqrt{\lambda_\text{mid}}`. Default
        :math:`2\sqrt{2}` (calibrates :math:`W` to the literal narrow
        diameter for a planar ellipse-boundary curve).
    epsilon : float, optional
        Numerical floor on :math:`\lambda_\text{mid}`. Default
        ``1e-20``.
    """

    def __init__(self, curve, scale=_DEFAULT_SCALE, epsilon=_DEFAULT_EPSILON):
        self.curve = curve
        self.scale = scale
        self.epsilon = epsilon
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda g, gd: _ellipse_width_pure(g, gd, scale, epsilon))
        self.dJ_dgamma = jit(lambda g, gd: grad(self.J_jax, argnums=0)(g, gd))
        self.dJ_dgammadash = jit(
            lambda g, gd: grad(self.J_jax, argnums=1)(g, gd))

    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        return (self.curve.dgamma_by_dcoeff_vjp(np.asarray(self.dJ_dgamma(g, gd)))
                + self.curve.dgammadash_by_dcoeff_vjp(np.asarray(self.dJ_dgammadash(g, gd))))

    return_fn_map = {'J': J, 'dJ': dJ}

# =============================================================================
# Global radius of curvature objective ========================================
@jit
def _global_radius_curvature_pure(gamma, gammadash,
                                     minimum_radius, exp_weight):
    r"""Gonzalez--Maddocks self-contact barrier in 3D Cartesian space."""
    dl = jnp.linalg.norm(gammadash, axis=-1)
    safe_dl = jnp.where(dl > 0.0, dl, 1.0)
    tau = gammadash / safe_dl[:, None]

    diff = gamma[None, :, :] - gamma[:, None, :]  # diff[i, j] = gamma_j - gamma_i
    dsq = jnp.sum(diff * diff, axis=-1)
    safe_dsq = jnp.where(dsq > 0.0, dsq, 1.0)
    dist = jnp.where(dsq > 0.0, jnp.sqrt(safe_dsq), 0.0)

    dot_pj_tj = jnp.sum(diff * tau[None, :, :], axis=-1)
    cos_safe = dot_pj_tj / jnp.where(dist > 0.0, dist, 1.0)
    n_ratio = 1.0 - cos_safe * cos_safe
    safe_nr = jnp.where(n_ratio > 0.0, n_ratio, 1.0)
    valid = (n_ratio > 0.0) & (dist > 0.0)
    S_C = jnp.where(valid, dist / safe_nr, jnp.full_like(dist, 1.0e12))

    barrier = jnp.exp(-(S_C - minimum_radius) / exp_weight)

    N = gamma.shape[0]
    weights = dl[:, None] * dl[None, :]
    return jnp.sum(barrier * weights) / (N * N)


class GlobalRadiusCurvature(Optimizable):
    r"""
    Smooth self-intersection penalty based on the global radius of curvature
    of a curve, computed directly from its 3D Cartesian coordinates.

    The objective is the arc-length-weighted, exponentially smoothed double
    integral of the Gonzalez--Maddocks self-contact function

    .. math::
        J = \frac{1}{N^{2}} \sum_{i \ne j}
            \exp\!\bigl(-(S_C(\gamma_i, \gamma_j, \hat\tau_j) - R_{\min})
                       / \varepsilon\bigr)\,
            \,\lVert \dot\gamma_i \rVert \, \lVert \dot\gamma_j \rVert,

    with

    .. math::
        S_C(p_1, p_2, \tau_2) =
            \frac{\lVert p_1 - p_2 \rVert}
                 {1 - \bigl((p_2 - p_1)\cdot \tau_2 /
                            \lVert p_1 - p_2 \rVert\bigr)^{2}}.

    :math:`J` is :math:`C^\infty` in the curve DOFs (the exponential barrier
    smooths the underlying ``min``), accumulates contributions from every
    near-self-contact pair, and acts as a one-sided log-barrier on the
    constraint :math:`\rho_G \ge R_{\min}`. The
    :math:`1 - \cos^2(\angle)` factor makes the barrier vanish on
    adjacent-quadpoint pairs automatically, so no neighbour-skip mask is
    required.

    Diagnostic methods :meth:`global_curvature_radii` and
    :meth:`shortest_radius` expose the raw global radius of
    curvature for post-hoc inspection; the raw minimum is non-smooth and is
    not recommended as an optimisation target.

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        Curve to penalise. Must expose ``gamma()``, ``gammadash()``,
        ``dgamma_by_dcoeff_vjp`` and ``dgammadash_by_dcoeff_vjp``.
    minimum_radius : float
        Activation threshold :math:`R_{\min}` (meters).
    exp_weight : float, optional
        Barrier softness :math:`\varepsilon` (meters). Default 0.01.
    """

    def __init__(self, curve, minimum_radius, exp_weight=0.01):
        self.curve = curve
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda g, gd: _global_radius_curvature_pure(
            g, gd, minimum_radius, exp_weight))
        self.dJ_dgamma = jit(lambda g, gd: grad(self.J_jax, argnums=0)(g, gd))
        self.dJ_dgammadash = jit(
            lambda g, gd: grad(self.J_jax, argnums=1)(g, gd))

    def global_curvature_radii(self):
        gamma = np.asarray(self.curve.gamma())
        gammadash = np.asarray(self.curve.gammadash())

        dl = np.linalg.norm(gammadash, axis=-1)
        tau = gammadash / dl[:, None]
        diff = gamma[None, :, :] - gamma[:, None, :]
        dsq = np.sum(diff * diff, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.sqrt(np.where(dsq > 0.0, dsq, 1.0))
            proj = np.sum(diff * tau[None, :, :], axis=-1) \
                / np.where(dist > 0.0, dist, 1.0)
            n_ratio = 1.0 - proj * proj
            S_C = np.where((n_ratio > 0.0) & (dsq > 0.0),
                           dist / np.where(n_ratio > 0.0, n_ratio, 1.0),
                           1.0e12)
        np.fill_diagonal(S_C, 1.0e12)
        return np.min(S_C, axis=1)

    def shortest_radius(self):
        return float(np.min(self.global_curvature_radii()))

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
    