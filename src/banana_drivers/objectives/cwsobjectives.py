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
# TODO: replace with 3D ellipse width

@jit
def _projected_ellipse_width_pure(gamma, gammadash,
                                  R_winding, a_winding, Z_winding,
                                  scale, epsilon):
    R = jnp.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]
    phi = jnp.arctan2(gamma[:, 1], gamma[:, 0])
    theta = jnp.arctan2(Z - Z_winding, -(R - R_winding))

    phi_ref = jnp.arctan2(jnp.mean(jnp.sin(phi)), jnp.mean(jnp.cos(phi)))
    dphi = jnp.mod(phi - phi_ref + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    u = R_winding * dphi
    v = a_winding * theta
    p = jnp.stack([u, v], axis=-1)  # shape (N, 2)

    dl = jnp.linalg.norm(gammadash, axis=-1)
    w = dl / jnp.sum(dl)

    mu = jnp.sum(w[:, None] * p, axis=0)
    pc = p - mu
    cov = (w[:, None] * pc).T @ pc

    a = cov[0, 0]
    b = 0.5 * (cov[0, 1] + cov[1, 0])
    c = cov[1, 1]
    tr = a + c
    disc_sq = jnp.maximum((a - c) ** 2 + 4.0 * b ** 2, 0.0)
    lam_minor = 0.5 * (tr - jnp.sqrt(disc_sq))

    return scale * jnp.sqrt(jnp.maximum(lam_minor, epsilon))

_DEFAULT_SCALE = 2.0 * np.sqrt(2.0)
_DEFAULT_EPSILON = 1.0e-20
class ProjectedEllipseWidth(Optimizable):
    r"""
    Smooth scalar measure of the narrow-direction width of a curve
    projected onto an axisymmetric winding surface.

    The curve is projected onto a winding torus of major radius
    :math:`R_\Sigma` and vertical position :math:`Z_\Sigma`; each
    quadpoint maps to a tangent-plane point :math:`(u_i, v_i)` in
    surface-meter units via the toroidal and poloidal metric
    coefficients. The arclength-weighted 2x2 sample covariance
    :math:`\Sigma` of these points has eigenvalues
    :math:`\lambda_+ \ge \lambda_- \ge 0`. The objective is

    .. math::
        W \;=\; s \, \sqrt{\max(\lambda_-,\, \varepsilon)},

    where :math:`s` is a user-configurable scale factor (default
    :math:`2\sqrt{2}`, chosen so that for an ellipse-boundary
    projection :math:`W` equals the literal narrow diameter), and
    :math:`\varepsilon` is a numerical floor that keeps the square
    root defined at exact collinearity.

    Wrap with :class:`simsopt.objectives.QuadraticPenalty` to enforce
    a minimum (anti-collapse) or maximum (port-fit) bound:

    .. code-block:: python

        from simsopt.objectives import QuadraticPenalty
        Jw = ProjectedEllipseWidth(curve, R_winding=0.976, a_winding=0.210)
        Jmin = QuadraticPenalty(Jw, 0.05, "min")  # don't let it collapse
        Jmax = QuadraticPenalty(Jw, 0.30, "max")  # fits through 30 cm port

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        Curve to evaluate. Must expose ``gamma()``, ``gammadash()``,
        ``dgamma_by_dcoeff_vjp`` and ``dgammadash_by_dcoeff_vjp``.
    R_winding : float
        Major radius of the winding-surface axis :math:`R_\Sigma`,
        in meters.
    a_winding : float
        Minor radius of the winding torus, in meters. Used as the
        poloidal metric coefficient.
    Z_winding : float, optional
        Vertical position of the winding-surface axis :math:`Z_\Sigma`,
        in meters. Default 0.0.
    scale : float, optional
        Scale factor :math:`s` in :math:`W = s\sqrt{\lambda_-}`.
        Default :math:`2\sqrt{2}` (calibrates :math:`W` to the literal
        narrow diameter for an ellipse-boundary projection).
    epsilon : float, optional
        Numerical floor on :math:`\lambda_-`. Default ``1e-20``.

    See Also
    --------
    covariance_ellipse_width.md : full derivation and citation trail.
    """

    def __init__(self, curve, R_winding, a_winding,
                 Z_winding=0.0, scale=_DEFAULT_SCALE, epsilon=_DEFAULT_EPSILON):
        self.curve = curve
        self.R_winding = R_winding
        self.a_winding = a_winding
        self.Z_winding = Z_winding
        self.scale = scale
        self.epsilon = epsilon
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda g, gd: _projected_ellipse_width_pure(
            g, gd, R_winding, a_winding, Z_winding, scale, epsilon))
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
# Self distance objective =====================================================

def _self_distance_pure(gamma, gammadash, minimum_distance, mask, normalize):
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
        d = np.minimum(d, N - d)
        mask_np = (d > neighbor_skip).astype(np.float64)
        self._mask_np = mask_np
        self._mask = jnp.asarray(mask_np)

        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda g, gd: _self_distance_pure(
            g, gd, minimum_distance, self._mask, normalize))
        self.dJ_dgamma = jit(lambda g, gd: grad(self.J_jax, argnums=0)(g, gd))
        self.dJ_dgammadash = jit(
            lambda g, gd: grad(self.J_jax, argnums=1)(g, gd))

    def shortest_self_distance(self):
        g = self.curve.gamma()
        diff = g[:, None, :] - g[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        d = np.where(self._mask_np > 0, d, np.inf)
        return float(np.min(d))

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

# =============================================================================
# Global curvature objective ==================================================

def _project_to_winding_uv(gamma, gammadash, R_winding, a_winding, Z_winding):
    x = gamma[:, 0]
    y = gamma[:, 1]
    z = gamma[:, 2]
    xd = gammadash[:, 0]
    yd = gammadash[:, 1]
    zd = gammadash[:, 2]

    R = jnp.sqrt(x * x + y * y)
    Rd = (x * xd + y * yd) / R

    phi = jnp.arctan2(y, x)
    phid = (x * yd - y * xd) / (R * R)
    phi_ref = jnp.arctan2(jnp.mean(jnp.sin(phi)), jnp.mean(jnp.cos(phi)))
    dphi = jnp.mod(phi - phi_ref + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    a_ = z - Z_winding
    b_ = -(R - R_winding)
    theta = jnp.arctan2(a_, b_)
    thetad = (b_ * zd - a_ * (-Rd)) / (a_ * a_ + b_ * b_)

    u = R_winding * dphi
    v = a_winding * theta
    ud = R_winding * phid
    vd = a_winding * thetad

    p = jnp.stack([u, v], axis=-1)
    pdot = jnp.stack([ud, vd], axis=-1)
    dl_uv = jnp.sqrt(ud * ud + vd * vd)
    return p, pdot, dl_uv

@jit
def _global_radius_curvature_pure(gamma, gammadash,
                                  R_winding, a_winding, Z_winding,
                                  min_curvature_radius, exp_weight):
    p, pdot, dl_uv = _project_to_winding_uv(
        gamma, gammadash, R_winding, a_winding, Z_winding)

    safe_dl = jnp.where(dl_uv > 0.0, dl_uv, 1.0)
    tau = pdot / safe_dl[:, None]

    diff = p[None, :, :] - p[:, None, :]
    dsq = jnp.sum(diff * diff, axis=-1)
    safe_dsq = jnp.where(dsq > 0.0, dsq, 1.0)
    dist = jnp.where(dsq > 0.0, jnp.sqrt(safe_dsq), 0.0)

    dot_pj_tj = jnp.sum(diff * tau[None, :, :], axis=-1)
    cos_safe = dot_pj_tj / jnp.where(dist > 0.0, dist, 1.0)
    n_ratio = 1.0 - cos_safe * cos_safe
    safe_nr = jnp.where(n_ratio > 0.0, n_ratio, 1.0)
    valid = (n_ratio > 0.0) & (dist > 0.0)
    S_C = jnp.where(valid,
                    dist / safe_nr,
                    jnp.full_like(dist, 1.0e12))

    barrier = jnp.exp(-(S_C - min_curvature_radius) / exp_weight)

    N = gamma.shape[0]
    weights = dl_uv[:, None] * dl_uv[None, :]
    return jnp.sum(barrier * weights) / (N * N)

class GlobalRadiusCurvature(Optimizable):
    r"""
    Smooth self-intersection penalty based on the global radius of curvature
    of a CWS curve projected onto the winding-surface tangent plane.

    The 3-D curve is projected to surface-meter coordinates
    :math:`p_i = (R_\Sigma\,(\phi_i - \phi_{\mathrm{ref}}),\,
    a_\Sigma\,\theta_{\mathrm{in},i})`. The objective is the
    arc-length-weighted, exponentially smoothed double integral of the
    Gonzalez--Maddocks self-contact function (ALPOpt's
    ``summed_proximity``, Walker eq. (26)):

    .. math::
        J = \frac{1}{N^{2}} \sum_{i \ne j}
            \exp\!\bigl(-(S_C(p_i, p_j, \hat\tau_j) - R_{\min}) / \varepsilon\bigr)\,
            \,\lVert \dot p_i \rVert \, \lVert \dot p_j \rVert,

    .. math::
        S_C(p_1, p_2, \tau_2) =
            \frac{\lVert p_1 - p_2 \rVert}
                 {1 - \bigl((p_2 - p_1)\cdot \tau_2 /
                            \lVert p_1 - p_2 \rVert\bigr)^{2}}.

    :math:`J` is :math:`C^\infty` in the curve DOFs (the exponential
    barrier smooths the underlying ``min``), accumulates contributions
    from every near-self-contact pair, and acts as a one-sided
    log-barrier on the constraint :math:`\rho_G \ge R_{\min}`. The
    barrier is concentrated where :math:`S_C \approx R_{\min}` and
    decays to zero for safely-separated pairs (rate :math:`1/\varepsilon`).

    Diagnostic methods :meth:`global_curvature_radii` and
    :meth:`min_global_curvature_radius` expose the raw
    ``global_curvature_surface`` quantity for post-hoc inspection; the
    raw minimum is non-smooth and is not recommended as an optimisation
    target.

    Wrap with ``ScaledOptimizable`` (or sum into the scalar objective
    directly) to drive ``J -> 0``. No further wrapping is necessary --
    the threshold is baked into ``min_curvature_radius`` and the barrier
    is already one-sided.

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        Curve to penalise. Must expose ``gamma()``, ``gammadash()``,
        ``dgamma_by_dcoeff_vjp`` and ``dgammadash_by_dcoeff_vjp``.
        For banana-coil use, ``CurveCWSFourierCPP``.
    R_winding : float
        Major radius of the winding-surface axis :math:`R_\Sigma`
        (meters).
    a_winding : float
        Minor radius of the winding torus :math:`a_\Sigma` (meters).
        Used as the poloidal metric coefficient.
    min_curvature_radius : float, optional
        Activation threshold :math:`R_{\min}` (meters). Pairs with
        :math:`S_C < R_{\min}` give barrier > 1 and dominate ``J``;
        pairs with :math:`S_C \gg R_{\min}` contribute negligibly. For
        the banana hardware bound this is the minimum bending radius
        (i.e. ``1 / kappa_max``, currently 0.01 m at
        :math:`\kappa_{\max} = 100\ \mathrm{m}^{-1}`). Default 0.05.
    exp_weight : float, optional
        Barrier softness :math:`\varepsilon` (meters). Smaller values
        mean a sharper (more interior-point-like) barrier and stiffer
        gradients near the threshold; larger values smooth the barrier
        at the cost of allowing more drift past :math:`R_{\min}` before
        the penalty saturates. Default 0.01.
    Z_winding : float, optional
        Vertical position of the winding-surface axis :math:`Z_\Sigma`
        (meters). Default 0.0.

    Notes
    -----
    * The :math:`\rho_G` formulation requires no neighbour-skip mask;
      the geometric factor :math:`1 - \cos^2(\angle)` drives the
      barrier to zero on adjacent quadpoints automatically.
    * For ``CurveCWSFourierCPP`` coils, self-intersection of the
      projected (u, v) curve is equivalent to self-intersection of the
      3-D embedded curve, so the 2-D penalty is a faithful proxy for
      the 3-D problem.
    """

    def __init__(self, curve, R_winding, a_winding,
                 min_curvature_radius=0.05, exp_weight=0.01,
                 Z_winding=0.0):
        self.curve = curve
        self.R_winding = R_winding
        self.a_winding = a_winding
        self.Z_winding = Z_winding
        self.min_curvature_radius = min_curvature_radius
        self.exp_weight = exp_weight

        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda g, gd: _global_radius_curvature_pure(
            g, gd, R_winding, a_winding, Z_winding,
            min_curvature_radius, exp_weight))
        self.dJ_dgamma = jit(lambda g, gd: grad(self.J_jax, argnums=0)(g, gd))
        self.dJ_dgammadash = jit(
            lambda g, gd: grad(self.J_jax, argnums=1)(g, gd))

    def global_curvature_radii(self):
        gamma = np.asarray(self.curve.gamma())
        gammadash = np.asarray(self.curve.gammadash())

        x = gamma[:, 0]
        y = gamma[:, 1]
        z = gamma[:, 2]
        xd = gammadash[:, 0]
        yd = gammadash[:, 1]
        zd = gammadash[:, 2]
        R = np.sqrt(x * x + y * y)
        Rd = (x * xd + y * yd) / R
        phi = np.arctan2(y, x)
        phid = (x * yd - y * xd) / (R * R)
        phi_ref = np.arctan2(np.mean(np.sin(phi)), np.mean(np.cos(phi)))
        dphi = np.mod(phi - phi_ref + np.pi, 2.0 * np.pi) - np.pi
        a_ = z - self.Z_winding
        b_ = -(R - self.R_winding)
        theta = np.arctan2(a_, b_)
        thetad = (b_ * zd - a_ * (-Rd)) / (a_ * a_ + b_ * b_)
        u = self.R_winding * dphi
        v = self.a_winding * theta
        ud = self.R_winding * phid
        vd = self.a_winding * thetad
        dl_uv = np.sqrt(ud * ud + vd * vd)
        tau_u = ud / dl_uv
        tau_v = vd / dl_uv

        du = u[None, :] - u[:, None]
        dv = v[None, :] - v[:, None]
        dsq = du * du + dv * dv
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.sqrt(np.where(dsq > 0.0, dsq, 1.0))
            proj = (du * tau_u[None, :] + dv * tau_v[None, :]) \
                / np.where(dist > 0.0, dist, 1.0)
            n_ratio = 1.0 - proj * proj
            S_C = np.where((n_ratio > 0.0) & (dsq > 0.0),
                           dist / np.where(n_ratio > 0.0, n_ratio, 1.0),
                           1.0e12)
        np.fill_diagonal(S_C, 1.0e12)
        return np.min(S_C, axis=1)

    def min_global_curvature_radius(self):
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