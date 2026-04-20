import numpy as np
from jax import grad
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit


# ─────────────────────────────────────────────────────────────────────────────
# Poloidal extent objective (Lp-norm penalty on |theta_in| above a threshold)
# ─────────────────────────────────────────────────────────────────────────────

@jit
def _poloidal_extent_pure(gamma, gammadash, R_winding, Z_winding,
                          theta_target, p):
    """Per-quadpoint Lp penalty on inboard poloidal-angle violation.

    Parameters
    ----------
    gamma : (N, 3) array
        Curve points in Cartesian coordinates.
    gammadash : (N, 3) array
        Curve tangents d(gamma)/ds.
    R_winding, Z_winding : float
        Major and vertical position of the axisymmetric winding-surface axis.
    theta_target : float
        Allowed |theta_in| measured from the inboard midplane (radians).
    p : int
        Lp-norm exponent (>= 2 to keep the objective C^1 at the threshold).
    """
    R = jnp.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]
    # arctan2(Z - Z_w, -(R - R_w)) puts the inboard midplane at theta = 0 and
    # places the (-pi, pi] branch cut at the *outboard* midplane, where banana
    # coils never go.
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

    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        return (self.curve.dgamma_by_dcoeff_vjp(np.asarray(self.dJ_dgamma(g, gd)))
                + self.curve.dgammadash_by_dcoeff_vjp(np.asarray(self.dJ_dgammadash(g, gd))))

    return_fn_map = {'J': J, 'dJ': dJ}
