import numpy as np
from jax import grad
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit

# ─────────────────────────────────────────────────────────────────────────────
# Projected ellipse width (covariance-based, smooth)
#
# Mathematical derivation, citation trail, and rationale for every
# design choice are in covariance_ellipse_width.md (sibling file).
# The implementation below is a literal translation of §6 of that doc;
# any drift between the two should be treated as a bug in the doc or
# the code.
# ─────────────────────────────────────────────────────────────────────────────

# Default for the W = c * sqrt(lambda_-) scale factor. c = 2 * sqrt(2) makes
# W equal the literal narrow diameter of an ellipse-boundary projection;
# see covariance_ellipse_width.md §8 for the rationale.
_DEFAULT_SCALE = 2.0 * np.sqrt(2.0)
_DEFAULT_EPSILON = 1.0e-20


@jit
def _projected_ellipse_width_pure(gamma, gammadash,
                                  R_winding, a_winding, Z_winding,
                                  scale, epsilon):
    """Smooth scalar describing the narrow-direction width of a curve
    projected onto an axisymmetric winding surface.

    Returns ``W = scale * sqrt(max(lambda_-, epsilon))``, where
    ``lambda_-`` is the smaller eigenvalue of the arclength-weighted
    sample covariance of the projected points in tangent-plane meters.
    See covariance_ellipse_width.md for the full derivation.
    """
    # 1. Project to (R, theta, phi) on the winding surface (§3.1).
    R = jnp.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]
    phi = jnp.arctan2(gamma[:, 1], gamma[:, 0])
    theta = jnp.arctan2(Z - Z_winding, -(R - R_winding))

    # 2. Recenter phi at its circular mean to avoid the +/-pi branch cut
    #    of arctan2 (§3.3). The wrap is continuous for any coil whose
    #    total toroidal span is under 360 degrees.
    phi_ref = jnp.arctan2(jnp.mean(jnp.sin(phi)), jnp.mean(jnp.cos(phi)))
    dphi = jnp.mod(phi - phi_ref + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # 3. Tangent-plane coordinates with surface-meter units (§3.2).
    u = R_winding * dphi
    v = a_winding * theta
    p = jnp.stack([u, v], axis=-1)  # shape (N, 2)

    # 4. Arclength weights, sum to 1 (§3.4). Decouples the metric from
    #    SIMSOPT's s-uniform parameterisation.
    dl = jnp.linalg.norm(gammadash, axis=-1)
    w = dl / jnp.sum(dl)

    # 5. Weighted 2x2 covariance, omitting the unbiased prefactor
    #    (§3.4, §4.1) -- any positive scale absorbs into `scale`.
    mu = jnp.sum(w[:, None] * p, axis=0)
    pc = p - mu
    cov = (w[:, None] * pc).T @ pc

    # 6. Closed-form smaller eigenvalue (§2.6, §4.2). The (a-c)^2 + 4b^2
    #    form is manifestly non-negative; the inner jnp.maximum guards
    #    only against subtraction-cancellation roundoff. The defensive
    #    symmetrisation of `b` kills any trailing-bit asymmetry from
    #    floating-point covariance assembly (§6 implementation note).
    a = cov[0, 0]
    b = 0.5 * (cov[0, 1] + cov[1, 0])
    c = cov[1, 1]
    tr = a + c
    disc_sq = jnp.maximum((a - c) ** 2 + 4.0 * b ** 2, 0.0)
    lam_minor = 0.5 * (tr - jnp.sqrt(disc_sq))

    # 7. Scaled square root with epsilon floor for exact collinearity
    #    (§3.5).
    return scale * jnp.sqrt(jnp.maximum(lam_minor, epsilon))


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
