import jax.numpy as jnp
from simsopt.geo.jit import jit
from jax import jvp

from simsopt.geo import FramedCurve

class FramedCurveCWS(FramedCurve):
    def __init__(self, curve, R0, rotation=None):
        FramedCurve.__init__(self, curve, rotation)
        self.R0 = R0

    def _n_surf(self):
        g = self.curve.gamma()
        x, y, z = g[:, 0], g[:, 1], g[:, 2]
        R = jnp.sqrt(x**2 + y**2)
        n = jnp.stack([(R - self.R0) * x / R,
                       (R - self.R0) * y / R,
                       z], axis=1)
        return n / jnp.linalg.norm(n, axis=1)[:, None]

    def rotated_frame(self):
        return rotated_cws_frame(
            self.curve.gammadash(),
            self._n_surf(),
            self.rotation.alpha(self.curve.quadpoints))

    def rotated_frame_dash(self):
        return rotated_cws_frame_dash(
            self.curve.gammadash(), self.curve.gammadashdash(),
            self._n_surf(),
            self.rotation.alpha(self.curve.quadpoints),
            self.rotation.alphadash(self.curve.quadpoints))


@jit
def rotated_cws_frame(gammadash, n_surf, alpha):
    t = gammadash
    t *= 1./jnp.linalg.norm(gammadash, axis=1)[:, None]
    n = n_surf - jnp.sum(n_surf * t, axis=1)[:, None] * t
    n *= 1./jnp.linalg.norm(n, axis=1)[:, None]
    b = jnp.cross(t, n, axis=1)

    nn = jnp.cos(alpha)[:, None] * n - jnp.sin(alpha)[:, None] * b
    bb = jnp.sin(alpha)[:, None] * n + jnp.cos(alpha)[:, None] * b
    return t, nn, bb


rotated_cws_frame_dash = jit(
    lambda gammadash, gammadashdash, n_surf, alpha, alphadash: jvp(
        rotated_cws_frame,
        (gammadash, n_surf, alpha),
        (gammadashdash, jnp.zeros_like(n_surf), alphadash))[1])