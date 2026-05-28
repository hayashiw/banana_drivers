import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import derivative_dec

class ScaledCurrentWrapper(Optimizable):
    """Wrap a ScaledCurrent so QuadraticPenalty can use it (exposes .J()/.dJ()).

    Returns |I| so that ``QuadraticPenalty(..., "max")`` penalizes |I| above
    the threshold.

    Gradient: the banana current is ``ScaledCurrent(Current(1), scale)`` where
    ``scale = current_init`` (~10 kA).  The DOF is the underlying ``Current(1)``
    value ``x``, and the physical current is ``I(x) = scale * x``, so
    ``dI/dx = scale`` (NOT 1).  With ``J(x) = |I(x)|``,

        dJ/dx = sign(scale*x) * scale = sign(I) * scale.

    ``scaled_current.vjp(np.array([1.0]))`` returns ``1 * ∂get_value/∂dofs``
    attached to the correct DOF slot in the graph, which evaluates to
    ``scale`` at the underlying Current DOF.  Multiplying by ``sign(I)``
    gives the full chain-rule-correct gradient.  Writing ``dJ = sign(I)``
    alone would drop the ``scale`` factor and underweight the current
    penalty gradient by ~10^4.
    """

    def __init__(self, scaled_current):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[scaled_current])
        self.scaled_current = scaled_current

    def J(self):
        return abs(self.scaled_current.get_value())

    @derivative_dec
    def dJ(self):
        sign = np.sign(self.scaled_current.get_value())
        return sign * self.scaled_current.vjp(np.array([1.0]))