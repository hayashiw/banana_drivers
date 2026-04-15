import numpy as np

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

def find_center(curve):
    gamma = curve.gamma()
    R = np.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]

    idx_bot = Z.argmin()
    idx_mid = np.abs(Z - Z.mean()).argmin()
    idx_top = Z.argmax()

    p1 = np.array([R[idx_bot], Z[idx_bot]])
    p2 = np.array([R[idx_mid], Z[idx_mid]])
    p3 = np.array([R[idx_top], Z[idx_top]])

    v1 = p2 - p1
    v2 = p3 - p2
    A = np.array([v1, v2])
    b = np.array([np.dot(v1, (p1 + p2) / 2), np.dot(v2, (p2 + p3) / 2)])
    center = np.linalg.solve(A, b)
    # Assumes center is at the midplane i.e. center[1] = Zax = 0
    return center

def project_onto_winding_surface(curve, major_radius=None):
    gamma = curve.gamma()
    R = np.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]

    if major_radius is None:
        center = find_center(curve)
        major_radius = center[0]
    Rax = major_radius
    Zax = 0.0

    Reff = R - Rax
    Zeff = Z - Zax
    # The `pi - ` is to set the origin to the inboard midplane with theta
    # increasing in the clockwise direction.
    theta_proj = np.pi - ( np.arctan2(Zeff, Reff) % (2 * np.pi) )
    phi_proj = np.arctan2(gamma[:, 1], gamma[:, 0])
    return theta_proj, phi_proj

def fit_semi_minor_axis(x, y):
    r"""
    Function adapted from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse
    """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    coeffs = np.concatenate((ak, T @ ak)).ravel()

    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]
    numerator = 2*(a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    denominator = b**2 - a*c
    if denominator > 0:
        raise ValueError("The fitted conic is not an ellipse.")
    elif denominator == 0:
        raise ValueError("The fitted conic is a parabola.")
    arg = np.sqrt((a - c)**2 + 4*b**2)
    major_axis = np.sqrt(numerator / denominator / ( arg - a - c))
    minor_axis = np.sqrt(numerator / denominator / (-arg - a - c))
    if minor_axis > major_axis:
        minor_axis, major_axis = major_axis, minor_axis
    return minor_axis

class PoloidalExtent(Optimizable):
    r"""
    Objective function that measures the poloidal extent of a curve on the winding
    surface. The curve is projected onto the winding surface, and the maximum
    absolute value of the poloidal angle is returned as the objective value. The
    poloidal angle is defined such that the inboard midplane corresponds to
    theta = 0, and theta increases in the clockwise direction. The major radius
    of the winding surface can be provided as an argument, or it will be estimated
    from the curve by finding the major radius, assuming that the winding 
    surface is a simple torus with Z = 0 as the midplane.
    """
    def __init__(self, curve, major_radius=None):
        super().__init__(depends_on=[curve])
        self.curve = curve
        if major_radius is None:
            center = find_center(curve)
            major_radius = center[0]
        self.major_radius = major_radius

    def J(self):
        r"""
        Objective value defined as

        .. math::
            J = \max |\theta_\text{proj}|
        where :math:`\theta_\text{proj}` is the poloidal angle of the curve
        projected onto the winding surface defined with respect to the coil
        dofs (:math:`x_0^c, y_0^c, z_0^c, \ldots, x_n^c, y_n^c, z_n^c`) as
        
        .. math::
            \theta_\text{proj} = \pi - \arctan \left( \frac{ ( z_0^c + \ldots + z_n^c ) - Z_\text{ax} }{ \sqrt{ ( x_0^c + \ldots + x_n^c )^2 + ( y_0^c + \ldots + y_n^c )^2 } - R_\text{ax}} \right)

        """
        theta_proj, _ = project_onto_winding_surface(self.curve, self.major_radius)
        abs_max = np.abs(theta_proj).max()
        return abs_max

    @derivative_dec
    def dJ(self):
        r"""
        Analytic derivative of the objective with respect to the curve dofs
        (:math:`\mathcal{C} \in \{ x_0^c, y_0^c, z_0^c, \ldots, x_n^c, y_n^c, z_n^c \}`):

        .. math::
            \frac{\partial J}{\partial \mathcal{C}} &= - \frac{1}{\Theta^2 + 1} \\        
            \Theta &= \frac{ ( z_0^c + \ldots + z_n^c ) - Z_\text{ax} }{ \sqrt{ ( x_0^c + \ldots + x_n^c )^2 + ( y_0^c + \ldots + y_n^c )^2 } - R_\text{ax}}

        """
        # TODO: Implement gradient for PoloidalExtent.
        return 1

    return_fn_map = {'J': J, 'dJ': dJ}


class ProjectedEllipseWidth(Optimizable):
    def __init__(self, curve, major_radius=None):
        super().__init__(depends_on=[curve])
        self.curve = curve
        if major_radius is None:
            center = find_center(curve)
            major_radius = center[0]
        self.major_radius = major_radius

    def J(self):
        theta_proj, phi_proj = project_onto_winding_surface(self.curve, self.major_radius)
        x = phi_proj
        y = theta_proj
        semi_minor_axis = fit_semi_minor_axis(x, y)
        return semi_minor_axis

    @derivative_dec
    def dJ(self):
        r"""
        Analytical derivative of the objective with respect to the curve dofs
        """
        # TODO: Implement gradient for ProjectedEllipseWidth.
        return 1

    return_fn_map = {'J': J, 'dJ': dJ}
