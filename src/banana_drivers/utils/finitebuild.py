import numpy as np

from simsopt.geo import CurveFilament, FrameRotation, ZeroRotation

from .framedcurve import FramedCurveCWS

def create_cws_multifilament_grid(
    curve,
    R0,
    numfilaments_n,
    numfilaments_b,
    offset_from_horizontal,
    offset_from_vertical,
    horizontal_spacing,
    rotation_order=None,
    rotation_scaling=None
):
    shifts_n = np.array([-1, 1]) * offset_from_horizontal
    shifts_b = np.arange(numfilaments_b)-numfilaments_b//2
    shifts_b = (np.sign(shifts_n)*offset_from_vertical)[:, None] + \
        (shifts_b*horizontal_spacing)[None, :]

    if rotation_scaling is None:
        rotation_scaling = 1/max(offset_from_horizontal, offset_from_vertical)
    if rotation_order is None:
        rotation = ZeroRotation(curve.quadpoints)
    else:
        rotation = FrameRotation(
            curve.quadpoints, rotation_order, scale=rotation_scaling)

    framedcurve = FramedCurveCWS(curve, R0, rotation=rotation)
    filaments = []
    for i in range(numfilaments_n):
        for j in range(numfilaments_b):
            filaments.append(CurveFilament(
                framedcurve, shifts_n[i], shifts_b[i, j]
            ))
    return filaments