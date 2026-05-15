"""Quantity estimation."""

from typing import Any

from robot_constants import WHEEL_BASE_LENGTH


def estimate_quantities(
    measurements_1: dict[str, Any],
    measurements_2: dict[str, Any],
    absolute_offset_r: float,
    absolute_offset_l: float,
) -> dict[str, float]:
    """
    Given two successive robot measurements, together with initial encoder values, compute usefull quantities.

    params:
        measurements_1: first measurement
        measurements_2: second measurement
        absolute_offset_r: initial right encoder value
        absolute_offset_l: initial left encoder value
    returns:
        quantities: computed quantities
    """

    # difference in left motor displacement
    dsl = measurements_2.get("leftEnc", absolute_offset_l) - measurements_1.get(
        "leftEnc", absolute_offset_l
    )
    dsl /= 1000.0  # convert to meters

    # difference in right motor displacement
    dsr = measurements_2.get("rightEnc", absolute_offset_r) - measurements_1.get(
        "rightEnc", absolute_offset_r
    )
    dsr /= 1000.0  # convert to meters

    # change in angle estimation
    dtheta = (dsl - dsr) / WHEEL_BASE_LENGTH

    # overal displacement estimation
    ds = (dsl + dsr) / 2

    # velocity estimation
    dt = measurements_2.get("timestamp", 0) - measurements_1.get("timestamp", 0)
    v = ds / max(dt, 1e-6)

    return {
        "dsl": dsl,
        "dsr": dsr,
        "dtheta": dtheta,
        "ds": ds,
        "v": v,
    }
