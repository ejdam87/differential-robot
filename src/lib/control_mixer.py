"""
Differential drive control mixer.

This is a work of Martin Skalský (skalsky@mail.muni.cz) 
"""

from __future__ import annotations

from typing import Iterable, Tuple


class ControlMixer:
    """
    Simple WASD-based differential drive mixer.

    - W/S: increase/decrease forward power
    - A/D: increase/decrease rotational power
    - When no W/S pressed, forward power decays toward 0
    - When no A/D pressed, rotational power decays toward 0
    """

    def __init__(
        self, accel: float = 0.05, decay: float = 0.08, max_power: float = 1.0
    ) -> None:
        self.accel = float(accel)
        self.decay = float(decay)
        self.max_power = float(max_power)
        self.v = 0.0  # forward/backward component
        self.w = 0.0  # rotational component

    def _step_axis(
        self, value: float, pos_inc: float, neg_inc: float, decay: float
    ) -> float:
        # Apply input.
        value += pos_inc
        value += neg_inc

        # Decay toward 0 if no input.
        if pos_inc == 0.0 and neg_inc == 0.0:
            if value > 0:
                value = max(0.0, value - decay)
            elif value < 0:
                value = min(0.0, value + decay)

        # Clamp.
        if value > self.max_power:
            value = self.max_power
        if value < -self.max_power:
            value = -self.max_power
        return value

    def get_power(self, keys: Iterable[str]) -> Tuple[float, float]:
        """
        Compute left/right motor powers from key states.

        Args:
            keys: Iterable of pressed keys (e.g. set of strings).

        Returns:
            (left_power, right_power)
        """
        keys = set(keys)
        w = "w" in keys
        s = "s" in keys
        a = "a" in keys
        d = "d" in keys

        # Forward/backward.
        f_inc = self.accel if w and not s else 0.0
        f_dec = -self.accel if s and not w else 0.0
        self.v = self._step_axis(self.v, f_inc, f_dec, self.decay)

        # Rotation.
        r_left = self.accel if a and not d else 0.0
        r_right = -self.accel if d and not a else 0.0
        self.w = self._step_axis(self.w, r_left, r_right, self.decay)

        # Differential mix.
        left = self.v - self.w
        right = self.v + self.w

        # Clamp outputs.
        left = max(-self.max_power, min(self.max_power, left))
        right = max(-self.max_power, min(self.max_power, right))
        return left, right
