"""Implements Pure Pursuit path tracker with P(roportional) control."""

from math import atan2
import numpy as np
from numpy.typing import NDArray

from lib.poses import Pose
from lib.types import PlannedPath, SpatialPoint
from robot_constants import WHEEL_BASE_LENGTH


def clip(x: float, min_val: float, max_val: float) -> float:
    """
    Clips given value to given range.

    params:
        x: value to clip
        min_val: minimum value
        max_val: maximum value
    returns:
        clipped value
    """
    return max(min_val, min(max_val, x))


def closest_point_on_segment(p: NDArray, a: NDArray, b: NDArray) -> NDArray:
    """
    Computes a closest point on line segment [a, b] from point p

    params:
        p: point
        a: start of segment
        b: end of segment
    returns:
        closest point
    """

    # vector from a to b
    ab = b - a

    # squared size of the vector
    denom = float(np.dot(ab, ab))

    # single point segment
    if denom == 0.0:
        return a

    # compute perpendicular projection of p onto ab (closest point)
    t = float(np.dot(p - a, ab) / denom)

    # if the projection falls outside given segment, select one of the boundary points
    t = clip(t, 0.0, 1.0)
    projection = a + t * ab

    return projection


class PurePursuitDiffDrive:
    def __init__(
        self,
        path: PlannedPath,
        lookahead_distance: float,
        k_theta: float,
        k_v: float,
    ) -> None:
        """
        Implements pure pursuit path tracking together with P-controller

        params:
            path: planned path
            lookahead_distance: lookahead distance
            k_theta: proportional constant for orientation
            k_v: proportional constant for velocity
        """
        self.path: list[np.ndarray] = [np.array(p, dtype=float) for p in path]
        self.Ld: float = lookahead_distance
        self.k_theta: float = k_theta
        self.k_v: float = k_v

    def find_lookahead_point(self, pose: Pose) -> SpatialPoint:
        """
        Given a current pose, find the lookahead point on the path.

        params:
            pose: current pose
        returns:
            lookahead point
        """
        x, y, _ = pose.get_raw_pose()
        p = np.array([x, y], dtype=float)

        best_dist = float("inf")
        best_proj = None
        best_seg_idx = 0

        # --- Step 1: find closest projection on path ---
        for i in range(len(self.path) - 1):
            a = self.path[i]
            b = self.path[i + 1]

            proj = closest_point_on_segment(p, a, b)
            dist = float(np.linalg.norm(p - proj))

            if dist < best_dist:
                best_dist = dist
                best_proj = proj
                best_seg_idx = i

        if best_proj is None:
            return tuple(self.path[-1])

        # --- Step 2: march forward along path ---
        remaining = self.Ld
        current_point = best_proj
        i = best_seg_idx

        # first segment (partial)
        seg_vec = self.path[i + 1] - current_point
        seg_len = float(np.linalg.norm(seg_vec))

        if seg_len >= remaining and seg_len > 0:
            target = current_point + seg_vec / seg_len * remaining
            return float(target[0]), float(target[1])

        remaining -= seg_len
        i += 1

        # continue through full segments
        while i < len(self.path) - 1:
            seg_vec = self.path[i + 1] - self.path[i]
            seg_len = float(np.linalg.norm(seg_vec))

            if seg_len >= remaining and seg_len > 0:
                target = self.path[i] + seg_vec / seg_len * remaining
                return float(target[0]), float(target[1])

            remaining -= seg_len
            i += 1

        # fallback: end of path
        end = self.path[-1]
        return float(end[0]), float(end[1])

    def compute_control(self, pose: Pose) -> tuple[float, float]:
        """
        For a given pose, compute motion commands for robot motors.

        params:
            pose: current pose
        returns:
            control commands
        """
        x, y, theta = pose.get_raw_pose()
        tx, ty = self.find_lookahead_point(pose)

        dx = tx - x
        dy = ty - y

        theta_target = atan2(dy, dx)

        # normalized error in our orientation (account for periodicity)
        e_theta = theta_target - theta
        e_theta = (e_theta + np.pi) % (2 * np.pi) - np.pi

        # dampen small oscilations
        if abs(e_theta) < 0.05:
            e_theta = 0.0

        # we are at the turn -> do not go forward
        if np.cos(e_theta) < 0.6:  # this value was empirically estimated
            v = 0
            omega = self.k_theta * 2 * e_theta  # increase turning rate in that case
        else:
            v = self.k_v * np.cos(e_theta)  # more aligned -> faster
            v = max(0.0, v)  # forbid going backward
            omega = self.k_theta * e_theta

        # convert to wheel speeds
        u_l = v + (WHEEL_BASE_LENGTH / 2) * omega
        u_r = v - (WHEEL_BASE_LENGTH / 2) * omega

        return clip(u_l, -1.0, 1.0), clip(u_r, -1.0, 1.0)
