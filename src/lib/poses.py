"""Pose / Transform representation."""

import numpy as np


class Pose:
    """
    This class represents a pose of an object in some reference frame.

    (For instance, pose of a robot's center in the world frame.)
    """

    def __init__(self, angle: float, dx: float, dy: float) -> None:  # rad  # m  # m
        """
        params:
            angle: orientation w.r.t. parent frame
            dx: offset in x direction w.r.t. parent frame
            dy: offset in y direction w.r.t. parent frame
        """

        self.angle = angle
        self.rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        self.translation_vector = np.array([dx, dy])

    def __mul__(self, other: "Pose") -> "Pose":
        """
        Composition of poses (translating between reference frames).

        params:
            other: other pose

        returns:
            new pose
        """

        # classic composition of transforms
        new_translation = (
            self.translation_vector + self.rotation_matrix @ other.translation_vector
        )
        new_angle = self.angle + other.angle
        return Pose(new_angle, new_translation[0], new_translation[1])

    def get_raw_pose(self) -> tuple[float, float, float]:
        """
        Get pose as a tuple of (dx, dy, angle)

        returns:
            (dx, dy, angle)
        """
        return self.translation_vector[0], self.translation_vector[1], self.angle

    def inverse(self) -> "Pose":
        """
        Get an inverse Transform / Pose.

        returns:
            inverse pose
        """

        # these can be computed from the definition of transform composition
        inv_rot = self.rotation_matrix.T
        inv_trans = -inv_rot @ self.translation_vector
        return Pose(-self.angle, inv_trans[0], inv_trans[1])

    def __repr__(self) -> str:
        return f"Rotation:\n{self.rotation_matrix}\nShift:\n{self.translation_vector}\n"
