"""Physical constants connected with the robot."""

from lib.poses import Pose

SENSOR_POSES_IN_ROBOT_FRAME: list[Pose] = [
    Pose(0, 0.02, 0.0395),  # m
    Pose(0, 0.02, -0.0395),  # m
    Pose(0, -0.02, 0.0395),  # m
    Pose(0, -0.02, -0.0395),  # m
]

WHEEL_BASE_LENGTH = 0.083  # m
ROBOT_DIAMETER = 0.1  # m
