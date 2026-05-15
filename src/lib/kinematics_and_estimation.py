"""Contains both deterministic and probabilstic pose estimation."""

from lib.poses import Pose
from lib.map import Map
from lib.types import Particle

from robot_constants import SENSOR_POSES_IN_ROBOT_FRAME

import numpy as np

from math import cos, sin, exp

# --- Predict functions (Given current pose and estimated changes in orientation and displacement, get a new pose)


def naive_predict(current_pose: Pose, dtheta: float, ds: float) -> Pose:
    """
    Given current_pose and change in orientation and displacement, get a new pose in a naive way.

    params:
        current_pose: current pose
        dtheta: change in orientation
        ds: change in displacement
    returns:
        new pose
    """
    dx = ds * cos(current_pose.angle)
    dy = ds * sin(current_pose.angle)

    x, y, angle = current_pose.get_raw_pose()
    return Pose(angle + dtheta, x + dx, y + dy)


def deterministic_predict(current_pose: Pose, dtheta: float, ds: float) -> Pose:
    """
    Given current_pose and change in orientation and displacement, get a new pose in a traditional deterministic way.

    params:
        current_pose: current pose
        dtheta: change in orientation
        ds: change in displacement
    returns:
        new pose
    """

    # assuming constant twist

    # if almost no change in direction, assume no change
    if abs(dtheta) < 1e-6:
        dx = ds
        dy = 0
    # from the kinematics equations
    else:
        r = ds / dtheta
        dx = r * sin(dtheta)
        dy = r * (1 - cos(dtheta))

    return current_pose * Pose(dtheta, dx, dy)


def probabilistic_predict(
    current_pose: Pose,
    dtheta: float,
    ds: float,
    noise_std: dict[str, float],
) -> Pose:
    """
    Given current_pose and change in orientation and displacement and noise parameters, get a new pose in a traditional probabilistic way.

    params:
        current_pose: current pose
        dtheta: change in orientation
        ds: change in displacement
        noise_std: noise parameters for orientation and displacement
    returns:
        new pose
    """
    noisy_dtheta = dtheta + np.random.normal(0, noise_std["dtheta"])
    noisy_ds = ds + np.random.normal(0, noise_std["ds"])

    # basically the same as deterministic_predict, but add noise
    if abs(noisy_dtheta) < 1e-6:
        dx = noisy_ds
        dy = 0
    else:
        r = noisy_ds / noisy_dtheta
        dx = r * sin(noisy_dtheta)
        dy = r * (1 - cos(noisy_dtheta))

    return current_pose * Pose(noisy_dtheta, dx, dy)


# ---


def normalize_weights(particles: list[Particle]) -> list[Particle]:
    """
    Given weight scores for particles, normalize to obtain probability distribution over particle weights.

    params:
        particles: list of particles
    returns:
        normalized particles (weights summing up to 1)
    """
    if len(particles) == 0:
        return []

    total = 0
    for _, _, _, weight in particles:
        total += weight

    if total == 0:
        n = len(particles)
        return [(x, y, a, 1 / n) for x, y, a, _ in particles]

    return [(x, y, angle, weight / total) for x, y, angle, weight in particles]


def initial_particles(
    initial_pose: Pose, n_particles: int, noise_std: dict[str, float]
) -> list[Particle]:
    """
    Given initial pose (generally known), obtain initial population of particles.

    params:
        initial_pose: initial pose
        n_particles: number of particles

    returns:
        list of particles
    """
    particles = []
    for _ in range(n_particles):

        # get noisy versions of original pose
        particle_pose = probabilistic_predict(
            initial_pose,
            0,
            0,
            noise_std,
        )
        my_x, my_y, my_angle = particle_pose.get_raw_pose()
        particles.append(
            (my_x, my_y, my_angle, 1 / n_particles)
        )  # uniform weight initially

    return particles


def weighted_update_particles(
    particles: list[Particle],
    dtheta: float,
    ds: float,
    map_measurements: list[float],
    noise_std: dict[str, float],
    map_obj: Map,
    sensor_std: float,
    real_robot: bool,
) -> list[Particle]:
    """
    Create new generation of particles based on previous generation and new observations.

    params:
        particles: list of particles
        dtheta: change in orientation
        ds: change in displacement
        map_measurements: list of real measurements from four robot sensors (range depeneds on simulation / physical robot)
        noise_std: noise parameters for orientation and displacement
        map_obj: map object
        sensor_std: relative std of sensor readings (estimated)
        real_robot: whether we use real robot or simulated one - range changes

    returns:
        new generation of particles
    """

    new_particles = []
    for px, py, p_angle, weight in particles:

        # --- Motion update (estimate where did this particle move with some error) ---
        particle_pose = probabilistic_predict(
            current_pose=Pose(p_angle, px, py),
            dtheta=dtheta,
            ds=ds,
            noise_std=noise_std,
        )
        n_x, n_y, n_angle = particle_pose.get_raw_pose()
        # ---

        new_weight = weight

        # --- Measurement update (scale weight in accordance with the observations) ---
        if len(map_measurements) > 0:
            likelihood = sensor_model(
                (n_x, n_y, n_angle, weight),
                map_measurements,
                map_obj,
                sensor_std,
                real_robot,
            )
            new_weight = likelihood * weight

        new_particles.append((n_x, n_y, n_angle, new_weight))

    return new_particles


def sensor_model(
    particle: Particle,
    map_measurements: list[float],
    map_obj: Map,
    sensor_std: float,
    real_robot: bool,
) -> float:
    """
    Re-weight particle in accordance with the observations.

    params:
        particle: particle to explore
        map_measurements: real (ground truth) measurements
        map_obj: map ojbect to obtain measurements for this particle
        sensor_std: relative std of sensor readings (estimated)
        real_robot: whether we use real robot or not (range of readings depend on it)

    returns:
        weight of the particle (more aligned with the observations, bigger the weights)
    """

    # use log space to avoid numerical instability
    log_prob = 0.0

    # is a single estimation of robots center (in world coordinates)
    x, y, angle, _ = particle
    particle_pose = Pose(angle, x, y)

    # different scales for simulation and real robot
    scalar = 3 if real_robot else 4
    max_sensor_value = 600 if real_robot else 1020

    # compute likelihood of the particle being the true position (joint on all 4 sensors)
    for i, map_measurement in enumerate(map_measurements):

        s_pose = SENSOR_POSES_IN_ROBOT_FRAME[i]
        world_sx, world_sy, _ = (particle_pose * s_pose).get_raw_pose()

        try:
            # if this particle is right, what would I observe ?
            expected = (
                map_obj.read_value(world_sx, world_sy) * scalar
            )  # match the measurements from the simulation
        except:
            return 0.0  # out of bounds → impossible

        error = (expected - map_measurement) / max_sensor_value
        log_prob += -(error**2) / (2 * sensor_std**2)

    return exp(log_prob) + 0.15  # add constant not to kill by accident


def estimate_pose(particles: list[Particle]) -> Pose:
    """
    Given list of particles, estimate a single pose from them.

    params:
        particles: list of particles
    returns:
        estimated pose
    """
    weights = np.array([p[3] for p in particles])

    xs = np.array([p[0] for p in particles])
    ys = np.array([p[1] for p in particles])
    angles = np.array([p[2] for p in particles])

    # position estimate (weighted mean)
    x_est = np.sum(xs * weights)
    y_est = np.sum(ys * weights)

    # angle estimate (weighted circular mean)
    sin_sum = np.sum(np.sin(angles) * weights)
    cos_sum = np.sum(np.cos(angles) * weights)
    theta_est = np.arctan2(sin_sum, cos_sum)

    return Pose(theta_est, x_est, y_est)
