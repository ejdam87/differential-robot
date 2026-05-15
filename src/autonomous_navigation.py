"""Main script for autonomous navigation presentation."""

import pygame
import numpy as np
import random
from typing import Optional, Any

from lib.robot_control import RobotControl
from lib.robot_sim import RobotSimulator
from lib.visualiser import Visualizer
from lib.resampling import effective_sample_size, stochastic_universal_resampling
from lib.robot_discovery import discover_robot
from lib.poses import Pose
from lib.kinematics_and_estimation import (
    initial_particles,
    normalize_weights,
    weighted_update_particles,
    estimate_pose,
    deterministic_predict,
)
from lib.map import Map
from lib.measurements import estimate_quantities
from lib.RRT.plan_path import obtain_path
from lib.controller import PurePursuitDiffDrive
from lib.types import PlannedPath, Particle

CONFIGURATION: dict[str, Any] = {
    "seed": 42,
    "use_real_robot": False,
    "robot": {"ip": None, "port": 37020},
    "use_particles": True,
    "particles": {
        "n_particles": 500,
        "resample_threshold": 350,
        "reflect_sensor_std": 0.2,
        "noise_std": {"dtheta": np.pi / 1000, "ds": 0.002},
    },
    "show_estimate": True,
    "use_gt_estimate": False,
    "map_file_path": "maps/map_maze.png",
    "dialted_map_file_path": "maps/dilated_map.png",
    "pixel_size": 0.001,  # m / pixel
    "goal_radius": 0.10,  # m
    "look_ahead_distance": 0.20,  # m
    "controller": {
        "real": {"k_v": 0.3, "k_theta": 1.5},
        "sim": {"k_v": 0.5, "k_theta": 3},
    },
}


# -------------------- Setup Functions --------------------


def setup_robot(config: dict[str, Any]) -> tuple[RobotControl, RobotSimulator | None]:
    """
    Given a configuration, create a robot control structures and optionally simulator if enabled

    params:
        config: configuration dictionary
    returns:
        robot_control: robot control structure
        robot_sim: robot simulator structure
    """
    if config["use_real_robot"]:
        if not config["robot"]["ip"]:
            found = discover_robot()
            if not found:
                raise RuntimeError("No robot discovered.")

            config["robot"]["ip"] = found["ip"]
            config["robot"]["port"] = found["port"]

        robot_control = RobotControl(
            robot_ip=config["robot"]["ip"],
            robot_port=config["robot"]["port"],
        )
        robot_sim = None
    else:
        robot_sim = RobotSimulator(
            robot_port=37020,
            init_pose=(0.3, 0.2, np.pi / 2),
            map_path=config["map_file_path"],
            pixel_size=config["pixel_size"],
        )
        robot_sim.start()

        robot_control = RobotControl("127.0.0.1", 37020)

    robot_control.engage()
    return robot_control, robot_sim


def setup_visualization(config: dict[str, Any]) -> Visualizer:
    """
    Setup visualizer

    params:
        config: configuration dictionary
    returns:
        vis: visualizer object
    """
    vis = Visualizer(window_size=(800, 600), origin=(0.1, 0.1), world_width=3.0)
    vis.set_background_image(config["map_file_path"], pixel_size_m=config["pixel_size"])
    return vis


def setup_planner_and_controller(
    config: dict[str, Any],
    start_pose: Pose,
    target_pose: Pose,
    vis: Visualizer,
) -> tuple[PlannedPath, PurePursuitDiffDrive]:
    """
    Setup planner, create a plan, and create controller

    params:
        config: configuration dictionary
        start_pose: start pose
        target_pose: target pose
        vis: visualizer object
    returns:
        path: planned path
        controller: controller object
    """
    path = obtain_path(
        config["map_file_path"],
        start_pose,
        target_pose,
        config["goal_radius"],
        vis,
        dilated_map_path=config["dialted_map_file_path"],
    )

    params = (
        config["controller"]["real"]
        if config["use_real_robot"]
        else config["controller"]["sim"]
    )

    controller = PurePursuitDiffDrive(
        path,
        config["look_ahead_distance"],
        k_theta=params["k_theta"],
        k_v=params["k_v"],
    )

    return path, controller


def setup_particles(
    config: dict[str, Any], initial_pose: Pose
) -> list[Particle] | None:
    """
    Setup particles

    params:
        config: configuration dictionary
        initial_pose: initial pose
    returns:
        particles: list of particles
    """
    if not config["use_particles"]:
        return None

    return initial_particles(
        initial_pose,
        config["particles"]["n_particles"],
        config["particles"]["noise_std"],
    )


# -------------------- Processing Functions --------------------


def process_measurements(
    robot_control: RobotControl,
    prev: dict[str, Any],
    last: dict[str, Any],
    offsets: dict[str, float | None],
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Optional[float]], dict[str, float] | None
]:
    """
    Derives useful quantities from two consecutive measurements

    params:
        robot_control: robot control object
        prev: previous measurement
        last: last measurement
        offsets: initial encoder values
    returns:
        prev: updated previous measurement
        last: updated last measurement
        offsets: updated initial encoder values
        quantities: quantities
    """

    new_measurements = robot_control.get_measurements()
    if not new_measurements:
        return prev, last, offsets, None  # no new quantities

    prev = last
    last = new_measurements[-1]

    left = last.get("leftEnc", 0)
    right = last.get("rightEnc", 0)

    # in the firt iteration, assign offsets
    if offsets["l"] is None:
        offsets["l"] = left
    if offsets["r"] is None:
        offsets["r"] = right

    quantities = estimate_quantities(prev, last, offsets["r"], offsets["l"])

    return prev, last, offsets, quantities


def update_localization(
    config: dict[str, Any],
    particles: list[Particle] | None,
    quantities: dict[str, float],
    reflect: list[float],
    map_obj: Map,
    initial_pose: Pose,
) -> tuple[list[Particle] | None, Pose]:
    """
    Update localization

    params:
        config: configuration dictionary
        particles: list of particles
        quantities: quantities
        reflect: sensor readings
        map_obj: map object
        initial_pose: initial pose (in this step)
    returns:
        particles: updated list of particles
        est_pose: estimated pose
    """

    # choose if to use MCL or deterministic update
    if config["use_particles"] and particles is not None:
        est_pose = estimate_pose(particles)

        # decide if we should resample
        if effective_sample_size(particles) < config["particles"]["resample_threshold"]:
            particles = stochastic_universal_resampling(particles)

        particles = weighted_update_particles(
            particles,
            quantities["dtheta"],
            quantities["ds"],
            reflect,
            config["particles"]["noise_std"],
            map_obj,
            config["particles"]["reflect_sensor_std"],
            config["use_real_robot"],
        )

        particles = normalize_weights(particles)
        return particles, est_pose

    else:
        new_pose = deterministic_predict(
            initial_pose, quantities["dtheta"], quantities["ds"]
        )
        return None, new_pose


def render(
    vis: Visualizer,
    config: dict[str, Any],
    path: PlannedPath | None,
    target_pose: Pose | None,
    est_pose: Pose,
    particles: list[Particle] | None,
    last_measurement: dict[str, Any],
    robot_sim: RobotSimulator | None,
    controls: list[float],
) -> None:
    """
    Render the current situation

    params:
        vis: visualizer object
        config: configuration dictionary
        path: planned path
        target_pose: target pose
        est_pose: estimated pose
        particles: list of particles
        last_measurement: last measurement
        robot_sim: robot simulator object
        controls: control values
    """

    vis.clean()

    # --- Debug text
    left_enc = last_measurement.get("leftEnc", 0)
    right_enc = last_measurement.get("rightEnc", 0)
    reflect = last_measurement.get("reflect", [])

    vis.draw_text(
        text=f"leftEnc:{left_enc}, rightEnc:{right_enc}",
        pos=(10, 50),
        color=(255, 0, 0),
    )
    vis.draw_text(
        text=f"sensors:{reflect}",
        pos=(10, 90),
        color=(255, 0, 0),
    )
    vis.draw_text(
        text=f"controls:{controls}",
        pos=(450, 90),
        color=(255, 0, 0),
    )

    # --- Draw goal
    if target_pose is not None:
        tx, ty, _ = target_pose.get_raw_pose()
        vis.draw_pose(tx, ty, yaw=0, color=(200, 0, 0), thickness=3)

        # --- Draw path
        vis.draw_edges(
            [[(x1, y1), (x2, y2)] for (x1, y1), (x2, y2) in zip(path, path[1:])],
            color=(0, 0, 255),
            thickness=3,
        )
    else:
        vis.draw_text(
            text="Waiting for goal...",
            pos=(350, 50),
            color=(255, 0, 0),
        )

    # --- Ground truth (simulation only)
    if robot_sim is not None and config["use_gt_estimate"]:
        x, y, yaw = robot_sim.get_pose()

        vis.draw_pose(x=x, y=y, yaw=yaw, text="GT", color=(255, 0, 0), thickness=3)
        vis.draw_text(
            text=f"Pose: {x:.3f}, {y:.3f}, {yaw:.3f}",
            pos=(10, 10),
            color=(0, 0, 0),
        )

        sensors = [(sx, sy, yaw) for (sx, sy) in robot_sim.get_sensor_poses()]
        vis.draw_poses(sensors, size=0.02, arrow_len=0.0, color=(0, 0, 255))

    # --- Particles
    if config["use_particles"] and particles is not None:
        vis.draw_particles(particles)

    # --- Estimated pose
    if config["show_estimate"]:
        x, y, yaw = est_pose.get_raw_pose()
        vis.draw_pose(
            x=x,
            y=y,
            yaw=yaw,
            text="Estimate",
            color=(255, 0, 0),
            thickness=3,
        )


def pose_distance(a: Pose | None, b: Pose | None) -> float | None:
    """
    Compute proximity between two poses

    params:
        a: first pose
        b: second pose
    returns:
        proximity: distance between the two poses
    """
    if a is None or b is None:
        return float("inf")

    ax, ay, _ = a.get_raw_pose()
    bx, by, _ = b.get_raw_pose()
    return np.hypot(ax - bx, ay - by)


# -------------------- Main Loop --------------------


def main_loop(
    config: dict[str, Any],
    robot_control: RobotControl,
    robot_sim: RobotSimulator | None,
    vis: Visualizer,
    map_obj: Map,
    particles: list[Particle],
    initial_pose: Pose,
) -> None:
    """
    Main loop of the program.

    params:
        config: configuration dictionary
        robot_control: robot control object
        robot_sim: robot simulation object
        vis: visualizer object
        map_obj: map object
        particles: list of particles
        initial_pose: initial pose
    """

    # store two consecutive measurements
    prev: dict[str, Any] = {}
    last: dict[str, Any] = {}

    # initial encoder values (use as absolute offset)
    offsets: dict[str, Optional[float]] = {"l": None, "r": None}

    clock = pygame.time.Clock()

    # this is interactively chosen
    target_pose = None
    est_pose = initial_pose

    path, controller = None, None
    left, right = 0, 0
    reached_goal = True

    while True:
        vis.handle_events()

        # interactive target selection
        click = vis.consume_target()

        # on new target, replan and stop current control
        if click is not None:
            wx, wy = click
            target_pose = Pose(0, wx, wy)
            robot_control.set_control(0.0, 0.0)
            path, controller = setup_planner_and_controller(
                config,
                est_pose,
                target_pose,
                vis,
            )
            reached_goal = False

        prev, last, offsets, quantities = process_measurements(
            robot_control, prev, last, offsets
        )

        # this prevents applying previous commands if we have not received new data from robot
        if quantities is None:
            render(
                vis,
                config,
                path,
                target_pose,
                est_pose,
                particles,
                last,
                robot_sim,
                [0, 0],
            )
            pygame.display.flip()
            continue

        # map measurements
        reflect = last.get("reflect", [])

        particles, est_pose = update_localization(
            config, particles, quantities, reflect, map_obj, est_pose
        )

        # move only if we have a target that is not reached
        if target_pose is not None and (not reached_goal):
            left, right = controller.compute_control(est_pose)
        else:
            left, right = 0, 0

        # check if we reached a goal already
        if pose_distance(est_pose, target_pose) <= config["goal_radius"]:
            reached_goal = True
            left, right = 0.0, 0.0

        robot_control.set_control(left, right)

        render(
            vis,
            config,
            path,
            target_pose,
            est_pose,
            particles,
            last,
            robot_sim,
            [float(round(left, 2)), float(round(right, 2))],
        )

        pygame.display.flip()
        # time.sleep(0.05)
        clock.tick(20)


# -------------------- Entry Point --------------------


def main() -> None:
    # seed randomness
    np.random.seed(CONFIGURATION["seed"])
    random.seed(CONFIGURATION["seed"])

    # setup robot communication and optionally simulation
    robot_control, robot_sim = setup_robot(CONFIGURATION)

    # setup visualizer
    vis = setup_visualization(CONFIGURATION)

    # setup map object to check expected observations
    map_obj = Map(CONFIGURATION["map_file_path"], CONFIGURATION["pixel_size"])

    # one of the circles oriented into the map
    start_pose = Pose(np.pi / 2, 0.25, 0.25)

    # if particles enabled, generate original population from the start pose
    particles = setup_particles(CONFIGURATION, start_pose)

    try:
        main_loop(
            CONFIGURATION,
            robot_control,
            robot_sim,
            vis,
            map_obj,
            particles,
            start_pose,
        )
    finally:
        robot_control.disengage()
        robot_control.close()


if __name__ == "__main__":
    main()
