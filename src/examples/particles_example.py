import pygame
import numpy as np

from lib.control_mixer import ControlMixer
from lib.keyboard_monitor import KeyboardMonitor
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
)
from lib.map import Map
from lib.measurements import estimate_quantities

# wifi ssid: pb120-robotics
# wifi password: robotisfriend26

if __name__ == "__main__":
    np.random.seed(42)

    REAL_ROBOT = True
    PIXEL_SIZE = 0.001

    N_PARTICLES = 500
    RESAMPLE_THRESHOLD = N_PARTICLES * 0.8
    REFLECT_SENSOR_STD = 0.3
    NOISE_STD = {"dtheta": np.pi / 1000, "ds": 0.002}  # rad  # mm

    robot_ip = "172.24.12.202"
    robot_port = 37020

    if REAL_ROBOT:
        if not robot_ip:
            found = discover_robot()
            if not found:
                print("No robot discovered. Exiting.")

            robot_ip = found["ip"]
            robot_port = found["port"]
            mac = found.get("mac")
            print(f"Discovered robot @ {robot_ip}:{robot_port} (mac={mac})")
    else:
        robot_sim = RobotSimulator(
            robot_port=37020,
            init_pose=(0.3, 0.2, 0),
            map_path="maps/map_maze.png",
            pixel_size=PIXEL_SIZE,
        )
        robot_sim.start()

    monitor = KeyboardMonitor()
    monitor.start()

    if REAL_ROBOT and robot_ip and robot_port:
        robot_control = RobotControl(robot_ip=robot_ip, robot_port=robot_port)
    else:
        robot_control = RobotControl(robot_ip="127.0.0.1", robot_port=37020)

    robot_control.engage()

    vis = Visualizer(window_size=(800, 600), origin=(0.1, 0.1), world_width=3.0)
    vis.set_background_image("maps/map_maze.png", pixel_size_m=PIXEL_SIZE)

    map = Map("maps/map_maze.png", PIXEL_SIZE)

    clock = pygame.time.Clock()
    t = 0.0

    control_mixer = ControlMixer()

    previous_measurement: dict = {}
    last_measurement: dict = {}

    # --- Initialize particle set
    if REAL_ROBOT:
        x, y, angle = 0.3, 0.2, 0
    else:
        x, y, angle = robot_sim.get_pose()

    initial_pose = Pose(angle, x, y)
    particles = initial_particles(initial_pose, N_PARTICLES, NOISE_STD)
    # ---

    absolute_offset_l, absolute_offset_r = None, None

    try:
        while True:

            new_measurements = robot_control.get_measurements()
            if new_measurements:
                previous_measurement = last_measurement
                last_measurement = new_measurements[-1]
            else:
                continue

            left_enc = last_measurement.get("leftEnc", 0)
            right_enc = last_measurement.get("rightEnc", 0)
            reflect = last_measurement.get("reflect", [])

            if not absolute_offset_l:
                absolute_offset_l = left_enc
            if not absolute_offset_r:
                absolute_offset_r = right_enc

            quantities = estimate_quantities(
                previous_measurement,
                last_measurement,
                absolute_offset_l,
                absolute_offset_r,
            )

            _pressed, _released, held = monitor.get_keys()
            left_power, right_power = control_mixer.get_power(held)
            robot_control.set_control(left_power, right_power)

            vis.clean()
            vis.draw_text(
                text=f"leftEnc:{left_enc}, rightEnc:{right_enc}",
                pos=(10, 50),
                color=(255, 0, 0),
            )
            vis.draw_text(text=f"sensors:{reflect}", pos=(10, 90), color=(255, 0, 0))
            vis.draw_text(
                text=f"controls:{[left_power, right_power]}",
                pos=(450, 90),
                color=(255, 0, 0),
            )

            if not REAL_ROBOT:
                # --- Visualisation of ground truth provided by simulation.
                x, y, yaw = robot_sim.get_pose()
                vis.draw_pose(
                    x=x, y=y, yaw=yaw, text="GT", color=(255, 0, 0), thickness=3
                )
                vis.draw_text(
                    text=f"Pose: {x:.3f}, {y:.3f}, {yaw:.3f}",
                    pos=(10, 10),
                    color=(0, 0, 0),
                )

                # Draw sensors as additional poses with arrow length 0.
                sensors = [(sx, sy, yaw) for (sx, sy) in robot_sim.get_sensor_poses()]
                vis.draw_poses(sensors, size=0.02, arrow_len=0.0, color=(0, 0, 255))
                # ---

            # --- Visualisation of single pose estimate
            x, y, yaw = estimate_pose(particles).get_raw_pose()
            vis.draw_pose(
                x=x, y=y, yaw=yaw, text="Estimate", color=(255, 0, 0), thickness=3
            )
            # vis.draw_text(text=f"Pose: {x:.3f}, {y:.3f}, {yaw:.3f}", pos=(10, 10), color=(0, 0, 0))

            vis.draw_particles(particles)
            # ---

            # --- Resampling
            if effective_sample_size(particles) < RESAMPLE_THRESHOLD:
                particles = stochastic_universal_resampling(particles)
            # ---

            # --- Particle Based Estimation
            particles = weighted_update_particles(
                particles,
                quantities["dtheta"],
                quantities["ds"],
                reflect,
                NOISE_STD,
                map,
                REFLECT_SENSOR_STD,
                REAL_ROBOT,
            )
            particles = normalize_weights(particles)
            # ---

            vis.handle_events()
            pygame.display.flip()
            clock.tick(20)
            t += 1.0 / 20.0

    finally:
        robot_control.disengage()
        monitor.stop()
        robot_control.close()
