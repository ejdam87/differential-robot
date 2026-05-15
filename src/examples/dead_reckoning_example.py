import pygame

from lib.control_mixer import ControlMixer
from lib.keyboard_monitor import KeyboardMonitor
from lib.robot_control import RobotControl
from lib.robot_sim import RobotSimulator
from lib.visualiser import Visualizer

from lib.poses import Pose
from lib.kinematics_and_estimation import deterministic_predict
from lib.measurements import estimate_quantities

if __name__ == "__main__":
    robot_sim = RobotSimulator(
        robot_port=37020, map_path="maps/map_maze.png", pixel_size=0.001
    )
    robot_sim.start()

    monitor = KeyboardMonitor()
    monitor.start()

    robot_control = RobotControl(robot_ip="127.0.0.1", robot_port=37020)
    robot_control.engage()

    vis = Visualizer(window_size=(800, 600), origin=(0.1, 0.1), world_width=3)
    vis.set_background_image("maps/map_maze.png", pixel_size_m=0.001)

    clock = pygame.time.Clock()
    t = 0.0

    control_mixer = ControlMixer()

    previous_measurement: dict = {}
    last_measurement: dict = {}

    x, y, angle = robot_sim.get_pose()
    initial_pose = Pose(angle, x, y)
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

            # --- Visualisation of ground truth provided by simulation.
            x, y, yaw = robot_sim.get_pose()
            vis.draw_pose(
                x=x, y=y, yaw=yaw, text="Robot", color=(255, 0, 0), thickness=3
            )
            vis.draw_text(
                text=f"Pose: {x:.3f}, {y:.3f}, {yaw:.3f}", pos=(10, 10), color=(0, 0, 0)
            )

            # Draw sensors as additional poses with arrow length 0.
            sensors = [(sx, sy, yaw) for (sx, sy) in robot_sim.get_sensor_poses()]
            vis.draw_poses(sensors, size=0.02, arrow_len=0.0, color=(0, 0, 255))
            # ---

            # --- Dead Reckoning Estimation
            quantities = estimate_quantities(
                previous_measurement,
                last_measurement,
                absolute_offset_l,
                absolute_offset_r,
            )
            initial_pose = deterministic_predict(
                initial_pose, quantities["dtheta"], quantities["ds"]
            )
            my_x, my_y, my_angle = initial_pose.get_raw_pose()
            vis.draw_pose(
                x=my_x,
                y=my_y,
                yaw=my_angle,
                text="Robot Est.",
                color=(255, 0, 0),
                thickness=3,
            )
            vis.draw_text(
                text=f"Pose: {my_x:.3f}, {my_y:.3f}, {my_angle:.3f}",
                pos=(400, 10),
                color=(0, 0, 0),
            )
            # ---

            vis.handle_events()
            pygame.display.flip()
            clock.tick(20)
            t += 1.0 / 20.0

    finally:
        robot_control.disengage()
        monitor.stop()
        robot_control.close()
