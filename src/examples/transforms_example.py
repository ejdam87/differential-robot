import pygame

from lib.visualiser import Visualizer
from lib.poses import Pose
from lib.map import Map

from robot_constants import SENSOR_POSES_IN_ROBOT_FRAME

if __name__ == "__main__":
    robot_pose_in_world = Pose(0.3, 0.2, 0.3)

    vis = Visualizer(window_size=(800, 600), origin=(0.1, 0.1), world_width=3)
    vis.set_background_image("maps/map_maze.png", pixel_size_m=0.001)

    map = Map("maps/map_maze.png", 0.001)

    clock = pygame.time.Clock()

    while True:
        vis.clean()
        x, y, angle = robot_pose_in_world.get_raw_pose()

        vis.draw_text(
            text=f"Pose: {x:.3f}, {y:.3f}, {angle:.3f}", pos=(10, 10), color=(0, 0, 0)
        )
        vis.draw_pose(x=x, y=y, yaw=angle, text="Robot", color=(255, 0, 0), thickness=3)

        for i, sensor_in_robot in enumerate(SENSOR_POSES_IN_ROBOT_FRAME):
            sx, sy, sangle = (robot_pose_in_world * sensor_in_robot).get_raw_pose()
            vis.draw_pose(
                x=sx,
                y=sy,
                yaw=sangle,
                text=f"{i}",
                color=(0, 255, 0),
                thickness=2,
                size=0.01,
            )
            my_intensity = map.read_value(sx, sy)
            vis.draw_text(
                text=f"{i} {my_intensity * 4}", pos=((i + 1) * 200, 50), color=(0, 0, 0)
            )

        vis.handle_events()

        pygame.display.flip()
        clock.tick(60)
