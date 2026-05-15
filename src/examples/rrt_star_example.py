"""Example of RRT* on a simple binary image map."""

from __future__ import annotations

import argparse
import pygame

from lib.RRT.my_primitives import Robot2DRRTPrimitives
from lib.RRT.rrt_star import RRTStar
from lib.visualiser import Visualizer
from lib.RRT.plan_path import find_best_goal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RRT* demo on a binary occupancy map.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Pause after each refresh and continue on key press.",
    )
    parser.add_argument(
        "-r",
        "--refresh",
        type=int,
        default=10,
        help="Number of RRT* iterations per visualization refresh (default: 10).",
    )
    parser.add_argument(
        "-n",
        "--neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbors used by RRT* rewiring (default: 10).",
    )
    args = parser.parse_args()
    if args.refresh <= 0:
        parser.error("-r/--refresh must be a positive integer.")
    if args.neighbors <= 0:
        parser.error("-n/--neighbors must be a positive integer.")
    return args


def wait_for_keypress() -> bool:
    """Block until a key press; return False if quit was requested."""
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                return False
            return True


def main() -> None:
    """

    :return:
    """
    args = parse_args()

    map_path = "maps/map_maze.png"
    pixel_size = 0.001  # meters per pixel
    goal_radius = 0.15

    start = (0.3, 0.3)
    goal = (1.26, 1.75)

    primitives = Robot2DRRTPrimitives(
        map_path,
        pixel_size=pixel_size,
        spatial_step_size=0.10,
        neighbor_k=args.neighbors,
    )

    rrt = RRTStar(primitives, start)

    vis = Visualizer(
        window_size=(600, 600),
        origin=(0.0, 0.0),
        world_width=2,
    )
    vis.set_background_image(map_path, pixel_size_m=pixel_size)
    clock = pygame.time.Clock()

    iteration = 0
    running = True
    while running:
        vis.handle_events()
        if not pygame.get_init() or not pygame.display.get_init():
            break

        # Grow the tree between visualization refreshes.
        iterations = args.refresh
        x_rands, x_nears, x_news = rrt.iterate(iterations)
        iteration += iterations

        vis.clean()
        vis.draw_edges(rrt.draw_tree(), color=(255, 0, 255), thickness=1)
        near_to_rand = [[x_near, x_rand] for x_near, x_rand in zip(x_nears, x_rands)]
        vis.draw_edges(near_to_rand, color=(160, 160, 160), thickness=1)

        # Attempt to extract a path to a node near the goal
        goal_node = find_best_goal(rrt, (goal[0], goal[1]), goal_radius)
        if goal_node is not None:
            path = rrt.extract_path(goal_node)
            vis.draw_edges(
                [[(x1, y1), (x2, y2)] for (x1, y1), (x2, y2) in zip(path, path[1:])],
                color=(0, 0, 255),
                thickness=3,
            )
        vis.draw_points(x_rands, color=(255, 255, 255), radius=4)
        vis.draw_points(x_nears, color=(255, 0, 0), radius=4)
        vis.draw_points(x_news, color=(0, 255, 0), radius=4)

        # Draw start/goal markers
        xs, ys = start
        xg, yg = goal
        vis.draw_pose(xs, ys, yaw=0, size=0.05, color=(0, 200, 0), thickness=3)
        vis.draw_pose(xg, yg, yaw=0, size=0.05, color=(200, 0, 0), thickness=3)

        mode = "interactive" if args.interactive else "continuous"
        vis.draw_text(
            (
                f"Iterations: {iteration} | refresh: {args.refresh} | "
                f"neighbors: {args.neighbors} | mode: {mode}"
            ),
            (10, 10),
            color=(255, 255, 0),
        )
        pygame.display.flip()
        if args.interactive:
            running = wait_for_keypress()
        else:
            clock.tick(60)


if __name__ == "__main__":
    main()
