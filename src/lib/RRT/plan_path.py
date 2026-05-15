from lib.RRT.rrt_star import RRTStar
from lib.RRT.my_primitives import Robot2DRRTPrimitives
from lib.poses import Pose
from lib.visualiser import Visualizer
from lib.types import PlannedPath

import math
import pygame
from typing import Optional


def find_best_goal(
    rrt: RRTStar, goal: tuple[float, float], radius: float
) -> Optional[int]:
    """
    From all states in our RRT* structure, pick the one closest to the goal with at most radius distance from the exact goal or None if such state does not exist.

    params:
        rrt: RRT* structure
        goal: goal position
    returns:
        state_id
    """

    best = None
    best_dist = float("inf")
    for nid, state in rrt.states.items():
        d = math.hypot(state[0] - goal[0], state[1] - goal[1])
        if d < radius and d < best_dist:
            best = nid
            best_dist = d
    return best


def obtain_path(
    map_path: str,
    start: Pose,
    goal: Pose,
    eps: float,
    vis: Visualizer,
    dilated_map_path: str | None = None,
) -> PlannedPath:
    """
    Given start and end poses, obtain a path in the map using RRT*

    params:
        map_path: path to map image
        start: start pose
        goal: goal pose
        eps: tolerance
        vis: visualizer object
        dilated_map_path: path to dilated map image if exists (avoid recomputing)
    returns:
        path: planned path
    """

    primitives = Robot2DRRTPrimitives(
        map_path,
        spatial_step_size=0.10,
        save_obstacle_map=False,
        dilated_map_path=dilated_map_path,
    )

    sx, sy, _ = start.get_raw_pose()
    gx, gy, _ = goal.get_raw_pose()

    rrt = RRTStar(primitives, (sx, sy))

    clock = pygame.time.Clock()
    iterations_per_frame = 50

    running = True

    # run RRT* algorithm with visualization active
    while running:
        vis.handle_events()
        x_rands, x_nears, x_news = rrt.iterate(iterations_per_frame)
        vis.clean()

        vis.draw_edges(rrt.draw_tree(), color=(255, 0, 255), thickness=1)

        near_to_rand = [[x_near, x_rand] for x_near, x_rand in zip(x_nears, x_rands)]
        vis.draw_edges(near_to_rand, color=(160, 160, 160), thickness=1)

        vis.draw_points(x_rands, color=(255, 255, 255), radius=3)
        vis.draw_points(x_nears, color=(255, 0, 0), radius=3)
        vis.draw_points(x_news, color=(0, 255, 0), radius=3)

        goal_node = find_best_goal(rrt, (gx, gy), eps)

        if goal_node is not None:
            path = rrt.extract_path(goal_node)

            vis.draw_edges(
                [[(x1, y1), (x2, y2)] for (x1, y1), (x2, y2) in zip(path, path[1:])],
                color=(0, 0, 255),
                thickness=3,
            )

            # Return immediately once path exists
            pygame.display.flip()
            return path

        vis.draw_pose(sx, sy, yaw=0, size=0.05, color=(0, 200, 0), thickness=3)
        vis.draw_pose(gx, gy, yaw=0, size=0.05, color=(200, 0, 0), thickness=3)

        pygame.display.flip()
        clock.tick(20)
