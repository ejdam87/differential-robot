"""My primitives used by RRT* planner."""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from PIL import Image
from rtree import index

from .rrt_primitives_base import Control, NodeId, RRTPrimitivesBase, State
from robot_constants import ROBOT_DIAMETER


def disk_structuring_element(diameter: int) -> NDArray:
    """
    Creates a circular SE for morphological dilatation of the map to avoid going close to the obstacles.

    params:
        diameter: diameter of the disk
    returns:
        mask: structuring element
    """
    radius = diameter // 2
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask


class Robot2DRRTPrimitives(RRTPrimitivesBase):
    """
    This object encapsulates RRT* primitives with pre-defined interface
    """

    def __init__(
        self,
        image_path: str,
        pixel_size: float = 0.001,  # meter / pixel
        spatial_step_size: float = 0.10,  # meters
        eps_position: float = 1e-3,  # meters
        neighbor_k: int = 15,
        dilated_map_path: str | None = None,
        save_obstacle_map: bool = True,
    ) -> None:
        """
        params:
            image_path: path to map image
            pixel_size: size of a pixel
            spatial_step_size: extension towards x_rand
            eps_position: tolerance
            neighbor_k: number of neighbors to consider
            dilated_map_path: path to dilated map image if exists (avoid recomputing)
            save_obstacle_map: whether to save the dilated map image
        """

        self.pixel_size = pixel_size
        self.spatial_step_size = spatial_step_size
        self.eps_position = eps_position

        self.neighbor_k = neighbor_k

        # load dilated map directly
        if dilated_map_path is not None:
            self.dilated_grid = np.array(
                Image.open(dilated_map_path).convert("L"), dtype=np.uint8
            ) > 0
        else:
            img = Image.open(image_path).convert("L")
            self.grid = np.array(img, dtype=np.uint8)
            self.binary_grid = (self.grid > 0) & (self.grid < 255)  # True ~ obstacle

            # distances to obstacles
            self.free_space_distance = distance_transform_edt(~self.binary_grid)
            self.dilated_grid = self.free_space_distance <= (
                ROBOT_DIAMETER * 1.5 / self.pixel_size
            )

        if save_obstacle_map:
            img = Image.fromarray(self.dilated_grid)
            img.save("maps/dilated_map.png")

        self.height, self.width = self.dilated_grid.shape
        self.max_x = self.width * pixel_size
        self.max_y = self.height * pixel_size
        free_pixels = np.argwhere(self.dilated_grid == False)

        if free_pixels.size == 0:
            raise ValueError("Map has no free space (black pixels).")
        self.free_pixels = free_pixels

        # Spatial index over points encoded as zero-area bounds: (x, y, x, y).
        self.index = index.Index()

        self.states: List[Tuple[float, float]] = []
        self.next_id = 0

    # ---------------- primitives ----------------
    def sample(self) -> State:
        """
        Unformly sample a new 2D point in the map

        returns:
            newly sampled point
        """
        # (x, y) of the center of the robot (ignoring angle)
        x = random.uniform(0, self.max_x)
        y = random.uniform(0, self.max_y)
        return x, y

    def spatial_distance(self, state_a: State, state_b: State) -> float:
        """
        Computes a spatial distance (ignoring orientation) between two states

        params:
            state_a: first state
            state_b: second state
        returns:
            spatial distance
        """
        ax, ay = state_a
        bx, by = state_b
        return math.hypot(ax - bx, ay - by)

    def distance(self, state_a: State, state_b: State) -> float:
        """
        Computes a distance between two states

        params:
            state_a: first state
            state_b: second state
        returns:
            distance
        """
        return self.spatial_distance(state_a, state_b)

    def steer(self, start: State, goal: State) -> Optional[State]:
        """
        Computes a linear steering towards a goal state

        params:
            start: start state
            goal: goal state
        returns:
            new state
        """
        sx, sy = start
        gx, gy = goal
        dx = gx - sx
        dy = gy - sy

        spatial_dist = self.distance(start, goal)

        # we are there
        if spatial_dist < self.eps_position:
            return None

        spatial_step = min(self.spatial_step_size, spatial_dist)
        if spatial_dist < self.eps_position:
            nx = sx
            ny = sy
        else:
            nx = sx + dx / spatial_dist * spatial_step
            ny = sy + dy / spatial_dist * spatial_step

        candidate = (nx, ny)

        if self._edge_free(start, candidate):
            return candidate
        return None

    def node_added(self, state: State) -> NodeId:
        """
        Adds new state to our structures

        params:
            state: new state
        returns:
            node id
        """
        node_id = self.next_id
        self.next_id += 1
        self.states.append(state)
        self.index.insert(node_id, self._point_bounds(state))
        return node_id

    def get_neighbors(self, state: State) -> Iterable[NodeId]:
        """
        Returns k nearest neighbours of given state

        params:
            state: state
        returns:
            list of node ids
        """
        k = min(self.neighbor_k, len(self.states))
        if k == 0:
            return []

        candidate_ids = list(
            self.index.nearest(self._point_bounds(state), num_results=k)
        )

        # Rtree may return > k ids on equal-distance ties; keep stable cardinality.
        return [int(node_id) for node_id in candidate_ids[:k]]

    def get_nearest(self, state: State) -> NodeId:
        """
        Returns nearest neighbour of given state

        params:
            state: state
        returns:
            node id
        """
        if not self.states:
            raise RuntimeError("No states in the tree yet.")

        candidate_ids = list(
            self.index.nearest(self._point_bounds(state), num_results=1)
        )
        if not candidate_ids:
            raise RuntimeError("Spatial index is empty.")
        return int(candidate_ids[0])

    def get_control(self, start: State, goal: State) -> Optional[Control]:
        """
        Computes control from start to goal state

        params:
            start: start state
            goal: goal state
        returns:
            control
        """
        return (
            (goal[0] - start[0], goal[1] - start[1])
            if self._edge_free(start, goal)
            else None
        )

    def edge_cost(self, start: State, control: Optional[Control], goal: State) -> float:
        """
        Computes edge cost from start to goal state

        params:
            start: start state
            control: control (not used in our case, its here just to preserve protocol)
            goal: goal state
        returns:
            edge cost
        """
        return self.distance(start, goal)

    def getPath2D(
        self, start: State, control: Optional[Control], goal: State
    ) -> list[State]:
        """
        Obtains a 2D path from start to goal state

        params:
            start: initial state
            control: control (not used in our case, its here just to preserve protocol)
            goal: target state
        returns:
            2D path
        """
        x1, y1 = start
        x2, y2 = goal

        return [(x1, y1), (x2, y2)]

    # ---------------- helpers ----------------
    def _edge_free(self, start: State, goal: State) -> bool:
        """
        Computes whether the edge between start and goal does not cross any obstacles

        params:
            start: start state
            goal: goal state
        returns:
            whether the edge is free
        """
        sx, sy = start
        gx, gy = goal
        dist = self.distance(start, goal)
        if dist == 0:
            return self._is_free(sx, sy)
        steps = int(dist / (self.pixel_size * 0.5)) + 1
        for i in range(steps + 1):
            t = i / steps
            x = sx + (gx - sx) * t
            y = sy + (gy - sy) * t
            if not self._is_free(x, y):
                return False
        return True

    def _is_free(self, x: float, y: float) -> bool:
        """
        Computes whether a point is free

        params:
            x: x coordinate
            y: y coordinate
        returns:
            whether the point is free
        """
        px = int(round(x / self.pixel_size))
        py = int(round(y / self.pixel_size))
        if px < 0 or py < 0 or px >= self.width or py >= self.height:
            return False
        return self.dilated_grid[py, px] == False

    @staticmethod
    def _point_bounds(state: State) -> tuple[float, float, float, float]:
        """
        Obtains bounding box of given state as just a point itself

        params:
            state: state
        returns:
            bounding box
        """
        x, y = state
        return (x, y, x, y)
