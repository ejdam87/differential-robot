"""
Minimal RRT planner built on :class:`RRTPrimitivesBase` primitives.

This is a work of Martin Skalský (skalsky@mail.muni.cz)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .rrt_primitives_base import NodeId, RRTPrimitivesBase, State


class RRT:
    def __init__(self, problem: RRTPrimitivesBase, start_state: State) -> None:
        self.problem = problem
        self.root: NodeId = self.problem.node_added(start_state)
        self.parents: Dict[NodeId, Optional[NodeId]] = {self.root: None}
        self.states: Dict[NodeId, State] = {self.root: start_state}

    def iterate(self, iterations: int = 1) -> None:
        """Perform ``iterations`` RRT growth steps."""
        for _ in range(iterations):
            x_rand = self.problem.sample()
            near_id = self.problem.get_nearest(x_rand)
            x_near = self.states[near_id]

            x_new = self.problem.steer(x_near, x_rand)
            if x_new is None:
                continue
            control = self.problem.get_control(x_near, x_new)
            if control is None:
                continue

            new_id = self.problem.node_added(x_new)
            self.parents[new_id] = near_id
            self.states[new_id] = x_new

    def extract_path(self, to_node: NodeId) -> List[State]:
        """Return the state path from root to ``to_node``."""
        path: List[State] = []
        cur: Optional[NodeId] = to_node
        while cur is not None:
            path.append(self.states[cur])
            cur = self.parents.get(cur)
        path.reverse()
        return path
