"""
Classic RRT* implementation driven by :class:`RRTPrimitivesBase` primitives.

This is a work of Martin Skalský (skalsky@mail.muni.cz)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .rrt_primitives_base import Control, NodeId, RRTPrimitivesBase, State


class RRTStar:
    def __init__(self, problem: RRTPrimitivesBase, start_state: State) -> None:
        self.problem = problem
        self.root: NodeId = self.problem.node_added(start_state)
        self.parents: Dict[NodeId, Optional[NodeId]] = {self.root: None}
        self.costs: Dict[NodeId, float] = {self.root: 0.0}
        self.edge_controls: Dict[NodeId, Optional[Control]] = {self.root: None}
        self.children: Dict[NodeId, Set[NodeId]] = {self.root: set()}
        self.states: Dict[NodeId, State] = {self.root: start_state}

    def _choose_parent(
        self,
        x_new: State,
        neighbors: List[NodeId],
        default_parent: NodeId,
        default_control: Control,
    ) -> tuple[NodeId, Optional[Control], float]:
        best_parent = default_parent
        best_control: Optional[Control] = default_control

        best_cost = self.costs[default_parent] + self.problem.edge_cost(
            self.states[default_parent], best_control, x_new
        )

        for nid in neighbors:
            if nid == default_parent:
                continue
            u = self.problem.get_control(self.states[nid], x_new)
            if u is None:
                continue
            cost = self.costs[nid] + self.problem.edge_cost(self.states[nid], u, x_new)
            if cost < best_cost:
                best_cost = cost
                best_parent = nid
                best_control = u

        return best_parent, best_control, best_cost

    def _set_parent(
        self, child: NodeId, new_parent: NodeId, control: Optional[Control]
    ) -> None:
        old_parent = self.parents.get(child)
        if old_parent is not None:
            self.children.setdefault(old_parent, set()).discard(child)
        self.parents[child] = new_parent
        self.edge_controls[child] = control
        self.children.setdefault(new_parent, set()).add(child)

    def _update_subtree_costs(self, node_id: NodeId) -> None:
        stack = list(self.children.get(node_id, []))
        while stack:
            child = stack.pop()
            parent = self.parents[child]
            control = self.edge_controls[child]
            parent_state = self.states[parent]
            child_state = self.states[child]
            self.costs[child] = self.costs[parent] + self.problem.edge_cost(
                parent_state, control, child_state
            )
            stack.extend(self.children.get(child, []))

    def _rewire(self, new_id: NodeId, x_new: State, neighbors: List[NodeId]) -> None:
        for nid in neighbors:
            if nid == self.root or nid == new_id:
                continue
            x_near = self.states[nid]
            u = self.problem.get_control(x_new, x_near)
            if u is None:
                continue
            new_cost = self.costs[new_id] + self.problem.edge_cost(x_new, u, x_near)
            if new_cost + 1e-9 < self.costs.get(nid, float("inf")):
                self._set_parent(nid, new_id, u)
                self.costs[nid] = new_cost
                self._update_subtree_costs(nid)

    def iterate(
        self, iterations: int = 1
    ) -> tuple[list[State], list[State], list[State]]:
        x_rands: list[State] = []
        x_nears: list[State] = []
        x_news: list[State] = []
        for _ in range(iterations):
            x_rand = self.problem.sample()
            near_id = self.problem.get_nearest(x_rand)
            x_near = self.states[near_id]

            x_new = self.problem.steer(x_near, x_rand)
            if x_new is None:
                continue
            u_near = self.problem.get_control(x_near, x_new)
            if u_near is None:
                continue

            neighbors = list(self.problem.get_neighbors(x_new))
            parent_id, parent_control, new_cost = self._choose_parent(
                x_new, neighbors, near_id, u_near
            )
            if new_cost == float("inf"):
                continue

            new_id = self.problem.node_added(x_new)
            self.children.setdefault(new_id, set())
            self._set_parent(new_id, parent_id, parent_control)
            self.costs[new_id] = new_cost
            self.states[new_id] = x_new
            self._rewire(new_id, x_new, neighbors)

            x_rands.append(x_rand)
            x_nears.append(x_near)
            x_news.append(x_new)

        return x_rands, x_nears, x_news

    def extract_path(self, to_node: Optional[NodeId] = None) -> list[State]:
        if to_node is None:
            raise ValueError(
                "to_node must be provided; RRT* does not track a last node here."
            )
        path: list[State] = []
        cur: Optional[NodeId] = to_node
        while cur is not None:
            path.append(self.states[cur])
            cur = self.parents.get(cur)
        path.reverse()
        return path

    def draw_tree(self) -> List[List[State]]:
        edges: List[List[State]] = []
        for child, parent in self.parents.items():
            if parent is None:
                continue
            start = self.states[parent]
            goal = self.states[child]
            control = self.edge_controls.get(child)
            edges.append(self.problem.getPath2D(start, control, goal))
        return edges
