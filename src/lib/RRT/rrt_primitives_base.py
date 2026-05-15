"""Base interfaces for RRT / RRT* planning problems.

Concrete planners should subclass :class:`RRTPrimitivesBase` and implement
all abstract methods. The interface is intentionally small and opinionated so
it can be slotted into a sampler-based planner without additional glue code.

This is a work of Martin Skalský (skalsky@mail.muni.cz) 
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

# Type aliases used by the interface
State = Any
Control = Any
NodeId = Any


class RRTPrimitivesBase(ABC):
    """Interface that exposes the primitives required by RRT/RRT* planners."""

    @abstractmethod
    def sample(self) -> State:
        """Draw a random valid state from the C space."""

    @abstractmethod
    def steer(self, start: State, goal: State) -> Optional[State]:
        """Generate a new state that moves from ``start`` toward ``goal``.

        Implementations typically cap the step length to keep expansions local.
        Might return goal if a direct path to goal is possible.
        Return None if no valid step is possible.

        """

    @abstractmethod
    def node_added(self, state: State) -> NodeId:
        """Called by RRT when a new node enters the tree; return a node identifier."""

    @abstractmethod
    def get_neighbors(self, state: State) -> Iterable[NodeId]:
        """Return node ids considered near the query ``state`` (for NN/rewiring), using spatial index"""

    @abstractmethod
    def get_nearest(self, state: State) -> NodeId:
        """Return the single nearest node id to ``state`` using the spatial index."""

    @abstractmethod
    def get_control(self, start: State, goal: State) -> Optional[Control]:
        """Return a control input that drives from ``start`` toward ``goal``.

        If no feasible control exists, return None.
        """

    @abstractmethod
    def edge_cost(self, start: State, control: Control, goal: State) -> float:
        """Compute traversal cost from ``start`` to ``goal`` given ``control``."""

    @abstractmethod
    def getPath2D(self, start: State, control: Optional[Control], goal: State):
        """Return a polyline (list of (x,y) points) representing the edge start->goal.
        This method. It is the"""
