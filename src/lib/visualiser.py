"""
PyGame visualizer object

This is a work of Martin Skalský (skalsky@mail.muni.cz) 
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pygame

# ---------------- Constants ----------------

BG_COLOR = (230, 230, 230)
GRID_COLOR = (200, 200, 200)
TICK_COLOR = (150, 150, 150)
LINE_WIDTH = 1

# Colors for low/high weight
COLOR_LOW = np.array([139, 69, 19], float)  # Brown
COLOR_HIGH = np.array([0, 128, 0], float)  # Green


# ---------------- Helpers ----------------


def sigmoid(x: float) -> float:
    """Map any real value to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def color_from_weight(weight: float) -> Tuple[int, int, int]:
    """Map weight -> brown->green using sigmoid interpolation."""
    s = sigmoid(weight)
    c = COLOR_LOW * (1.0 - s) + COLOR_HIGH * s
    return tuple(np.clip(c.astype(int), 0, 255))


@dataclass
class WorldBounds:
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0

    @property
    def left(self) -> float:
        return self.x_min

    @property
    def right(self) -> float:
        return self.x_max

    @property
    def bottom(self) -> float:
        return self.y_max

    @property
    def top(self) -> float:
        return self.y_min

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


# ---------------- Screen mapping ----------------
class View:
    def __init__(
        self,
        bounding_rect: pygame.Rect,
        world_width: float,
        zero_pos: Tuple[float, float],
    ) -> None:
        """
        Args:
            bounding_rect: Bounding rectangle of the view in the screen.
            world_width: Width of the shown world in meters; determines the scale.
            zero_pos: Position of the origin in the world. -zero_pos are minimum visible
                coordinates.
        """
        self.world_width = float(world_width)
        self.update(bounding_rect, zero_pos=zero_pos)

    def update(
        self,
        bounding_rect: pygame.Rect,
        zero_pos: Optional[Tuple[float, float]] = None,
        zero_pos_px: Optional[Tuple[float, float]] = None,
        world_width: Optional[float] = None,
    ) -> None:
        """
        Update view geometry.

        Provide either zero_pos (in world units) or zero_pos_px (in pixels). Only one should be
        given.
        Optionally update world_width to change scale.
        """
        self.rect = bounding_rect

        if world_width is not None:
            self.world_width = float(world_width)

        if zero_pos is not None and zero_pos_px is not None:
            raise ValueError("Provide only one of zero_pos or zero_pos_px.")

        # Compute scale from rect and world_width
        self.scale = self.rect.width / self.world_width

        # Resolve zero position in pixels
        if zero_pos is not None:
            zero_pos_px = (zero_pos[0] * self.scale, zero_pos[1] * self.scale)
            self.offset_x = zero_pos_px[0] + self.rect.left
            self.offset_y = zero_pos_px[1] + self.rect.top
        elif zero_pos_px is not None:
            self.offset_x = zero_pos_px[0]
            self.offset_y = zero_pos_px[1]
        else:
            # Default to current offsets if available
            self.offset_x = getattr(self, "offset_x", 0.0)
            self.offset_y = getattr(self, "offset_y", 0.0)
            zero_pos_px = (self.offset_x, self.offset_y)

        # World bounds in current view
        self.world_bounds = WorldBounds()
        self.world_bounds.x_min = -zero_pos_px[0] / self.scale
        self.world_bounds.x_max = self.world_bounds.x_min + self.world_width
        self.world_bounds.y_min = -zero_pos_px[1] / self.scale
        self.world_bounds.y_max = (
            self.world_bounds.y_min
            + self.world_width * self.rect.height / self.rect.width
        )

    def world_to_px(self, x: float, y: float) -> Tuple[int, int]:
        px = int(x * self.scale + self.offset_x)
        py = int(y * self.scale + self.offset_y)
        return px, py

    def px_to_world(self, px: int, py: int) -> Tuple[float, float]:
        x = (px - self.offset_x) / self.scale
        y = (py - self.offset_y) / self.scale
        return x, y

    def world_size_to_px(self, w: float, h: float) -> Tuple[int, int]:
        """Convert world size (meters) to pixel size for current scale."""
        return int(round(w * self.scale)), int(round(h * self.scale))


# ---------------- Visualization Class ----------------
class Visualizer:
    def __init__(
        self,
        world_width: float = 4.0,
        origin: Tuple[float, float] = (0.5, 0.5),
        window_size: Tuple[int, int] = (1200, 900),
        font_size: int = 24,
    ) -> None:
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("PB120 Visualizer")

        self.font = pygame.font.SysFont("monospace", font_size)

        bounding_rect = self.screen.get_rect()
        self.view = View(bounding_rect, world_width, origin)
        self.origin_px = self.view.world_to_px(0, 0)

        self.ticks = 7
        self.tick = self._find_tick_size(self.view.world_width, self.ticks)

        self.fullscreen = False

        self.last_click_world: Optional[Tuple[float, float]] = None

        # Internal state
        self.particle_data: List[Tuple[float, float, float, float]] = []

        # Drag state for panning the world origin
        self._dragging = False
        self._drag_start_px: Optional[Tuple[int, int]] = None
        self._drag_start_origin: Optional[Tuple[float, float]] = None

        # Zoom config
        self._zoom_factor = 1.15
        self._world_width_min = 0.1
        self._world_width_max = 1000.0

        # Background image cache
        self._bg_image: Optional[pygame.Surface] = None
        self._bg_image_px_size: Optional[Tuple[int, int]] = None
        self._bg_pixel_size_m: Optional[float] = None

    def clean(self) -> None:
        self.screen.fill(BG_COLOR)
        self.draw_background_image()
        pygame.draw.rect(self.screen, GRID_COLOR, self.view.rect, width=LINE_WIDTH)

        tick = self.tick

        for x in np.arange(
            math.floor(self.view.world_bounds.left / tick) * tick,
            self.view.world_bounds.right + 1e-9,
            tick,
        ):
            px, py = self.view.world_to_px(x, 0)
            pygame.draw.line(
                self.screen, TICK_COLOR, (px, py), (px, py + 15), LINE_WIDTH
            )
            label = self.font.render(f"{x:g}", True, TICK_COLOR)
            self.screen.blit(label, (px - 20, py + 20))

        # Horizontal zero line
        pygame.draw.line(
            self.screen,
            TICK_COLOR,
            self.view.world_to_px(self.view.world_bounds.left, 0),
            self.view.world_to_px(self.view.world_bounds.right, 0),
            LINE_WIDTH,
        )

        for y in np.arange(
            math.floor(self.view.world_bounds.top / tick) * tick,
            self.view.world_bounds.bottom + 1e-9,
            tick,
        ):
            px, py = self.view.world_to_px(0, y)
            pygame.draw.line(
                self.screen, TICK_COLOR, (px - 15, py), (px, py), LINE_WIDTH
            )
            label = self.font.render(f"{y:g}", True, TICK_COLOR)
            self.screen.blit(label, (px + 30, py - 15))

        # Vertical zero line
        pygame.draw.line(
            self.screen,
            TICK_COLOR,
            (self.view.world_to_px(0, 0)[0], self.view.rect.top),
            (self.view.world_to_px(0, 0)[0], self.view.rect.bottom),
            LINE_WIDTH,
        )

    def draw_particles(
        self,
        particles: Iterable[Tuple[float, float, float, float]],
        max_size: float = 0.5,
    ) -> None:
        for x, y, yaw, weight in particles:
            size = weight * max_size
            color = color_from_weight(weight)
            self.draw_pose(x, y, yaw, size=size, color=color, thickness=2)

    def draw_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        size: float = 0.1,
        arrow_len: float = 2.0,
        text: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = LINE_WIDTH,
    ) -> None:
        radius_px = size * self.view.scale / 2.0
        arrow_len_px = radius_px * arrow_len
        px, py = self.view.world_to_px(x, y)
        tip = (
            int(px + arrow_len_px * math.cos(yaw)),
            int(py + arrow_len_px * math.sin(yaw)),
        )
        pygame.draw.line(self.screen, color, (px, py), tip, thickness)
        pygame.draw.circle(self.screen, color, (px, py), radius_px, thickness)
        pygame.draw.circle(self.screen, color, (px, py), 3)
        if text is not None:
            label = self.font.render(text, True, color)
            self.screen.blit(label, (px + radius_px, py + radius_px))

    def draw_poses(
        self,
        robots: Iterable[Tuple[float, float, float]],
        size: float = 0.1,
        arrow_len: float = 2.0,
        text: Optional[str] = None,
        idx_prefix: bool = True,
        color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = LINE_WIDTH,
    ) -> None:
        for idx, (x, y, yaw) in enumerate(robots):
            final_text = text
            if idx_prefix:
                final_text = f"{idx}" + (text if text is not None else "")
            self.draw_pose(
                x,
                y,
                yaw,
                size=size,
                text=final_text,
                arrow_len=arrow_len,
                color=color,
                thickness=thickness,
            )

    def draw_points(
        self,
        points: Iterable[Tuple[float, float]],
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 2,
    ) -> None:
        for x, y in points:
            px, py = self.view.world_to_px(x, y)
            pygame.draw.circle(self.screen, color, (px, py), radius)

    def draw_edges(
        self,
        edges: Iterable[Iterable[Tuple[float, float]]],
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = LINE_WIDTH,
    ) -> None:
        """Draw multiple polylines; each edge is an iterable of (x, y) points."""
        for edge in edges:
            pts = [self.view.world_to_px(x, y) for x, y in edge]
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, color, False, pts, thickness)

    def draw_text(
        self,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """Render a text string onto the screen at a specified location."""
        label = self.font.render(text, True, color)
        self.screen.blit(label, pos)

    def set_background_image(self, image_path: str, pixel_size_m: float) -> None:
        """
        Load a background image from disk and remember its pixel size in meters.

        The image world origin (0,0) is the image top-left. It will be drawn so that
        its top-left aligns to world (0,0), scaled according to current zoom.
        """
        try:
            img = pygame.image.load(image_path).convert_alpha()
        except Exception:
            img = pygame.image.load(image_path).convert()  # fallback if no alpha
        self._bg_image = img
        self._bg_image_px_size = img.get_size()
        self._bg_pixel_size_m = float(pixel_size_m)

    def draw_background_image(self) -> None:
        """
        Draw the loaded background image so that it starts at world (0,0).

        Each image pixel corresponds to _bg_pixel_size_m meters.
        Scales with zoom and pans with the current view.
        """
        if not self._bg_image or not self._bg_pixel_size_m:
            return

        img_w_px, img_h_px = self._bg_image_px_size
        img_w_m = img_w_px * self._bg_pixel_size_m
        img_h_m = img_h_px * self._bg_pixel_size_m

        top_left_screen = self.view.world_to_px(0.0, 0.0)
        dest_w_px, dest_h_px = self.view.world_size_to_px(img_w_m, img_h_m)

        if dest_w_px <= 0 or dest_h_px <= 0:
            return

        scaled = pygame.transform.smoothscale(self._bg_image, (dest_w_px, dest_h_px))
        self.screen.blit(scaled, top_left_screen)

    def handle_events(self) -> None:
        """Handle Pygame events (quit, key presses, resize, mouse drag, zoom)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                self.fullscreen = not self.fullscreen

                if self.fullscreen:
                    self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                else:
                    self.screen = pygame.display.set_mode(
                        self.window_size, pygame.RESIZABLE
                    )
                self.view.update(self.screen.get_rect(), zero_pos_px=self.origin_px)

            elif event.type == pygame.VIDEORESIZE:
                self.view.update(self.screen.get_rect(), zero_pos_px=self.origin_px)
                self.clean()

            # Mouse drag to pan origin
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if self.view.rect.collidepoint(event.pos):
                    wx, wy = self.view.px_to_world(*event.pos)
                    self.last_click_world = (wx, wy)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.view.rect.collidepoint(event.pos):
                    self._dragging = True
                    self._drag_start_px = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging = False
                self.origin_px = self.view.world_to_px(0, 0)

            elif event.type == pygame.MOUSEMOTION and self._dragging:
                mx, my = event.pos
                sx, sy = self._drag_start_px
                dx_px, dy_px = mx - sx, my - sy

                self.view.update(
                    self.screen.get_rect(),
                    zero_pos_px=(self.origin_px[0] + dx_px, self.origin_px[1] + dy_px),
                )

            # Mouse wheel zoom (focus on cursor)
            elif event.type == pygame.MOUSEWHEEL:
                if self.view.rect.collidepoint(pygame.mouse.get_pos()):
                    cx, cy = pygame.mouse.get_pos()

                    origin_dx = cx - self.origin_px[0]
                    origin_dy = cy - self.origin_px[1]

                    scale_change = self._zoom_factor**event.y

                    new_world_width = self.view.world_width / scale_change
                    new_world_width = max(
                        self._world_width_min,
                        min(self._world_width_max, new_world_width),
                    )

                    real_scale_change = self.view.world_width / new_world_width
                    self.origin_px = (
                        cx - origin_dx * real_scale_change,
                        cy - origin_dy * real_scale_change,
                    )

                    self.view.update(
                        self.screen.get_rect(),
                        zero_pos_px=self.origin_px,
                        world_width=new_world_width,
                    )
                    self.tick = self._find_tick_size(self.view.world_width, self.ticks)

    def _find_tick_size(self, world_width: float, num_ticks: int = 5) -> float:
        """Find appropriate tick size for given world width."""
        good_ticks = [1, 2, 5]

        magnitude = 10 ** math.floor(math.log10(world_width))
        target = world_width / num_ticks

        best_tick = float("inf")
        for base in good_ticks:
            for mult in [0.1, 1, 10]:
                tick = base * magnitude * mult
                if abs(tick - target) < abs(best_tick - target):
                    best_tick = tick

        return best_tick

    def consume_target(self) -> Optional[Tuple[float, float]]:
        click = self.last_click_world
        self.last_click_world = None
        return click


if __name__ == "__main__":
    vis = Visualizer()
    clock = pygame.time.Clock()
    t = 0.0

    # Example usage: draw background map starting at 0,0 with given pixel size in meters
    # vis.set_background_image("maps/dalmatin_map.png", pixel_size_m=0.001)

    while True:
        particles = []
        for i in range(6):
            angle = t + i * math.pi / 3
            x = 0.8 * math.cos(angle)
            y = 0.8 * math.sin(angle)
            yaw = angle + math.pi / 2
            weight = math.sin(i)
            particles.append((x, y, yaw, weight))

        vis.handle_events()
        vis.clean()
        vis.draw_background_image()
        vis.draw_particles(particles)
        x = math.sin(t) * 0.5
        vis.draw_text(
            text=f"Pose: {x:.3f}, {0.0:.3f}, {0.0:.3f}", pos=(10, 10), color=(0, 0, 0)
        )
        vis.draw_pose(x=x, y=0, yaw=0, text="Robot", color=(255, 0, 0), thickness=3)
        pygame.display.flip()
        clock.tick(60)
        t += 0.01
