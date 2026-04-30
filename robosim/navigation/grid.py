"""Occupancy-grid and A* path planning utilities."""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from robosim.navigation.geometry import Point2D

GridIndex = tuple[int, int]


@dataclass(slots=True)
class OccupancyGrid:
    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int
    occupied: list[bool]

    @classmethod
    def empty(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: float,
    ) -> "OccupancyGrid":
        if resolution <= 0.0:
            raise ValueError("Grid resolution must be positive")
        width = max(1, math.ceil((x_max - x_min) / resolution))
        height = max(1, math.ceil((y_max - y_min) / resolution))
        return cls(
            origin_x=x_min,
            origin_y=y_min,
            resolution=resolution,
            width=width,
            height=height,
            occupied=[False] * (width * height),
        )

    def copy(self) -> "OccupancyGrid":
        return OccupancyGrid(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            resolution=self.resolution,
            width=self.width,
            height=self.height,
            occupied=list(self.occupied),
        )

    def _flat_index(self, cell: GridIndex) -> int:
        ix, iy = cell
        return iy * self.width + ix

    def in_bounds(self, cell: GridIndex) -> bool:
        ix, iy = cell
        return 0 <= ix < self.width and 0 <= iy < self.height

    def clamp_cell(self, cell: GridIndex) -> GridIndex:
        ix, iy = cell
        return (
            max(0, min(self.width - 1, ix)),
            max(0, min(self.height - 1, iy)),
        )

    def world_to_grid(self, point: Point2D | tuple[float, float]) -> GridIndex:
        if isinstance(point, Point2D):
            x, y = point.x, point.y
        else:
            x, y = point
        return (
            math.floor((x - self.origin_x) / self.resolution),
            math.floor((y - self.origin_y) / self.resolution),
        )

    def grid_to_world(self, cell: GridIndex) -> Point2D:
        ix, iy = cell
        return Point2D(
            x=self.origin_x + (ix + 0.5) * self.resolution,
            y=self.origin_y + (iy + 0.5) * self.resolution,
        )

    def is_occupied(self, cell: GridIndex) -> bool:
        if not self.in_bounds(cell):
            return True
        return self.occupied[self._flat_index(cell)]

    def is_free(self, cell: GridIndex) -> bool:
        return self.in_bounds(cell) and not self.occupied[self._flat_index(cell)]

    def set_occupied(self, cell: GridIndex, value: bool = True) -> None:
        if self.in_bounds(cell):
            self.occupied[self._flat_index(cell)] = value

    def occupied_count(self) -> int:
        return sum(1 for value in self.occupied if value)

    def mark_rect(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        ix0, iy0 = self.world_to_grid((x_min, y_min))
        ix1, iy1 = self.world_to_grid((x_max, y_max))
        for ix in range(max(0, ix0), min(self.width - 1, ix1) + 1):
            for iy in range(max(0, iy0), min(self.height - 1, iy1) + 1):
                self.set_occupied((ix, iy), True)

    def mark_disc(self, center: Point2D, radius: float) -> None:
        center_cell = self.world_to_grid(center)
        radius_cells = math.ceil(radius / self.resolution)
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if math.hypot(dx * self.resolution, dy * self.resolution) > radius:
                    continue
                self.set_occupied((center_cell[0] + dx, center_cell[1] + dy), True)

    def inflate(self, radius: float) -> None:
        if radius <= 0.0:
            return
        occupied_cells = [
            (ix, iy)
            for iy in range(self.height)
            for ix in range(self.width)
            if self.occupied[self._flat_index((ix, iy))]
        ]
        radius_cells = math.ceil(radius / self.resolution)
        inflated = list(self.occupied)
        for ox, oy in occupied_cells:
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if math.hypot(dx * self.resolution, dy * self.resolution) > radius:
                        continue
                    cell = (ox + dx, oy + dy)
                    if self.in_bounds(cell):
                        inflated[self._flat_index(cell)] = True
        self.occupied = inflated

    def cells_on_line(self, a: GridIndex, b: GridIndex) -> list[GridIndex]:
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        cells: list[GridIndex] = []

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            err2 = 2 * err
            if err2 >= dy:
                err += dy
                x0 += sx
            if err2 <= dx:
                err += dx
                y0 += sy
        return cells

    def line_is_free(self, a: GridIndex, b: GridIndex) -> bool:
        return all(self.is_free(cell) for cell in self.cells_on_line(a, b))

    def nearest_free(
        self,
        cell: GridIndex,
        max_radius_cells: int | None = None,
    ) -> GridIndex | None:
        start = self.clamp_cell(cell)
        if self.is_free(start):
            return start
        limit = max_radius_cells or max(self.width, self.height)
        queue: deque[tuple[GridIndex, int]] = deque([(start, 0)])
        visited = {start}

        while queue:
            current, distance = queue.popleft()
            if distance >= limit:
                continue
            for neighbor in self.neighbors(current, include_occupied=True):
                if neighbor in visited:
                    continue
                if self.is_free(neighbor):
                    return neighbor
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return None

    def neighbors(
        self,
        cell: GridIndex,
        include_occupied: bool = False,
    ) -> Iterable[GridIndex]:
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if not self.in_bounds(neighbor):
                    continue
                if not include_occupied and not self.is_free(neighbor):
                    continue
                yield neighbor

    def render_ascii(
        self,
        path: Iterable[GridIndex] | None = None,
        start: GridIndex | None = None,
        goal: GridIndex | None = None,
        max_width: int = 120,
        max_height: int = 80,
    ) -> str:
        path_cells = set(path or [])
        stride_x = max(1, math.ceil(self.width / max_width))
        stride_y = max(1, math.ceil(self.height / max_height))
        lines: list[str] = []
        for iy in range(self.height - 1, -1, -stride_y):
            line: list[str] = []
            for ix in range(0, self.width, stride_x):
                cell = (ix, iy)
                if start is not None and cell == start:
                    line.append("S")
                elif goal is not None and cell == goal:
                    line.append("G")
                elif cell in path_cells:
                    line.append("*")
                elif self.is_occupied(cell):
                    line.append("#")
                else:
                    line.append(".")
            lines.append("".join(line))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PlanningResult:
    start_cell: GridIndex
    goal_cell: GridIndex
    raw_cells: list[GridIndex]
    cells: list[GridIndex]
    points: list[Point2D]


class AStarPlanner:
    def __init__(self, grid: OccupancyGrid, allow_diagonal: bool = True) -> None:
        self.grid = grid
        self.allow_diagonal = allow_diagonal

    def plan(
        self,
        start: Point2D | tuple[float, float],
        goal: Point2D | tuple[float, float],
        simplify: bool = True,
    ) -> PlanningResult:
        start_cell = self.grid.nearest_free(self.grid.world_to_grid(start))
        goal_cell = self.grid.nearest_free(self.grid.world_to_grid(goal))
        if start_cell is None:
            raise ValueError("Start is outside the navigable grid")
        if goal_cell is None:
            raise ValueError("Goal is outside the navigable grid")

        raw_cells = self._search(start_cell, goal_cell)
        cells = self._simplify_cells(raw_cells) if simplify else raw_cells
        return PlanningResult(
            start_cell=start_cell,
            goal_cell=goal_cell,
            raw_cells=raw_cells,
            cells=cells,
            points=[self.grid.grid_to_world(cell) for cell in cells],
        )

    def _search(self, start: GridIndex, goal: GridIndex) -> list[GridIndex]:
        heap: list[tuple[float, int, GridIndex]] = []
        counter = 0
        heapq.heappush(heap, (0.0, counter, start))
        came_from: dict[GridIndex, GridIndex | None] = {start: None}
        g_score: dict[GridIndex, float] = {start: 0.0}

        while heap:
            _, _, current = heapq.heappop(heap)
            if current == goal:
                return self._reconstruct(came_from, current)

            for neighbor in self._valid_neighbors(current):
                tentative = g_score[current] + self._move_cost(current, neighbor)
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                counter += 1
                priority = tentative + self._heuristic(neighbor, goal)
                heapq.heappush(heap, (priority, counter, neighbor))

        raise ValueError(f"No path from {start} to {goal}")

    def _valid_neighbors(self, cell: GridIndex) -> Iterable[GridIndex]:
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if not self.allow_diagonal and dx != 0 and dy != 0:
                    continue
                neighbor = (x + dx, y + dy)
                if not self.grid.is_free(neighbor):
                    continue
                if dx != 0 and dy != 0:
                    if not self.grid.is_free((x + dx, y)) or not self.grid.is_free((x, y + dy)):
                        continue
                yield neighbor

    @staticmethod
    def _move_cost(a: GridIndex, b: GridIndex) -> float:
        return math.sqrt(2.0) if a[0] != b[0] and a[1] != b[1] else 1.0

    @staticmethod
    def _heuristic(a: GridIndex, b: GridIndex) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    @staticmethod
    def _reconstruct(
        came_from: dict[GridIndex, GridIndex | None],
        current: GridIndex,
    ) -> list[GridIndex]:
        cells = [current]
        while came_from[current] is not None:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        return cells

    def _simplify_cells(self, cells: list[GridIndex]) -> list[GridIndex]:
        if len(cells) <= 2:
            return cells
        simplified = [cells[0]]
        anchor_index = 0
        while anchor_index < len(cells) - 1:
            next_index = len(cells) - 1
            while next_index > anchor_index + 1:
                if self.grid.line_is_free(cells[anchor_index], cells[next_index]):
                    break
                next_index -= 1
            simplified.append(cells[next_index])
            anchor_index = next_index
        return simplified
