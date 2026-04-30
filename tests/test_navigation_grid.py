"""Tests for the lightweight navigation grid planner."""

from __future__ import annotations

from robosim.navigation.geometry import Point2D
from robosim.navigation.grid import AStarPlanner, OccupancyGrid


def test_a_star_routes_through_gap() -> None:
    grid = OccupancyGrid.empty(0.0, 10.0, 0.0, 10.0, 1.0)
    for iy in range(10):
        if iy == 5:
            continue
        grid.set_occupied((4, iy), True)

    result = AStarPlanner(grid).plan(Point2D(1.5, 1.5), Point2D(8.5, 8.5))

    assert result.start_cell == (1, 1)
    assert result.goal_cell == (8, 8)
    assert (4, 5) in result.raw_cells
    assert result.points[0] == Point2D(1.5, 1.5)
    assert result.points[-1] == Point2D(8.5, 8.5)


def test_planner_simplifies_straight_line_path() -> None:
    grid = OccupancyGrid.empty(0.0, 10.0, 0.0, 10.0, 1.0)

    result = AStarPlanner(grid).plan(Point2D(1.5, 1.5), Point2D(8.5, 1.5))

    assert len(result.raw_cells) == 8
    assert result.cells == [(1, 1), (8, 1)]


def test_nearest_free_finds_adjacent_cell() -> None:
    grid = OccupancyGrid.empty(0.0, 5.0, 0.0, 5.0, 1.0)
    grid.set_occupied((2, 2), True)

    nearest = grid.nearest_free((2, 2), max_radius_cells=2)

    assert nearest is not None
    assert grid.is_free(nearest)
    assert abs(nearest[0] - 2) <= 1
    assert abs(nearest[1] - 2) <= 1
