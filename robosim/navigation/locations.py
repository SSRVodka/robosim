"""Location and waypoint parsing helpers."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import yaml

from robosim.navigation.geometry import Pose2D

DEFAULT_LOCATIONS_FILE = Path(
    "drivers_sim/mujoco/assets/worlds/two_bedroom_apartment/locations.yaml"
)


def load_locations(path: str | Path = DEFAULT_LOCATIONS_FILE) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid locations file: {path}")
    return loaded


def find_location(config: dict[str, Any], query: str) -> dict[str, Any]:
    locations = config["locations"]
    if query in locations:
        return locations[query]
    for item in locations.values():
        if query in item.get("aliases", []):
            return item
    raise ValueError(f"Unknown location: {query}")


def pose_from_mapping(mapping: dict[str, Any]) -> Pose2D:
    return Pose2D(
        x=float(mapping["x"]),
        y=float(mapping["y"]),
        yaw=float(mapping.get("yaw", 0.0)),
    )


def target_pose_for_location(config: dict[str, Any], query: str) -> Pose2D:
    return pose_from_mapping(find_location(config, query)["target_pose"])


def build_graph(edges: list[list[str]]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    for a, b in edges:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    return graph


def find_route(graph: dict[str, list[str]], start: str, goal: str) -> list[str]:
    queue = deque([start])
    came_from: dict[str, str | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for next_waypoint in graph.get(current, []):
            if next_waypoint in came_from:
                continue
            came_from[next_waypoint] = current
            queue.append(next_waypoint)

    if goal not in came_from:
        raise ValueError(f"No route from {start} to {goal}")

    route: list[str] = []
    current: str | None = goal
    while current is not None:
        route.append(current)
        current = came_from[current]
    route.reverse()
    return route


def nearest_waypoint(config: dict[str, Any], pose: Pose2D) -> str:
    best_name = ""
    best_distance = float("inf")
    for waypoint_name, waypoint_item in config["waypoints"].items():
        waypoint_pose = pose_from_mapping(waypoint_item["pose"])
        distance = pose.point.distance_to(waypoint_pose.point)
        if distance < best_distance:
            best_name = waypoint_name
            best_distance = distance
    if not best_name:
        raise ValueError("No waypoints configured")
    return best_name
