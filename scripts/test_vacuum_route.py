#!/usr/bin/env python3
# 读 locations.yaml
# 根据 --location 找到目标地点
# 拿到目标地点的 nearest_waypoint
# 从 wp_start 规划到目标 waypoint
# 打印路线
# argparse  读取命令行参数，比如 --location test_north
# deque     做 BFS 找路用
# yaml      读取 locations.yaml
from __future__ import annotations

import argparse
from collections import deque

import yaml

from control_stubs.tools.client import RobosimClient
from test_vacuum_navigate_location import get_xy_yaw, go_to_pose, stop_robot



DEFAULT_LOCATIONS_FILE = (
    "drivers_sim/mujoco/assets/worlds/two_bedroom_apartment/locations.yaml"
)
#读取YAML并转换成字典
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

#根据地点名找location
def find_location(config: dict, query: str) -> dict:
    locations = config["locations"]

    if query in locations:
        return locations[query]

    for name, item in locations.items():
        aliases = item.get("aliases", [])
        if query in aliases:
            return item

    raise ValueError(f"Unknown location: {query}")

#把edges变成图
def build_graph(edges: list[list[str]]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}

    for a, b in edges:
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []

        graph[a].append(b)
        graph[b].append(a)

    return graph

#BFS找路线从 start 出发，沿着 edges 一层层找，找到 goal 后，返回一串 waypoint
#例如：wp_start -> wp_center -> wp_north
def find_route(
    graph: dict[str, list[str]],
    start: str,
    goal: str,
) -> list[str]:
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

#读取 waypoint 坐标的函数
def get_waypoint_pose(config: dict, waypoint_name: str) -> dict:
    waypoints = config["waypoints"]

    if waypoint_name not in waypoints:
        raise ValueError(f"Unknown waypoint: {waypoint_name}")

    return waypoints[waypoint_name]["pose"]

#新增print_route函数
def print_route_with_poses(config: dict, route: list[str]) -> None:
    print("Route: " + " -> ".join(route))
    print("Route poses:")

    for index, waypoint_name in enumerate(route, start=1):
        pose = get_waypoint_pose(config, waypoint_name)
        x = float(pose["x"])
        y = float(pose["y"])
        yaw = float(pose.get("yaw", 0.0))

        print(
            f"  {index}. {waypoint_name}: "
            f"x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}"
        )


#新增距离函数
def distance_to_waypoint(
    robot_x: float,
    robot_y: float,
    waypoint_pose: dict,
) -> float:
    dx = float(waypoint_pose["x"]) - robot_x
    dy = float(waypoint_pose["y"]) - robot_y
    return (dx * dx + dy * dy) ** 0.5

#找最近的waypoint
def find_nearest_waypoint(
    config: dict,
    robot_x: float,
    robot_y: float,
) -> str:
    best_name = ""
    best_distance = float("inf")

    for waypoint_name, waypoint_item in config["waypoints"].items():
        pose = waypoint_item["pose"]
        distance = distance_to_waypoint(robot_x, robot_y, pose)

        if distance < best_distance:
            best_name = waypoint_name
            best_distance = distance

    print(f"Nearest waypoint: {best_name} ({best_distance:.3f} m)")
    return best_name


#增加 follow_route 函数
def follow_route(
    client: RobosimClient,
    config: dict,
    route: list[str],
    timeout_per_waypoint: float,
) -> bool:
    for waypoint_name in route[1:]:
        pose = get_waypoint_pose(config, waypoint_name)

        target_x = float(pose["x"])
        target_y = float(pose["y"])

        print("=" * 72)
        print(f"Going to waypoint: {waypoint_name}")
        print(f"Target: x={target_x}, y={target_y}")

        ok = go_to_pose(
            client,
            target_x=target_x,
            target_y=target_y,
            timeout=timeout_per_waypoint,
        )

        if not ok:
            print(f"Failed at waypoint: {waypoint_name}")
            return False

    print("Route completed.")
    return True



#命令行参数
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--start-waypoint")
    parser.add_argument("--locations-file", default=DEFAULT_LOCATIONS_FILE)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--timeout-per-waypoint", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser

def main() -> int:
    args = build_parser().parse_args()

    config = load_config(args.locations_file)

    location = find_location(config, args.location)
    goal_waypoint = location["nearest_waypoint"]

    edges = config["edges"]
    graph = build_graph(edges)

    client = None

    try:
        if args.start_waypoint:
            start_waypoint = args.start_waypoint
        else:
            if args.dry_run:
                start_waypoint = "wp_start"
                print("Dry run without --start-waypoint: using wp_start.")
            else:
                client = RobosimClient(host=args.host, port=args.port)
                robot_x, robot_y, _ = get_xy_yaw(client)
                print(f"Robot pose: x={robot_x:.3f}, y={robot_y:.3f}")
                start_waypoint = find_nearest_waypoint(config, robot_x, robot_y)

        route = find_route(
            graph=graph,
            start=start_waypoint,
            goal=goal_waypoint,
        )

        print(f"Start waypoint: {start_waypoint}")
        print(f"Goal waypoint: {goal_waypoint}")
        print_route_with_poses(config, route)

        if args.dry_run:
            return 0

        if client is None:
            client = RobosimClient(host=args.host, port=args.port)

        ok = follow_route(
            client=client,
            config=config,
            route=route,
            timeout_per_waypoint=args.timeout_per_waypoint,
        )
        return 0 if ok else 1

    except KeyboardInterrupt:
        print("Interrupted.")
        if client is not None:
            stop_robot(client)
        return 130
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
