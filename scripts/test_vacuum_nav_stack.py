#!/usr/bin/env python3
"""Plan a grid path, then follow it with lidar safety around the robot vacuum."""

from __future__ import annotations

import argparse
from pathlib import Path

from control_stubs.tools.client import RobosimClient
from robosim.navigation.drive import get_xy_yaw, stop_robot
from robosim.navigation.geometry import Point2D, Pose2D
from robosim.navigation.grid import AStarPlanner
from robosim.navigation.inprocess_client import InProcessRobosimClient
from robosim.navigation.lidar import DEFAULT_SCAN_NAME, fmt_range
from robosim.navigation.locations import (
    DEFAULT_LOCATIONS_FILE,
    load_locations,
    target_pose_for_location,
)
from robosim.navigation.mujoco_map import GridBuildOptions, build_occupancy_grid
from robosim.navigation.path_follower import (
    NavigationParams,
    NavigationStep,
    navigate_with_lidar_safety,
)

DEFAULT_SCENE = Path("drivers_sim/mujoco/assets/worlds/two_bedroom_apartment/scene.xml")


def resolve_target(args: argparse.Namespace) -> Pose2D:
    if args.location:
        config = load_locations(args.locations_file)
        return target_pose_for_location(config, args.location)
    if args.x is None or args.y is None:
        raise ValueError("Use either --location or both --x and --y")
    return Pose2D(float(args.x), float(args.y), 0.0)


def resolve_dry_run_start(args: argparse.Namespace) -> Pose2D:
    if args.start_x is not None and args.start_y is not None:
        return Pose2D(float(args.start_x), float(args.start_y), 0.0)
    config = load_locations(args.locations_file)
    try:
        return target_pose_for_location(config, "start")
    except Exception:
        return Pose2D(2.0, -3.0, 0.0)


def build_grid_options(args: argparse.Namespace) -> GridBuildOptions:
    bounds = tuple(args.bounds) if args.bounds else None
    return GridBuildOptions(
        resolution=args.resolution,
        robot_radius=args.robot_radius,
        safety_margin=args.safety_margin,
        bounds=bounds,
        ignored_geom_names=tuple(args.ignore_geom_name),
        ignored_geom_prefixes=tuple(args.ignore_geom_prefix),
    )


def build_nav_params(args: argparse.Namespace) -> NavigationParams:
    return NavigationParams(
        timeout=args.timeout,
        arrive_tolerance=args.arrive_tolerance,
        waypoint_tolerance=args.waypoint_tolerance,
        lookahead_distance=args.lookahead_distance,
        direct_goal_distance=args.direct_goal_distance,
        stop_distance=args.stop_distance,
        slow_distance=args.slow_distance,
        side_distance=args.side_distance,
        normal_linear=args.normal_linear,
        slow_linear=args.slow_linear,
        max_angular=args.max_angular,
        goal_gain=args.goal_gain,
        avoid_gain=args.avoid_gain,
        turn_speed=args.turn_speed,
        replan_after=args.replan_after,
        replan_cooldown=args.replan_cooldown,
        progress_timeout=args.progress_timeout,
        progress_epsilon=args.progress_epsilon,
        dynamic_mark_distance=args.dynamic_mark_distance,
        dynamic_min_distance=args.dynamic_min_distance,
        dynamic_mark_half_angle_deg=args.dynamic_mark_half_angle_deg,
        dynamic_obstacle_radius=args.dynamic_obstacle_radius,
    )


def print_grid_summary(built_grid) -> None:  # noqa: ANN001
    stats = built_grid.stats
    x_min, x_max, y_min, y_max = stats.bounds
    print(
        "Grid: "
        f"{built_grid.grid.width}x{built_grid.grid.height}, "
        f"resolution={built_grid.grid.resolution:.3f} m, "
        f"bounds=({x_min:.2f},{x_max:.2f},{y_min:.2f},{y_max:.2f})"
    )
    print(
        "Obstacles: "
        f"geoms={stats.obstacle_geoms}, "
        f"raw_cells={stats.raw_occupied_cells}, "
        f"inflated_cells={stats.inflated_occupied_cells}, "
        f"skipped_robot={stats.skipped_robot_geoms}, "
        f"skipped_visual={stats.skipped_visual_geoms}, "
        f"skipped_ignored={stats.skipped_ignored_geoms}, "
        f"skipped_height={stats.skipped_height_geoms}, "
        f"skipped_unsupported={stats.skipped_unsupported_geoms}"
    )


def print_planned_path(planner: AStarPlanner, start: Pose2D, target: Pose2D) -> None:
    result = planner.plan(start.point, target.point)
    print(
        f"Path: raw_cells={len(result.raw_cells)}, "
        f"waypoints={len(result.points)}, "
        f"start_cell={result.start_cell}, goal_cell={result.goal_cell}"
    )
    for index, point in enumerate(result.points, start=1):
        print(f"  {index}. x={point.x:.3f}, y={point.y:.3f}")


def print_step(step: NavigationStep) -> None:
    print(
        f"state={step.state} path={step.target_index + 1}/{step.path_points} "
        f"x={step.pose.x:.3f}, y={step.pose.y:.3f}, yaw={step.pose.yaw:.3f}, "
        f"distance={step.goal_distance:.3f}, heading_error={step.heading_error:.3f}, "
        f"front={fmt_range(step.scan.front)}, left={fmt_range(step.scan.left_front)}, "
        f"right={fmt_range(step.scan.right_front)}, "
        f"closest={fmt_range(step.scan.closest_range)}@{step.scan.closest_angle_deg:.0f}deg, "
        f"linear={step.linear:.3f}, angular={step.angular:.3f}, replans={step.replans}"
    )


def make_step_printer(log_every: int):  # noqa: ANN201
    interval = max(1, log_every)
    counter = 0
    last_state = ""
    last_replans = -1

    def printer(step: NavigationStep) -> None:
        nonlocal counter, last_state, last_replans
        counter += 1
        should_print = (
            counter == 1
            or counter % interval == 0
            or step.state != last_state
            or step.replans != last_replans
        )
        last_state = step.state
        last_replans = step.replans
        if should_print:
            print_step(step)

    return printer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--inprocess", action="store_true")
    parser.add_argument("--locations-file", default=str(DEFAULT_LOCATIONS_FILE))
    parser.add_argument("--location")
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-map", action="store_true")
    parser.add_argument("--scan-name", default=DEFAULT_SCAN_NAME)
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--robot-radius", type=float, default=0.18)
    parser.add_argument("--safety-margin", type=float, default=0.08)
    parser.add_argument("--ignore-geom-name", action="append", default=[])
    parser.add_argument("--ignore-geom-prefix", action="append", default=[])
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--arrive-tolerance", type=float, default=0.12)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.18)
    parser.add_argument("--lookahead-distance", type=float, default=0.35)
    parser.add_argument("--direct-goal-distance", type=float, default=0.55)
    parser.add_argument("--stop-distance", type=float, default=0.30)
    parser.add_argument("--slow-distance", type=float, default=0.60)
    parser.add_argument("--side-distance", type=float, default=0.35)
    parser.add_argument("--normal-linear", type=float, default=0.18)
    parser.add_argument("--slow-linear", type=float, default=0.06)
    parser.add_argument("--max-angular", type=float, default=1.2)
    parser.add_argument("--goal-gain", type=float, default=1.6)
    parser.add_argument("--avoid-gain", type=float, default=0.9)
    parser.add_argument("--turn-speed", type=float, default=0.8)
    parser.add_argument("--replan-after", type=float, default=1.5)
    parser.add_argument("--replan-cooldown", type=float, default=2.0)
    parser.add_argument("--progress-timeout", type=float, default=3.0)
    parser.add_argument("--progress-epsilon", type=float, default=0.03)
    parser.add_argument("--dynamic-mark-distance", type=float, default=1.2)
    parser.add_argument("--dynamic-min-distance", type=float, default=0.22)
    parser.add_argument("--dynamic-mark-half-angle-deg", type=float, default=120.0)
    parser.add_argument("--dynamic-obstacle-radius", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target_pose = resolve_target(args)
    target = Point2D(target_pose.x, target_pose.y)
    built_grid = build_occupancy_grid(args.scene, build_grid_options(args))
    planner = AStarPlanner(built_grid.grid)

    print_grid_summary(built_grid)

    if args.dry_run:
        start = resolve_dry_run_start(args)
        print(f"Start: x={start.x:.3f}, y={start.y:.3f}")
        print(f"Goal: x={target_pose.x:.3f}, y={target_pose.y:.3f}")
        result = planner.plan(start.point, target)
        print_planned_path(planner, start, target_pose)
        if args.print_map:
            print(
                built_grid.grid.render_ascii(
                    path=result.raw_cells,
                    start=result.start_cell,
                    goal=result.goal_cell,
                )
            )
        return 0

    if args.inprocess:
        client = InProcessRobosimClient(args.scene, headless=True)
    else:
        client = RobosimClient(host=args.host, port=args.port)
    try:
        current = get_xy_yaw(client)
        print(f"Robot pose: x={current.x:.3f}, y={current.y:.3f}, yaw={current.yaw:.3f}")
        print(f"Goal: x={target_pose.x:.3f}, y={target_pose.y:.3f}")
        print_planned_path(planner, current, target_pose)
        outcome = navigate_with_lidar_safety(
            client=client,
            grid=built_grid.grid,
            target=target,
            params=build_nav_params(args),
            scan_name=args.scan_name,
            feedback=make_step_printer(args.log_every),
        )
        print(
            f"Navigation completed: ok={outcome.ok}, replans={outcome.replans}, "
            f"final_x={outcome.final_pose.x:.3f}, final_y={outcome.final_pose.y:.3f}, "
            f"final_distance={outcome.final_distance:.3f}, "
            f"track_steps={outcome.track_steps}, slow_steps={outcome.slow_steps}, "
            f"blocked_steps={outcome.blocked_steps}"
        )
        return 0 if outcome.ok else 1
    except KeyboardInterrupt:
        print("Interrupted.")
        stop_robot(client)
        return 130
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
