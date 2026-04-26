"""Path following with a lidar safety layer."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from control_stubs import sensing_pb2
from control_stubs.tools.client import RobosimClient
from robosim.navigation.drive import get_xy_yaw, send_base_velocity, stop_robot
from robosim.navigation.geometry import Point2D, Pose2D, clamp, normalize_angle
from robosim.navigation.grid import AStarPlanner, OccupancyGrid, PlanningResult
from robosim.navigation.lidar import (
    DEFAULT_SCAN_NAME,
    LidarSummary,
    angle_diff_deg,
    get_scan,
    iter_scan_angles,
    summarize_scan,
)


@dataclass(frozen=True, slots=True)
class NavigationParams:
    timeout: float = 60.0
    arrive_tolerance: float = 0.12
    waypoint_tolerance: float = 0.18
    lookahead_distance: float = 0.35
    direct_goal_distance: float = 0.55
    stop_distance: float = 0.30
    slow_distance: float = 0.60
    side_distance: float = 0.35
    normal_linear: float = 0.18
    slow_linear: float = 0.06
    max_angular: float = 1.2
    goal_gain: float = 1.6
    avoid_gain: float = 0.9
    turn_speed: float = 0.8
    loop_period: float = 0.05
    replan_after: float = 1.5
    replan_cooldown: float = 2.0
    progress_timeout: float = 3.0
    progress_epsilon: float = 0.03
    dynamic_mark_distance: float = 1.2
    dynamic_min_distance: float = 0.22
    dynamic_mark_half_angle_deg: float = 120.0
    dynamic_obstacle_radius: float = 0.35


@dataclass(frozen=True, slots=True)
class NavigationStep:
    state: str
    pose: Pose2D
    goal_distance: float
    heading_error: float
    target_index: int
    path_points: int
    scan: LidarSummary
    linear: float
    angular: float
    replans: int


@dataclass(frozen=True, slots=True)
class NavigationOutcome:
    ok: bool
    replans: int
    final_pose: Pose2D
    final_distance: float
    track_steps: int
    slow_steps: int
    blocked_steps: int
    cancelled: bool = False


FeedbackCallback = Callable[[NavigationStep], None]


def navigate_with_lidar_safety(
    client: RobosimClient,
    grid: OccupancyGrid,
    target: Point2D,
    params: NavigationParams | None = None,
    scan_name: str = DEFAULT_SCAN_NAME,
    feedback: FeedbackCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> NavigationOutcome:
    nav_params = params or NavigationParams()
    started = time.monotonic()
    planner = AStarPlanner(grid)
    pose = get_xy_yaw(client)
    plan = planner.plan(pose.point, target)
    path_points = plan.points
    target_index = 0
    replans = 0
    track_steps = 0
    slow_steps = 0
    blocked_steps = 0
    hazard_since: float | None = None
    last_replan_at = started - nav_params.replan_cooldown
    last_progress_at = started
    best_goal_distance = pose.point.distance_to(target)

    while True:
        now = time.monotonic()
        pose = get_xy_yaw(client)
        goal_distance = pose.point.distance_to(target)
        if should_cancel is not None and should_cancel():
            stop_robot(client)
            return NavigationOutcome(
                False,
                replans,
                pose,
                goal_distance,
                track_steps,
                slow_steps,
                blocked_steps,
                cancelled=True,
            )
        if goal_distance <= nav_params.arrive_tolerance:
            stop_robot(client)
            return NavigationOutcome(
                True,
                replans,
                pose,
                goal_distance,
                track_steps,
                slow_steps,
                blocked_steps,
            )
        if now - started > nav_params.timeout:
            stop_robot(client)
            return NavigationOutcome(
                False,
                replans,
                pose,
                goal_distance,
                track_steps,
                slow_steps,
                blocked_steps,
            )

        scan = summarize_scan(get_scan(client, scan_name))
        if goal_distance < best_goal_distance - nav_params.progress_epsilon:
            best_goal_distance = goal_distance
            last_progress_at = now

        front_hazard = scan.front < nav_params.slow_distance
        side_hazard = (
            scan.left_front < nav_params.side_distance
            or scan.right_front < nav_params.side_distance
        )
        progress_stalled = now - last_progress_at >= nav_params.progress_timeout
        hazard = front_hazard or (side_hazard and progress_stalled)
        if hazard:
            if hazard_since is None:
                hazard_since = now
        else:
            hazard_since = None

        replan_due_to_hazard = (
            hazard_since is not None
            and now - hazard_since >= nav_params.replan_after
        )
        replan_due_to_progress = hazard and progress_stalled
        replan_cooldown_elapsed = now - last_replan_at >= nav_params.replan_cooldown
        if replan_cooldown_elapsed and (replan_due_to_hazard or replan_due_to_progress):
            replanned = _try_dynamic_replan(
                planner=planner,
                base_grid=grid,
                pose=pose,
                target=target,
                scan_name=scan_name,
                client=client,
                params=nav_params,
            )
            if replanned is not None:
                plan = replanned
                path_points = plan.points
                target_index = _advance_target_index(
                    pose.point,
                    path_points,
                    _closest_path_index(pose.point, path_points),
                    nav_params.waypoint_tolerance,
                )
                replans += 1
                last_replan_at = now
                last_progress_at = now
                best_goal_distance = goal_distance
            hazard_since = now

        target_index = _advance_target_index(
            pose.point,
            path_points,
            target_index,
            nav_params.waypoint_tolerance,
        )
        target_point = _lookahead_target(
            pose.point,
            path_points,
            target_index,
            nav_params.lookahead_distance,
        )
        direct_goal_is_clear = (
            goal_distance <= nav_params.direct_goal_distance
            and _line_is_free(grid, pose.point, target)
        )
        if direct_goal_is_clear:
            target_point = target
            target_index = len(path_points) - 1

        dx = target_point.x - pose.x
        dy = target_point.y - pose.y
        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - pose.yaw)
        goal_angular = nav_params.goal_gain * heading_error
        avoid_angular = nav_params.avoid_gain * _obstacle_bias(
            scan.left_front,
            scan.right_front,
            nav_params.side_distance,
        )

        if scan.front < nav_params.stop_distance:
            state = "blocked"
            blocked_steps += 1
            linear = 0.0
            angular = (
                nav_params.turn_speed
                if scan.left_front >= scan.right_front
                else -nav_params.turn_speed
            )
        elif scan.front < nav_params.slow_distance:
            state = "slow"
            slow_steps += 1
            linear = nav_params.slow_linear
            angular = goal_angular + avoid_angular
        else:
            state = "track"
            track_steps += 1
            linear = min(nav_params.normal_linear, max(nav_params.slow_linear, 0.6 * goal_distance))
            angular = goal_angular + avoid_angular

        if abs(heading_error) > 1.0 and scan.front >= nav_params.stop_distance:
            linear = 0.0

        angular = clamp(angular, -nav_params.max_angular, nav_params.max_angular)
        if feedback is not None:
            feedback(
                NavigationStep(
                    state=state,
                    pose=pose,
                    goal_distance=goal_distance,
                    heading_error=heading_error,
                    target_index=target_index,
                    path_points=len(path_points),
                    scan=scan,
                    linear=linear,
                    angular=angular,
                    replans=replans,
                )
            )

        send_base_velocity(client, linear, angular)
        time.sleep(nav_params.loop_period)


def _advance_target_index(
    pose: Point2D,
    points: list[Point2D],
    current_index: int,
    tolerance: float,
) -> int:
    index = min(current_index, max(0, len(points) - 1))
    while index < len(points) - 1:
        current_distance = pose.distance_to(points[index])
        next_distance = pose.distance_to(points[index + 1])
        if current_distance <= tolerance or next_distance + tolerance * 0.5 < current_distance:
            index += 1
            continue
        break
    return index


def _closest_path_index(pose: Point2D, points: list[Point2D]) -> int:
    if not points:
        return 0
    return min(range(len(points)), key=lambda index: pose.distance_to(points[index]))


def _line_is_free(grid: OccupancyGrid, start: Point2D, goal: Point2D) -> bool:
    start_cell = grid.world_to_grid(start)
    goal_cell = grid.world_to_grid(goal)
    return grid.line_is_free(start_cell, goal_cell)


def _lookahead_target(
    pose: Point2D,
    points: list[Point2D],
    current_index: int,
    lookahead_distance: float,
) -> Point2D:
    if not points:
        raise ValueError("Path has no points")
    for index in range(current_index, len(points)):
        if pose.distance_to(points[index]) >= lookahead_distance:
            return points[index]
    return points[-1]


def _obstacle_bias(left_min: float, right_min: float, side_distance: float) -> float:
    left_pressure = max(0.0, side_distance - left_min)
    right_pressure = max(0.0, side_distance - right_min)
    return right_pressure - left_pressure


def _try_dynamic_replan(
    planner: AStarPlanner,
    base_grid: OccupancyGrid,
    pose: Pose2D,
    target: Point2D,
    scan_name: str,
    client: RobosimClient,
    params: NavigationParams,
) -> PlanningResult | None:
    scan = get_scan(client, scan_name)
    dynamic_grid = base_grid.copy()
    for point in _dynamic_obstacle_points(scan, pose, params):
        dynamic_grid.mark_disc(point, params.dynamic_obstacle_radius)
    dynamic_planner = AStarPlanner(dynamic_grid, allow_diagonal=planner.allow_diagonal)
    try:
        return dynamic_planner.plan(pose.point, target)
    except ValueError:
        return None


def _dynamic_obstacle_points(
    scan: sensing_pb2.LidarScan,
    pose: Pose2D,
    params: NavigationParams,
) -> list[Point2D]:
    points: list[Point2D] = []
    for ray_angle, distance in iter_scan_angles(scan):
        if distance < params.dynamic_min_distance or distance > params.dynamic_mark_distance:
            continue
        if abs(angle_diff_deg(math.degrees(ray_angle), 0.0)) > params.dynamic_mark_half_angle_deg:
            continue
        world_angle = pose.yaw + ray_angle
        points.append(
            Point2D(
                x=pose.x + distance * math.cos(world_angle),
                y=pose.y + distance * math.sin(world_angle),
            )
        )
    return points
