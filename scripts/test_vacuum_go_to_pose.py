#!/usr/bin/env python3
#给一个目标坐标 x/y，让机器人自己开过去。
# 1. 连接 RobosimClient
# 2. 读取当前 pose
# 3. quaternion -> yaw
# 4. 根据目标点算 linear/angular
# 5. 调用你已经验证过的 base drive 公式发左右轮速度


from __future__ import annotations

import argparse
import math
import time

from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.tools.client import RobosimClient
#固定参数
LEFT_JOINT = "rv_left_wheel_joint"
RIGHT_JOINT = "rv_right_wheel_joint"
GROUP = "base_wheels"

WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.32

#角度归一化
def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

#quaternion 转 yaw，函数接收顺序就是：x, y, z, w
def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

#读取当前 x/y/yaw；把 gRPC 返回的 pose 简化成：x/y/yaw
def get_xy_yaw(client: RobosimClient) -> tuple[float, float, float]:
    pose_stamped = client.mobility.get_robot_pose_in_map()
    pose = pose_stamped.pose

    px = pose.position.x
    py = pose.position.y

    q = pose.orientation
    yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    return px, py, yaw

#底盘速度换算成左右轮
def wheel_speeds(linear: float, angular: float) -> tuple[float, float]:
    left_linear = linear - angular * WHEEL_BASE / 2.0
    right_linear = linear + angular * WHEEL_BASE / 2.0

    left_joint = left_linear / WHEEL_RADIUS
    right_joint = right_linear / WHEEL_RADIUS

    return left_joint, right_joint

#发送速度
def send_base_velocity(client: RobosimClient, linear: float, angular: float) -> None:
    left, right = wheel_speeds(linear, angular)

    client.robot_core.set_joint_target(
        names=[LEFT_JOINT, RIGHT_JOINT],
        data=[left, right],
        mode=core_pb2.JointCommand.ControlMode.VELOCITY,
        jmg_name=GROUP,
    )

#停车
def stop_robot(client: RobosimClient) -> None:
    send_base_velocity(client, 0.0, 0.0)
    client.robot_core.emergency_stop()

#限幅：防止速度太大
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

#核心 go_to_pose
# 1. 读取当前位置
# 2. 算目标方向
# 3. 算当前朝向和目标方向差多少
# 4. 如果方向差太大，先原地转
# 5. 如果方向差不大，边走边修正
# 6. 距离小于 0.12m 就停车
def go_to_pose(
    client: RobosimClient,
    target_x: float,
    target_y: float,
    timeout: float,
) -> bool:
    start_time = time.monotonic()

    k_linear = 0.8
    k_angular = 2.0

    max_linear = 0.25
    max_angular = 1.5

    distance_tolerance = 0.12

    while True:
        if time.monotonic() - start_time > timeout:
            print("Timeout.")
            stop_robot(client)
            return False

        x, y, yaw = get_xy_yaw(client)

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - yaw)

        print(
            f"x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}, "
            f"distance={distance:.3f}, heading_error={heading_error:.3f}"
        )

        if distance < distance_tolerance:
            print("Arrived.")
            stop_robot(client)
            return True

        linear = clamp(k_linear * distance, -max_linear, max_linear)
        angular = clamp(k_angular * heading_error, -max_angular, max_angular)

        if abs(heading_error) > 0.8:
            linear = 0.0

        send_base_velocity(client, linear, angular)

        time.sleep(0.05)

#命令行参数
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser

def main() -> int:
    args = build_parser().parse_args()
    client = RobosimClient(host=args.host, port=args.port)

    try:
        ok = go_to_pose(
            client,
            target_x=args.x,
            target_y=args.y,
            timeout=args.timeout,
        )
        return 0 if ok else 1
    except KeyboardInterrupt:
        print("Interrupted.")
        stop_robot(client)
        return 130
    finally:
        client.close()

if __name__ == "__main__":
    raise SystemExit(main())
