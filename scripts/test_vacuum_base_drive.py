#!/usr/bin/env python3
#作用：把 linear / angular 这种底盘速度，换算成 left_wheel / right_wheel 速度，然后发给 RoboSim。
from __future__ import annotations

# argparse     用来读取命令行参数，比如 --port
# time         用来让机器人运动几秒
# core_pb2     用来拿 VELOCITY 这个控制模式
# RobosimClient 用来连接 gRPC server
import argparse
import time

from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.tools.client import RobosimClient

#固定参数LEFT_JOINT / RIGHT_JOINT 是左右轮名字
# GROUP 是 SRDF 里写的 base_wheels
# WHEEL_RADIUS 是轮子半径，约 0.05
# WHEEL_BASE 是左右轮距离，约 0.32

LEFT_JOINT = "rv_left_wheel_joint"
RIGHT_JOINT = "rv_right_wheel_joint"
GROUP = "base_wheels"

WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.32

#速度换算函数:linear/angular -> 左右轮速度
def wheel_speeds(linear: float,angular:float,inver_angular:bool)->tuple[float,float]:
    if inver_angular:
        angular = -angular
    
    left_linear = linear - angular*WHEEL_BASE/2.0
    right_linear = linear + angular * WHEEL_BASE / 2.0

    left_joint = left_linear / WHEEL_RADIUS
    right_joint = right_linear / WHEEL_RADIUS

    return left_joint,right_joint

#停车函数：负责停车。以后每次测试结束，都要调用它
def stop_robot(client: RobosimClient) -> None:
    client.robot_core.set_joint_target(
        names=[LEFT_JOINT, RIGHT_JOINT],
        data=[0.0, 0.0],
        mode=core_pb2.JointCommand.ControlMode.VELOCITY,
        jmg_name=GROUP,
    )
    client.robot_core.emergency_stop()

#发送底盘速度的函数：
def drive_for(
    client: RobosimClient,
    linear: float,
    angular: float,
    duration: float,
    invert_angular: bool,
) -> None:
# 算左右轮速度
    left, right = wheel_speeds(linear, angular, invert_angular)
# 打印出来
    print(f"linear={linear:.3f}, angular={angular:.3f}")
    print(f"left_joint={left:.3f}, right_joint={right:.3f}")
# 发给机器人
    client.robot_core.set_joint_target(
        names=[LEFT_JOINT, RIGHT_JOINT],
        data=[left, right],
        mode=core_pb2.JointCommand.ControlMode.VELOCITY,
        jmg_name=GROUP,
    )
# 等几秒
    time.sleep(duration)
# 停车
    stop_robot(client)

#命令行参数
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--linear", type=float, default=0.25)
    parser.add_argument("--angular", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--invert-angular", action="store_true")
    return parser

#主函数
def main()->int:
    args = build_parser().parse_args()

#依次测试前进、后退、原地左转/右转、向左/右弧线前进
    test = [
        ("forward", args.linear, 0.0),
        ("backward", -args.linear, 0.0),
        ("turn left", 0.0, args.angular),
        ("turn right", 0.0, -args.angular),
        ("forward left arc", args.linear * 0.7, args.angular * 0.5),
        ("forward right arc", args.linear * 0.7, -args.angular * 0.5),
    ]

    client = RobosimClient(host=args.host, port=args.port)
    
    try:
        for name, linear, angular in test:
            print()
            print("=" * 72)
            print(f"Test: {name}")
            input("Press Enter to start this test...")

            client.simulation.reset_world(seed=0, randomization_params={})
            time.sleep(0.5)

            drive_for(
                client,
                linear=linear,
                angular=angular,
                duration=args.duration,
                invert_angular=args.invert_angular,
            )

            input("Observe the result, then press Enter for next test...")
    finally:
        try:
            stop_robot(client)
        except Exception:
            pass
        client.close()

    return 0

if __name__=="__main__":
    raise SystemExit(main())