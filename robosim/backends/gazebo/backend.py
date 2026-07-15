"""Gazebo backend implementation using ROS2.

Dynamic sensor discovery by message type, not topic name.
Periodic refresh ensures new sensors are detected and removed sensors are cleaned up.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.action import graph as action_graph
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node, ReentrantCallbackGroup
from rclpy.subscription import Subscription
from rclpy.time import Duration, Time
from rclpy.timer import Timer
from sensor_msgs.msg import Image, Imu, JointState, LaserScan
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener

from control_stubs import common_pb2 as common_pb2
from control_stubs import mobility_ai_pb2
from control_stubs import robot_core_pb2 as core_pb2
from control_stubs import sensing_pb2 as sensing_pb2
from control_stubs.sensing_pb2 import SensorType
from robosim.core.backend import SimulatorBackend
from robosim.core.capabilities import Capability


@dataclass(slots=True)
class SensorRecord:
    topic_name: str
    type: SensorType
    data: JointState | Imu | LaserScan | Odometry | Image | None = None
    stamp_ns: int = -1


MSG_TYPE_TO_SENSOR_TYPE: dict[str, SensorType] = {
    "sensor_msgs/msg/JointState": SensorType.JOINT,
    "sensor_msgs/msg/Imu": SensorType.IMU,
    "sensor_msgs/msg/LaserScan": SensorType.LIDAR,
    "nav_msgs/msg/Odometry": SensorType.ODOM,
    "sensor_msgs/msg/Image": SensorType.CAMERA,
}

ACTION_TYPE_TO_CAP: dict[str, Capability] = {
    "nav2_msgs/action/NavigateToPose": Capability.NAVIGATION,
}

SENSOR_TYPE_TO_CAP: dict[SensorType, Capability] = {
    SensorType.JOINT: Capability.SENSOR_JOINT,
    SensorType.IMU: Capability.SENSOR_IMU,
    SensorType.LIDAR: Capability.SENSOR_LIDAR,
    SensorType.ODOM: Capability.SENSOR_ODOMETRY,
    SensorType.CAMERA: Capability.SENSOR_CAMERA,
}


class GazeboBackend(SimulatorBackend, Node):
    """Gazebo backend using ROS2 for communication.
    """
    DISCOVERY_DELAY = 2.0
    REFRESH_INTERVAL = 5.0

    def __init__(self, robot_name: str = "robot") -> None:
        SimulatorBackend.__init__(self)
        Node.__init__(self, f"robosim_gazebo_backend_{robot_name}")
        self.get_logger().info(f"Initializing GazeboBackend for robot: {robot_name}")

        self._robot_name = robot_name
        self._capabilities = Capability.NONE

        # One lock for backend state
        self._state_lock = threading.RLock()

        # Explicit callback group: do not use the node default group
        self._cb_group = ReentrantCallbackGroup()

        self._discovered_sensors: dict[str, SensorRecord] = {}
        self._cur_subscriptions: dict[str, Subscription] = {}
        self._known_topics: set[str] = set()

        self.get_logger().debug("Initializing TF listener...")
        self._init_tf()
        self.get_logger().debug("Initializing publishers...")
        self._init_publishers()

        self._nav_tracked_action_name: Optional[str] = None
        self._nav_client: Optional[ActionClient] = None

        # Spin this node on its own executor thread
        self._executor = MultiThreadedExecutor(
            num_threads=max(2, (os.cpu_count() or 2)),
        )
        self._executor.add_node(self)
        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            name=f"{self.get_name()}_ros_spin",
            daemon=True,
        )
        self._executor_thread.start()

        self.get_logger().info("Discovering and subscribing to sensor topics...")
        self._discover_and_subscribe()

        self._refresh_timer: Optional[Timer] = self.create_timer(
            self.REFRESH_INTERVAL,
            self._periodic_discovery,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"GazeboBackend initialized. Refresh interval: {self.REFRESH_INTERVAL}s"
        )

    def _init_tf(self) -> None:
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.get_logger().debug("TF listener created")

    def _init_publishers(self) -> None:
        self._pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.get_logger().debug("Command velocity publisher created: /cmd_vel")
    
    def _topic2sensor_name(self, topic_name: str) -> str:
        return topic_name.strip("/").replace("/", "_")

    def _periodic_discovery(self) -> None:
        try:
            self._discover_and_subscribe()
        except Exception as e:
            self.get_logger().error(f"Periodic discovery failed: {e}", throttle_duration_sec=60)

    def _discover_and_subscribe(self) -> None:
        topic_info = self.get_topic_names_and_types()
        all_actions = action_graph.get_action_names_and_types(node=self)

        current_topics: set[str] = set()
        new_topics: list[tuple[str, str]] = []

        for topic_name, msg_types in topic_info:
            if not msg_types:
                continue
            msg_type = msg_types[0]
            current_topics.add(topic_name)

            with self._state_lock:
                already_subscribed = topic_name in self._cur_subscriptions

            if msg_type in MSG_TYPE_TO_SENSOR_TYPE.keys() and not already_subscribed:
                new_topics.append((topic_name, msg_type))

        # NOTE: only find the very first navigation action
        current_nav_action_name: Optional[str] = None
        for action_name, action_types in all_actions:
            if not action_types:
                continue
            action_type = action_types[0]
            if action_type in ACTION_TYPE_TO_CAP.keys():
                current_nav_action_name = action_name
                break
        if current_nav_action_name is None:
            if self._nav_client is not None:
                # nav action disappear: cleanup the client
                with self._state_lock:
                    self._nav_client.destroy()
                    self._nav_client = None
                    self._nav_tracked_action_name = None
        else:
            if self._nav_client is None:
                # nav action appear: create the client
                with self._state_lock:
                    self._nav_client = ActionClient(
                        self, NavigateToPose,
                        current_nav_action_name.strip("/"),
                        callback_group=self._cb_group
                    )
                    self._nav_tracked_action_name = current_nav_action_name

        # Subscribe outside the lock
        for topic_name, msg_type in new_topics:
            self._subscribe_sensor(topic_name, msg_type)

        # Remove deleted topics without holding the lock during destroy_subscription()
        with self._state_lock:
            removed = self._known_topics - current_topics
            subs_to_destroy: list[Subscription] = []

            for topic_name in removed:
                self.get_logger().debug(f"Removing deleted sensor topic: {topic_name}")
                self._discovered_sensors.pop(topic_name, None)
                sub = self._cur_subscriptions.pop(topic_name, None)
                if sub is not None:
                    subs_to_destroy.append(sub)

            self._known_topics = current_topics

        for sub in subs_to_destroy:
            self.destroy_subscription(sub)

        if new_topics:
            self.get_logger().info(f"Discovered {len(new_topics)} new sensor topics")
            self._detect_capabilities()

        self.get_logger().debug(
            f"Discovery complete. Total sensors: {len(self._discovered_sensors)}, "
            f"Total topics tracked: {len(current_topics)}"
        )

    def _subscribe_sensor(self, topic_name: str, msg_type: str) -> None:
        sensor_type = MSG_TYPE_TO_SENSOR_TYPE.get(msg_type)
        if sensor_type is None:
            self.get_logger().warn(f"Unknown sensor type {msg_type} for topic {topic_name}")
            return

        self.get_logger().info(
            f"Subscribing to sensor topic: {topic_name} (type: {sensor_type}, msg: {msg_type})"
        )

        with self._state_lock:
            if topic_name in self._cur_subscriptions:
                return
            self._discovered_sensors[topic_name] = SensorRecord(
                topic_name=topic_name, type=sensor_type)

        match MSG_TYPE_TO_SENSOR_TYPE[msg_type]:
            case SensorType.JOINT:
                sub = self.create_subscription(
                    JointState, topic_name,
                    lambda m, tn=topic_name: self._on_joint_state(m, tn),
                    10, callback_group=self._cb_group
                )
            case SensorType.IMU:
                sub = self.create_subscription(
                    Imu, topic_name,
                    lambda m, tn=topic_name: self._on_imu(m, tn),
                    10, callback_group=self._cb_group
                )
            case SensorType.LIDAR:
                sub = self.create_subscription(
                    LaserScan, topic_name,
                    lambda m, tn=topic_name: self._on_scan(m, tn),
                    10, callback_group=self._cb_group
                )
            case SensorType.ODOM:
                sub = self.create_subscription(
                    Odometry, topic_name,
                    lambda m, tn=topic_name: self._on_odom(m, tn),
                    10, callback_group=self._cb_group
                )
            case SensorType.CAMERA:
                sub = self.create_subscription(
                    Image, topic_name,
                    lambda m, tn=topic_name: self._on_image(m, tn),
                    10, callback_group=self._cb_group
                )
            case _:
                return

        with self._state_lock:
            self._cur_subscriptions[topic_name] = sub

    def _stamp_ns(self, msg: Any) -> int:
        header = getattr(msg, "header", None)
        if header is None:
            return -1
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def _update_sensor(
        self, topic_name: str,
        msg: JointState | Imu | LaserScan | Odometry | Image,
        sensor_type: SensorType
    ) -> None:
        stamp_ns = self._stamp_ns(msg)
        with self._state_lock:
            rec = self._discovered_sensors.get(topic_name)
            if rec is None:
                return
            if stamp_ns >= rec.stamp_ns:
                rec.topic_name = topic_name
                rec.type = sensor_type
                rec.data = msg
                rec.stamp_ns = stamp_ns

    def _on_joint_state(self, msg: JointState, topic_name: str) -> None:
        self.get_logger().debug(f"JointState received from {topic_name}")
        self._update_sensor(topic_name, msg, SensorType.JOINT)

    def _on_imu(self, msg: Imu, topic_name: str) -> None:
        self.get_logger().debug(f"IMU data received from {topic_name}")
        self._update_sensor(topic_name, msg, SensorType.IMU)

    def _on_scan(self, msg: LaserScan, topic_name: str) -> None:
        self.get_logger().debug(f"LaserScan received from {topic_name}, ranges: {len(msg.ranges)}")
        self._update_sensor(topic_name, msg, SensorType.LIDAR)

    def _on_odom(self, msg: Odometry, topic_name: str) -> None:
        self.get_logger().debug(
            f"Odometry received from {topic_name}, pos: "
            f"({msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f})"
        )
        self._update_sensor(topic_name, msg, SensorType.ODOM)

    def _on_image(self, msg: Image, topic_name: str) -> None:
        self.get_logger().debug(f"Image received from {topic_name}, size: {msg.width}x{msg.height}")
        self._update_sensor(topic_name, msg, SensorType.CAMERA)

    def _detect_capabilities(self) -> None:
        """Detect capabilities based on dynamically discovered sensors."""
        caps = Capability.EMERGENCY_STOP

        with self._state_lock:
            for _, info in self._discovered_sensors.items():
                sensor_type = info.type
                sensor_cap = SENSOR_TYPE_TO_CAP[sensor_type]
                if sensor_cap is not None:
                    caps |= sensor_cap
                    if sensor_cap == Capability.SENSOR_JOINT and info.data is not None:
                        caps |= Capability.JOINT_READ
                        caps |= Capability.JOINT_WRITE

        # NOTE: Test navigation capabilities: Search for topics related to Nav2
        if self._nav_tracked_action_name is not None:
            caps |= Capability.NAVIGATION

        self._capabilities = caps
        self.get_logger().info(f"Capabilities detected: {caps.name} (0b{caps.value:08b})")

    @property
    def capabilities(self) -> Capability:
        return self._capabilities

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def headless_mode(self) -> bool:
        return False

    def set_headless_mode(self, enabled: bool) -> None:
        if enabled:
            raise NotImplementedError("Headless mode not supported for Gazebo")

    def get_robot_state(self) -> common_pb2.JointState:
        """Get joint state - get the first JointState topic."""
        with self._state_lock:
            records: list[JointState] = [
                rec.data
                for rec in self._discovered_sensors.values()
                if rec.type == SensorType.JOINT and rec.data is not None
                and isinstance(rec.data, JointState)
            ]
        for js in records:
            return GazeboBackend._build_joint_state(js)
        
        self.get_logger().warning("No joint state found")
        return common_pb2.JointState()

    # TODO: implement this later
    def get_robot_spec(self) -> core_pb2.RobotSpecification:
        joint_state = self.get_robot_state()
        joint_names = list(joint_state.name)
        if not joint_names:
            return core_pb2.RobotSpecification(robot_name=self._robot_name)
        return core_pb2.RobotSpecification(
            robot_name=self._robot_name,
            joints=[
                core_pb2.JointLimit(
                    name=name,
                    type="unknown",
                    jmg_names=["all"],
                    lower_limit=0.0,
                    upper_limit=0.0,
                    velocity_limit=0.0,
                    acceleration_limit=0.0,
                    effort_limit=0.0,
                )
                for name in joint_names
            ],
            joint_model_groups=[
                core_pb2.JointModelGroupSpec(
                    name="all",
                    joint_names=joint_names,
                )
            ],
        )

    def get_joint_command_state(self) -> common_pb2.JointState:
        return common_pb2.JointState()

    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: core_pb2.JointCommand.ControlMode,
        group: str | None = None,
    ) -> None:
        if not (self._capabilities & Capability.JOINT_WRITE):
            self.get_logger().error("Joint write not supported by this backend")
            raise NotImplementedError("Joint write not supported")

        raise NotImplementedError("Joint target setting not implemented")

    def servo_control_stream(
        self, request_iterator: Iterator[core_pb2.ServoCommand]) -> Iterator[common_pb2.JointState]:
        raise NotImplementedError("Servo control stream not implemented")

    def get_end_effector_state(self, group: str) -> core_pb2.EndEffectorState:
        raise NotImplementedError("End effector state not implemented")

    def list_sensors(self) -> sensing_pb2.SensorMetaList:
        """Return the list of dynamically discovered sensors."""
        with self._state_lock:
            items = list(self._discovered_sensors.items())

        return sensing_pb2.SensorMetaList(entries=[
            sensing_pb2.SensorMetaList.SensorMeta(
                name=self._topic2sensor_name(topic_name),
                type=rec.type,
            )
            for topic_name, rec in items
        ])

    def get_sensors(self, names: list[str]) -> sensing_pb2.SensorData:
        """Get sensor data by name, supports multiple matching modes."""
        self.get_logger().debug(f"Getting sensors: {names}")
        imus: list[sensing_pb2.ImuData] = []
        lidars: list[sensing_pb2.LidarScan] = []
        images: list[sensing_pb2.CameraImage] = []
        odometries: list[sensing_pb2.OdometryData] = []
        joints: list[sensing_pb2.JointData] = []

        def matches(name: str) -> bool:
            """Check if the name matches the requested list."""
            topic_name = self._topic2sensor_name(name)
            return name in names or topic_name in names

        matched_count = 0
        with self._state_lock:
            items = list(self._discovered_sensors.items())
        
        for topic_name, info in items:
            sensor_type = info.type
            data = info.data

            name = self._topic2sensor_name(topic_name)
            if names:
                if not matches(name) and not matches(topic_name):
                    continue

            if data is None:
                self.get_logger().debug(f"Sensor {topic_name} has no data yet")
                continue

            matched_count += 1
            match sensor_type:
                case SensorType.IMU:
                    assert isinstance(data, Imu)
                    imus.append(GazeboBackend._build_imu_data(data, name))
                case SensorType.LIDAR:
                    assert isinstance(data, LaserScan)
                    lidars.append(GazeboBackend._build_laser_scan_data(data, name))
                case SensorType.ODOM:
                    assert isinstance(data, Odometry)
                    odometries.append(GazeboBackend._build_odometry_data(data, name))
                case SensorType.CAMERA:
                    assert isinstance(data, Image)
                    images.append(GazeboBackend._build_image_data(data, name))
                case SensorType.JOINT:
                    assert isinstance(data, JointState)
                    joints.append(sensing_pb2.JointData(
                            name=name, joint_states=GazeboBackend._build_joint_state(data)))

        self.get_logger().debug(
            f"Returning sensor data: {len(imus)} imus, {len(lidars)} lidars, "
            f"{len(images)} images, {len(odometries)} odometries, {len(joints)} joints "
            f"(matched {matched_count} requested)"
        )
        return sensing_pb2.SensorData(
            imus=imus,
            lidars=lidars,
            images=images,
            odometries=odometries,
            joints=joints
        )
    
    def stream_sensors(self, names: list[str]) -> Iterator[sensing_pb2.SensorData]:
        raise NotImplementedError("Streaming not supported for now (to be implemented)")

    def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
        if not (self._capabilities & Capability.NAVIGATION):
            self.get_logger().error("Navigation not supported")
            raise NotImplementedError("Navigation not supported")

        try:
            transform = self._tf_buffer.lookup_transform(
                "map", "base_link", Time(), Duration(seconds=2)
            )

            pose_stamped = PoseStamped(
                header=transform.header,
                pose=Pose(
                    position=Point(
                        x=transform.transform.translation.x,
                        y=transform.transform.translation.y,
                        z=transform.transform.translation.z,
                    ),
                    orientation=Quaternion(
                        x=transform.transform.rotation.x,
                        y=transform.transform.rotation.y,
                        z=transform.transform.rotation.z,
                        w=transform.transform.rotation.w,
                    )
                )
            )

            self.get_logger().debug(
                f"Robot pose in map: "
                f"({pose_stamped.pose.position.x:.3f}, {pose_stamped.pose.position.y:.3f})"
            )
            return GazeboBackend._build_pose_stamped(pose_stamped)
        except Exception as e:
            self.get_logger().warn(f"Failed to lookup transform map->base_link: {e}")
            raise

    # NOTE: use polling instead of event-driven approach
    def navigate_to(self, goal: mobility_ai_pb2.NavGoal) -> Iterator[mobility_ai_pb2.TaskFeedback]:
        if not (self._capabilities & Capability.NAVIGATION) or self._nav_client is None:
            self.get_logger().error("Navigation not supported")
            raise NotImplementedError("Navigation not supported")

        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server unavailable")
            raise TimeoutError("Nav2 action server unavailable")

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = goal.target_frame or "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = goal.target_pose.position.x
        goal_pose.pose.position.y = goal.target_pose.position.y
        goal_pose.pose.position.z = goal.target_pose.position.z
        goal_pose.pose.orientation = Quaternion(
            x=goal.target_pose.orientation.x,
            y=goal.target_pose.orientation.y,
            z=goal.target_pose.orientation.z,
            w=goal.target_pose.orientation.w,
        )

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        goal_done = threading.Event()
        pending_feedbacks: deque[Any] = deque(maxlen=1)

        final_status_code = [common_pb2.STATUS_UNKNOWN]
        final_message = [""]

        goal_handle: ClientGoalHandle | None = None

        # ---- Correct feedback callback ----
        def _on_feedback(feedback_msg: Any) -> None:
            pending_feedbacks.append(feedback_msg.feedback)

        # ---- Correct goal response callback ----
        def _on_goal_response(future: Any) -> None:
            nonlocal goal_handle

            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                final_status_code[0] = common_pb2.STATUS_FAILURE
                final_message[0] = "Goal rejected by navigation server"
                goal_done.set()
                return

            self.get_logger().debug("Navigation goal accepted")

            # Correct result future usage
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(_on_result)

        # ---- Correct result callback ----
        def _on_result(future: Any) -> None:
            result = future.result()

            if result is None:
                final_status_code[0] = common_pb2.STATUS_FAILURE
                final_message[0] = "Navigation result unavailable"
                goal_done.set()
                return

            status = result.status

            if status == GoalStatus.STATUS_SUCCEEDED:
                final_status_code[0] = common_pb2.STATUS_SUCCESS
                final_message[0] = "Navigation succeeded"
            elif status == GoalStatus.STATUS_ABORTED:
                final_status_code[0] = common_pb2.STATUS_FAILURE
                final_message[0] = "Navigation aborted"
            elif status == GoalStatus.STATUS_CANCELED:
                final_status_code[0] = common_pb2.STATUS_PREEMPTED
                final_message[0] = "Navigation canceled"
            else:
                final_status_code[0] = common_pb2.STATUS_FAILURE
                final_message[0] = f"Navigation failed (status={status})"

            goal_done.set()

        # Send goal (correct usage)
        send_goal_future = self._nav_client.send_goal_async(
            goal_msg,
            feedback_callback=_on_feedback
        )
        send_goal_future.add_done_callback(_on_goal_response)

        yield GazeboBackend._make_feedback(
            task_id="",
            status_code=common_pb2.STATUS_RUNNING,
            message="Navigation started",
        )

        # Feedback loop
        while not goal_done.is_set():
            if pending_feedbacks:
                fb = pending_feedbacks.pop()
                distance = getattr(fb, "distance_remaining", 0.0)
                eta = getattr(fb, "estimated_time_remaining", None)
                eta_sec = getattr(eta, "sec", 0) if eta else 0

                yield GazeboBackend._make_feedback(
                    task_id="",
                    status_code=common_pb2.STATUS_RUNNING,
                    message=f"Distance remaining: {distance:.3f}m",
                    eta=eta_sec,
                )

            goal_done.wait(timeout=0.1)

        yield GazeboBackend._make_feedback(
            task_id="",
            status_code=final_status_code[0],
            message=final_message[0],
        )

    def reset_world(self, seed: int, randomization_params: dict[str, float]) -> None:
        del seed, randomization_params
        if not (self._capabilities & Capability.SIMULATION_CONTROL):
            self.get_logger().error("Simulation control not supported")
            raise NotImplementedError("Simulation control not supported")
        raise NotImplementedError("Reset world not implemented")

    def emergency_stop(self) -> None:
        self.get_logger().info("Emergency stop triggered, publishing zero velocity")
        twist = Twist()
        self._pub_cmd_vel.publish(twist)

    def shutdown(self) -> None:
        self.get_logger().info("Shutting down GazeboBackend...")

        if self._refresh_timer is not None:
            self._refresh_timer.cancel()

        try:
            self._executor.remove_node(self)
        except Exception:
            pass

        self._executor.shutdown()
        self._executor_thread.join(timeout=2.0)

        self.destroy_node()
        self.get_logger().info("GazeboBackend shutdown complete")

    @staticmethod
    def _build_header(header: Header) -> common_pb2.Header:
        return common_pb2.Header(
            seq=0,
            timestamp=header.stamp.sec + header.stamp.nanosec * 1e-9,
            frame_id=header.frame_id,
        )

    @staticmethod
    def _build_quaternion(q: Quaternion) -> common_pb2.Quaternion:
        return common_pb2.Quaternion(
            x=q.x,
            y=q.y,
            z=q.z,
            w=q.w,
        )

    @staticmethod
    def _build_point_from_vec3(p: Vector3) -> common_pb2.Point:
        return common_pb2.Point(x=p.x, y=p.y, z=p.z)

    @staticmethod
    def _build_point(p: Point) -> common_pb2.Point:
        return common_pb2.Point(x=p.x, y=p.y, z=p.z)

    @staticmethod
    def _build_pose(pose: Pose) -> common_pb2.Pose:
        return common_pb2.Pose(
            position=GazeboBackend._build_point(pose.position),
            orientation=GazeboBackend._build_quaternion(pose.orientation),
        )
    
    @staticmethod
    def _build_pose_stamped(pose_stamped: PoseStamped) -> common_pb2.PoseStamped:
        return common_pb2.PoseStamped(
            header=GazeboBackend._build_header(pose_stamped.header),
            pose=GazeboBackend._build_pose(pose_stamped.pose),
        )

    @staticmethod
    def _build_twist(twist: Twist) -> common_pb2.Twist:
        return common_pb2.Twist(
            linear=GazeboBackend._build_point_from_vec3(twist.linear),
            angular=GazeboBackend._build_point_from_vec3(twist.angular),
        )

    @staticmethod
    def _build_imu_data(imu: Imu, name: str) -> sensing_pb2.ImuData:
        return sensing_pb2.ImuData(
            header=GazeboBackend._build_header(imu.header),
            name=name,
            orientation=GazeboBackend._build_quaternion(imu.orientation),
            angular_velocity=GazeboBackend._build_point_from_vec3(imu.angular_velocity),
            linear_acceleration=GazeboBackend._build_point_from_vec3(imu.linear_acceleration),
        )
    
    @staticmethod
    def _build_laser_scan_data(laser_scan: LaserScan, name: str) -> sensing_pb2.LidarScan:
        return sensing_pb2.LidarScan(
            header=GazeboBackend._build_header(laser_scan.header),
            name=name,
            angle_min=laser_scan.angle_min,
            angle_max=laser_scan.angle_max,
            angle_increment=laser_scan.angle_increment,
            ranges=list(laser_scan.ranges),
            intensities=list(laser_scan.intensities),
        )
    
    @staticmethod
    def _build_image_data(image: Image, name: str) -> sensing_pb2.CameraImage:
        return sensing_pb2.CameraImage(
            header=GazeboBackend._build_header(image.header),
            name=name,
            width=image.width,
            height=image.height,
            encoding=image.encoding,
            is_bigendian=bool(image.is_bigendian),
            step=image.step,
            # TODO: validate this
            data=bytes(image.data),
        )

    @staticmethod
    def _build_odometry_data(odometry: Odometry, name: str) -> sensing_pb2.OdometryData:
        return sensing_pb2.OdometryData(
            header=GazeboBackend._build_header(odometry.header),
            name=name,
            pose=GazeboBackend._build_pose(odometry.pose.pose),
            twist=GazeboBackend._build_twist(odometry.twist.twist),
        )
    
    @staticmethod
    def _build_joint_state(joint_state: JointState) -> common_pb2.JointState:
        return common_pb2.JointState(
            header=GazeboBackend._build_header(joint_state.header),
            name=list(joint_state.name),
            position=list(joint_state.position),
            velocity=list(joint_state.velocity),
            effort=list(joint_state.effort),
        )
    
    @staticmethod
    def _make_feedback(
        task_id: str,
        status_code: common_pb2.StatusCode,
        message: str,
        eta: int = 0,
        feedback_text: str = "",
    ) -> Any:

        fb = mobility_ai_pb2.TaskFeedback()
        fb.task_id = task_id
        fb.status.code = status_code
        fb.status.message = message
        fb.eta = eta
        fb.feedback_text = feedback_text
        return fb
