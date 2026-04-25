"""Abstract base class for simulator backends."""

from abc import ABC, abstractmethod
from typing import Iterator

from control_stubs.common_pb2 import JointState, PoseStamped
from control_stubs.mobility_ai_pb2 import NavGoal, TaskFeedback
from control_stubs.robot_core_pb2 import (
    EndEffectorState,
    JointCommand,
    RobotSpecification,
    ServoCommand,
)
from control_stubs.sensing_pb2 import SensorData, SensorMetaList
from robosim.core.capabilities import Capability


class SimulatorBackend(ABC):
    """Abstract base class for simulator backends.

    Subclasses must implement all abstract methods. Unsupported capabilities
    should raise NotImplementedError.
    """

    @property
    @abstractmethod
    def capabilities(self) -> Capability:
        """Return the capabilities of this backend."""
        raise NotImplementedError

    @property
    @abstractmethod
    def robot_name(self) -> str:
        """Return the name of the robot being simulated."""
        raise NotImplementedError

    @property
    @abstractmethod
    def headless_mode(self) -> bool:
        """Return whether the backend is in headless mode"""
        raise NotImplementedError

    @abstractmethod
    def set_headless_mode(self, enabled: bool) -> None:
        """Set headless mode"""
        raise NotImplementedError

    @abstractmethod
    def get_robot_state(self) -> JointState:
        """Get current joint state (name, position, velocity, effort)."""
        raise NotImplementedError

    @abstractmethod
    def get_robot_spec(self) -> RobotSpecification:
        """Get robot specification (joints, groups, end effectors)."""
        raise NotImplementedError

    @abstractmethod
    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: JointCommand.ControlMode,
        group: str | None = None,
    ) -> None:
        """Set joint targets (position/velocity/torque)."""
        raise NotImplementedError
    
    @abstractmethod
    def servo_control_stream(
        self,
        request_iterator: Iterator[ServoCommand],
    ) -> Iterator[JointState]:
        """Servo control stream."""
        raise NotImplementedError

    @abstractmethod
    def get_end_effector_state(self, group: str) -> EndEffectorState:
        """Get end effector pose for a move group."""
        raise NotImplementedError
    
    @abstractmethod
    def get_joint_command_state(self) -> JointState:
        """Get replayable joint action state used as dataset action."""
        raise NotImplementedError

    @abstractmethod
    def list_sensors(self) -> SensorMetaList:
        """List available sensors."""
        raise NotImplementedError

    @abstractmethod
    def get_sensors(self, names: list[str]) -> SensorData:
        """Get sensor data for specified sensors."""
        raise NotImplementedError

    @abstractmethod
    def stream_sensors(self, names: list[str]) -> Iterator[SensorData]:
        """Stream sensor data. Override for streaming support."""
        raise NotImplementedError

    @abstractmethod
    def get_robot_pose_in_map(self) -> PoseStamped:
        """Get robot pose in map frame."""
        raise NotImplementedError

    @abstractmethod
    def navigate_to(self, goal: NavGoal) -> Iterator[TaskFeedback]:
        """Navigate to target pose. Returns iterator for feedback."""
        raise NotImplementedError

    @abstractmethod
    def reset_world(self, seed: int, randomization_params: dict[str, float]) -> None:
        """Reset simulation world."""
        raise NotImplementedError

    @abstractmethod
    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        raise NotImplementedError

    # def pause(self) -> None:
    #     """Pause simulation. You can use step / resume to continue simulation"""
    #     raise NotImplementedError
    
    # def step(self) -> None:
    #     """Step one simulation step"""
    #     raise NotImplementedError
    
    # def resume(self) -> None:
    #     """Resume simulation"""
    #     raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
