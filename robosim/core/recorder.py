from abc import ABC, abstractmethod
from dataclasses import dataclass

from control_stubs.common_pb2 import JointState, Status
from control_stubs.robot_core_pb2 import EndEffectorState
from control_stubs.robot_data_pb2 import RecordInfo, RecordJobInfo, RecordOptions
from control_stubs.sensing_pb2 import SensorData


@dataclass(slots=True)
class CaptureSnapshot:
    robot_state: JointState
    joint_command_state: JointState
    end_effector_states: dict[str, EndEffectorState]
    sensor_data: SensorData


class DataRecorder(ABC):
    @abstractmethod
    def episode_start(self, options: RecordOptions) -> RecordJobInfo:
        raise NotImplementedError

    @abstractmethod
    def episode_end(self) -> Status:
        raise NotImplementedError

    @abstractmethod
    def episode_cancel(self) -> Status:
        raise NotImplementedError

    @abstractmethod
    def episode_replay(self, info: RecordInfo) -> Status:
        raise NotImplementedError
