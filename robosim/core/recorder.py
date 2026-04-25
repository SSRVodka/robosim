from abc import ABC, abstractmethod

from control_stubs.common_pb2 import Status
from control_stubs.robot_data_pb2 import RecordJobInfo, RecordOptions


class DataRecorder(ABC):
    @abstractmethod
    def record_episode_start(self, options: RecordOptions) -> RecordJobInfo:
        raise NotImplementedError

    @abstractmethod
    def record_episode_end(self) -> Status:
        raise NotImplementedError
