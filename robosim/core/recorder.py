from abc import ABC, abstractmethod

from control_stubs.common_pb2 import Status
from control_stubs.robot_data_pb2 import RecordInfo, RecordJobInfo, RecordOptions


class DataRecorder(ABC):
    @abstractmethod
    def episode_start(self, options: RecordOptions) -> RecordJobInfo:
        raise NotImplementedError

    @abstractmethod
    def episode_end(self) -> Status:
        raise NotImplementedError

    @abstractmethod
    def episode_replay(self, info: RecordInfo) -> Status:
        raise NotImplementedError