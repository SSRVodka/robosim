from __future__ import annotations

import pytest

from robosim.backends.gazebo.backend import GazeboBackend
from robosim.core.capabilities import Capability


class DummyLogger:
    def error(self, message: str) -> None:
        del message

    def warn(self, message: str) -> None:
        del message


class ResetOnlyGazeboBackend(GazeboBackend):
    def __init__(self) -> None:
        self._capabilities = Capability.SIMULATION_CONTROL

    def get_logger(self) -> DummyLogger:
        return DummyLogger()


def test_gazebo_reset_reports_unimplemented() -> None:
    backend = ResetOnlyGazeboBackend()

    with pytest.raises(NotImplementedError, match="Reset world"):
        backend.reset_world(seed=0, randomization_params={})
