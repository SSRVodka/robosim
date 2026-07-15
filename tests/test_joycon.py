from __future__ import annotations

import pytest
from evdev import AbsInfo, ecodes

from control_stubs.tools import joycon
from control_stubs.tools.teleop import TeleopEvent


def _mapper() -> joycon.JoyConMapper:
    return joycon.JoyConMapper(
        x_axis=joycon.AxisRange(-32767, 32767),
        y_axis=joycon.AxisRange(-32767, 32767),
        deadzone=0.1,
        linear_speed=0.02,
        angular_speed=0.3,
        joint_speed=0.2,
    )


def test_joycon_mapper_normalizes_stick_and_applies_deadzone() -> None:
    mapper = _mapper()
    mapper.feed(ecodes.EV_ABS, ecodes.ABS_RX, 1000)
    mapper.feed(ecodes.EV_ABS, ecodes.ABS_RY, -32767)

    motion = mapper.motion()

    assert motion.linear == pytest.approx((0.02, 0.0, 0.0))
    assert motion.angular == (0.0, 0.0, 0.0)


def test_joycon_mapper_uses_modifier_for_rotation_and_buttons_for_joint() -> None:
    mapper = _mapper()
    mapper.feed(ecodes.EV_KEY, ecodes.BTN_TL, 1)
    mapper.feed(ecodes.EV_ABS, ecodes.ABS_RX, 32767)
    mapper.feed(ecodes.EV_KEY, ecodes.BTN_TR, 1)
    mapper.feed(ecodes.EV_KEY, ecodes.BTN_X, 1)

    motion = mapper.motion()

    assert motion.linear == (0.0, 0.0, 0.0)
    assert motion.angular == pytest.approx((0.0, -0.3, 0.3))
    assert motion.joint_velocity == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (ecodes.BTN_MODE, TeleopEvent.NEXT_TWIST_TARGET),
        (ecodes.BTN_THUMBR, TeleopEvent.NEXT_JOINT_TARGET),
        (ecodes.BTN_B, TeleopEvent.SAVE_EPISODE),
        (ecodes.BTN_A, TeleopEvent.RETRY_EPISODE),
        (ecodes.BTN_START, TeleopEvent.STOP),
    ],
)
def test_joycon_mapper_emits_episode_and_target_events_on_button_press(
    code: int, expected: TeleopEvent
) -> None:
    mapper = _mapper()

    assert mapper.feed(ecodes.EV_KEY, code, 1) == expected
    assert mapper.feed(ecodes.EV_KEY, code, 2) is None
    assert mapper.feed(ecodes.EV_KEY, code, 0) is None


class FakeDevice:
    name = "synthetic right Joy-Con"
    fd = 42

    def capabilities(self, *, absinfo: bool) -> dict[int, list]:
        assert absinfo
        axis = AbsInfo(value=0, min=-100, max=100, fuzz=0, flat=0, resolution=0)
        return {
            ecodes.EV_ABS: [(ecodes.ABS_RX, axis), (ecodes.ABS_RY, axis)],
            ecodes.EV_KEY: [
                ecodes.BTN_A,
                ecodes.BTN_B,
                ecodes.BTN_X,
                ecodes.BTN_Y,
                ecodes.BTN_TL,
                ecodes.BTN_TR,
                ecodes.BTN_TR2,
                ecodes.BTN_START,
                ecodes.BTN_MODE,
                ecodes.BTN_THUMBR,
            ],
        }

    def read(self) -> list[joycon.DeviceEvent]:
        return [
            joycon.DeviceEvent(ecodes.EV_ABS, ecodes.ABS_RY, -100),
            joycon.DeviceEvent(ecodes.EV_KEY, ecodes.BTN_B, 1),
        ]

    def close(self) -> None:
        return None


def test_joycon_input_reads_device_without_hardware() -> None:
    adapter = joycon.JoyConInput(
        FakeDevice(),
        deadzone=0.1,
        linear_speed=0.02,
        angular_speed=0.3,
        joint_speed=0.2,
    )

    snapshot = adapter.read()

    assert snapshot.motion.linear == pytest.approx((0.02, 0.0, 0.0))
    assert snapshot.events == (TeleopEvent.SAVE_EPISODE,)
