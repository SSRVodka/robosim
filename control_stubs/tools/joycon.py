"""Linux evdev adapter for the right Joy-Con teleoperation profile."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from evdev import InputDevice, ecodes

from control_stubs.tools.teleop import InputSnapshot, TeleopEvent, TeleopMotion


@dataclass(frozen=True, slots=True)
class DeviceEvent:
    type: int
    code: int
    value: int


class EventDevice(Protocol):
    name: str
    fd: int

    def capabilities(self, *, absinfo: bool) -> dict[int, list[Any]]: ...

    def read(self) -> Iterable[DeviceEvent]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AxisRange:
    minimum: int
    maximum: int

    def normalize(self, value: int, deadzone: float) -> float:
        center = (self.minimum + self.maximum) / 2.0
        radius = (self.maximum - self.minimum) / 2.0
        normalized = max(-1.0, min(1.0, (value - center) / radius))
        return 0.0 if abs(normalized) <= deadzone else normalized


class JoyConMapper:
    """Translate right Joy-Con event codes into device-neutral motion/events."""

    _EVENTS = {
        ecodes.BTN_MODE: TeleopEvent.NEXT_TWIST_TARGET,
        ecodes.BTN_THUMBR: TeleopEvent.NEXT_JOINT_TARGET,
        ecodes.BTN_B: TeleopEvent.SAVE_EPISODE,
        ecodes.BTN_A: TeleopEvent.RETRY_EPISODE,
        ecodes.BTN_START: TeleopEvent.STOP,
    }

    def __init__(
        self,
        *,
        x_axis: AxisRange,
        y_axis: AxisRange,
        deadzone: float,
        linear_speed: float,
        angular_speed: float,
        joint_speed: float,
    ) -> None:
        if not 0.0 <= deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        self._x_axis = x_axis
        self._y_axis = y_axis
        self._deadzone = deadzone
        self._linear_speed = linear_speed
        self._angular_speed = angular_speed
        self._joint_speed = joint_speed
        self._x = 0.0
        self._y = 0.0
        self._pressed: set[int] = set()

    def feed(self, event_type: int, code: int, value: int) -> TeleopEvent | None:
        if event_type == ecodes.EV_ABS:
            if code == ecodes.ABS_RX:
                self._x = self._x_axis.normalize(value, self._deadzone)
            elif code == ecodes.ABS_RY:
                self._y = self._y_axis.normalize(value, self._deadzone)
            return None
        if event_type != ecodes.EV_KEY:
            return None
        if value == 0:
            self._pressed.discard(code)
            return None
        self._pressed.add(code)
        return self._EVENTS.get(code) if value == 1 else None

    def motion(self) -> TeleopMotion:
        vertical = -self._y
        horizontal = -self._x
        shoulder = float(ecodes.BTN_TR in self._pressed) - float(
            ecodes.BTN_TR2 in self._pressed
        )
        joint = float(ecodes.BTN_X in self._pressed) - float(
            ecodes.BTN_Y in self._pressed
        )
        if ecodes.BTN_TL in self._pressed:
            return TeleopMotion(
                angular=(
                    vertical * self._angular_speed,
                    horizontal * self._angular_speed,
                    shoulder * self._angular_speed,
                ),
                joint_velocity=joint * self._joint_speed,
            )
        return TeleopMotion(
            linear=(
                vertical * self._linear_speed,
                horizontal * self._linear_speed,
                shoulder * self._linear_speed,
            ),
            joint_velocity=joint * self._joint_speed,
        )


class JoyConInput:
    """Read a validated right Joy-Con evdev device without owning session logic."""

    _REQUIRED_BUTTONS = {
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
    }

    def __init__(
        self,
        device: EventDevice,
        *,
        deadzone: float,
        linear_speed: float,
        angular_speed: float,
        joint_speed: float,
    ) -> None:
        capabilities = device.capabilities(absinfo=True)
        axes = dict(capabilities.get(ecodes.EV_ABS, []))
        missing_axes = {ecodes.ABS_RX, ecodes.ABS_RY} - axes.keys()
        missing_buttons = self._REQUIRED_BUTTONS - set(
            capabilities.get(ecodes.EV_KEY, [])
        )
        if missing_axes or missing_buttons:
            raise ValueError(f"device '{device.name}' does not match joycon-right profile")
        self._device = device
        self._mapper = JoyConMapper(
            x_axis=AxisRange(axes[ecodes.ABS_RX].min, axes[ecodes.ABS_RX].max),
            y_axis=AxisRange(axes[ecodes.ABS_RY].min, axes[ecodes.ABS_RY].max),
            deadzone=deadzone,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            joint_speed=joint_speed,
        )

    @classmethod
    def open(
        cls,
        path: str,
        *,
        deadzone: float,
        linear_speed: float,
        angular_speed: float,
        joint_speed: float,
    ) -> JoyConInput:
        return cls(
            InputDevice(path),  # type: ignore[arg-type]
            deadzone=deadzone,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            joint_speed=joint_speed,
        )

    @property
    def fd(self) -> int:
        return self._device.fd

    def read(self) -> InputSnapshot:
        events = tuple(
            event
            for raw_event in self._device.read()
            if (event := self._mapper.feed(raw_event.type, raw_event.code, raw_event.value))
            is not None
        )
        return InputSnapshot(self._mapper.motion(), events)

    def close(self) -> None:
        self._device.close()
