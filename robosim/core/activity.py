from __future__ import annotations

import threading


class ActivityCoordinator:
    """Serialize mutually-exclusive active control modes on one backend instance."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_mode: str | None = None

    def acquire(self, mode: str) -> None:
        with self._lock:
            if self._active_mode is not None and self._active_mode != mode:
                raise RuntimeError(f"'{self._active_mode}' is already in progress")
            self._active_mode = mode

    def release(self, mode: str) -> None:
        with self._lock:
            if self._active_mode == mode:
                self._active_mode = None

    @property
    def active_mode(self) -> str | None:
        with self._lock:
            return self._active_mode
