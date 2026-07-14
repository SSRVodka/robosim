from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.robot_utils import precise_sleep

from control_stubs import common_pb2
from control_stubs.policy_pb2 import PolicyLoadRequest, PolicyStartRequest, PolicyStatus
from control_stubs.robot_core_pb2 import JointCommand
from robosim.core.activity import ActivityCoordinator
from robosim.core.backend import SimulatorBackend
from robosim.core.impl.lerobot_io import LerobotObservationAdapter, PolicyRuntimeSpec


@dataclass(slots=True)
class LoadedPolicy:
    path: str
    dataset_repo_name: str
    dataset_root: Path
    config: PreTrainedConfig
    policy: PreTrainedPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    adapter: LerobotObservationAdapter
    runtime_spec: PolicyRuntimeSpec
    task_text: str
    control_fps: int


class LerobotPolicyRunner:
    """Run a loaded LeRobot policy against a backend control loop."""

    def __init__(
        self,
        repo_root: Path,
        backend: SimulatorBackend,
        activity_coordinator: ActivityCoordinator | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._backend = backend
        self._datasets_root = repo_root / "data" / "lerobot"
        self._activity = activity_coordinator
        self._lock = threading.RLock()
        self._loaded: LoadedPolicy | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error = ""

    def load_policy(self, request: PolicyLoadRequest) -> common_pb2.Status:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("policy inference is already running")

            dataset_repo_name = request.dataset_repo_name.strip()
            if not dataset_repo_name:
                raise ValueError("dataset_repo_name must not be empty")
            dataset_root = self._datasets_root / dataset_repo_name
            if not dataset_root.exists():
                raise ValueError(f"dataset '{dataset_repo_name}' does not exist")

            policy_path = request.policy_path.strip()
            if not policy_path:
                raise ValueError("policy_path must not be empty")

            config = PreTrainedConfig.from_pretrained(policy_path)
            if request.device:
                config.device = request.device

            policy = get_policy_class(config.type).from_pretrained(
                policy_path,
                config=config,
            )
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=policy_path,
                preprocessor_overrides={
                    "device_processor": {"device": policy.config.device},
                },
                postprocessor_overrides={
                    "device_processor": {"device": policy.config.device},
                },
            )

            dataset_meta = LeRobotDatasetMetadata(
                repo_id=dataset_repo_name,
                root=dataset_root,
            )
            control_fps = int(request.control_fps) or int(dataset_meta.fps)
            if control_fps <= 0:
                raise ValueError("control_fps must be positive")
            adapter = LerobotObservationAdapter(
                backend=self._backend,
                dataset_meta=dataset_meta,
                policy_config=policy.config,
                group_name=request.jmg_name or None,
            )

            self._loaded = LoadedPolicy(
                path=policy_path,
                dataset_repo_name=dataset_repo_name,
                dataset_root=dataset_root,
                config=policy.config,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                adapter=adapter,
                runtime_spec=adapter.runtime_spec,
                task_text=request.task_text,
                control_fps=control_fps,
            )
            self._last_error = ""

        return common_pb2.Status(
            code=common_pb2.STATUS_SUCCESS,
            message="policy loaded",
        )

    def start_policy(self, request: PolicyStartRequest) -> common_pb2.Status:
        with self._lock:
            loaded = self._require_loaded()
            if self._thread is not None:
                raise RuntimeError("policy inference is already running")
            if request.task_text:
                loaded.task_text = request.task_text
            if request.control_fps:
                loaded.control_fps = int(request.control_fps)
            if loaded.control_fps <= 0:
                raise ValueError("control_fps must be positive")
            if self._activity is not None:
                self._activity.acquire("policy")

            self._stop_event.clear()
            loaded.policy.reset()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="lerobot_policy_runner",
                daemon=True,
            )
            self._thread.start()

        return common_pb2.Status(
            code=common_pb2.STATUS_SUCCESS,
            message="policy started",
        )

    def stop_policy(self) -> common_pb2.Status:
        thread: threading.Thread | None
        with self._lock:
            thread = self._thread
            if thread is None:
                return common_pb2.Status(
                    code=common_pb2.STATUS_SUCCESS,
                    message="policy already stopped",
                )
            self._stop_event.set()

        thread.join()
        return common_pb2.Status(
            code=common_pb2.STATUS_SUCCESS,
            message="policy stopped",
        )

    def get_status(self) -> PolicyStatus:
        with self._lock:
            loaded = self._loaded
            return PolicyStatus(
                status=common_pb2.Status(
                    code=(
                        common_pb2.STATUS_RUNNING
                        if self._thread is not None
                        else common_pb2.STATUS_SUCCESS
                    ),
                    message=self._last_error,
                ),
                loaded=loaded is not None,
                running=self._thread is not None,
                policy_type=loaded.config.type if loaded is not None else "",
                policy_path=loaded.path if loaded is not None else "",
                dataset_repo_name=loaded.dataset_repo_name if loaded is not None else "",
                device=loaded.config.device if loaded is not None else "",
                jmg_name=loaded.runtime_spec.group_name if loaded is not None else "",
                control_fps=loaded.control_fps if loaded is not None else 0,
                task_text=loaded.task_text if loaded is not None else "",
                active_mode=self._activity.active_mode if self._activity is not None else "",
            )

    def notify_world_reset(self) -> None:
        with self._lock:
            loaded = self._loaded
            if loaded is not None:
                loaded.policy.reset()

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            self.stop_policy()

    def _run_loop(self) -> None:
        try:
            deadline = time.monotonic()
            while not self._stop_event.is_set():
                self._step_once()
                loaded = self._require_loaded()
                deadline += 1.0 / loaded.control_fps
                precise_sleep(deadline - time.monotonic())
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                self._thread = None
                self._stop_event.clear()
            if self._activity is not None:
                self._activity.release("policy")

    def _step_once(self) -> None:
        with self._lock:
            loaded = self._require_loaded()
            observation = loaded.adapter.capture_observation()
            prepared = prepare_observation_for_inference(
                observation=observation,
                device=torch.device(str(loaded.config.device)),
                task=loaded.task_text,
                robot_type=self._backend.robot_name,
            )
            processed = loaded.preprocessor(prepared)
            action = loaded.policy.select_action(processed)
            action = loaded.postprocessor(action)

        action_tensor = action.squeeze(0).to("cpu")
        action_values = [float(value) for value in action_tensor.tolist()]
        expected_dim = len(loaded.runtime_spec.action_joint_names)
        if len(action_values) != expected_dim:
            raise ValueError(
                f"policy action dimension mismatch: {len(action_values)} != {expected_dim}"
            )
        self._backend.set_joint_target(
            loaded.runtime_spec.action_joint_names,
            action_values,
            JointCommand.ControlMode.POSITION,
            loaded.runtime_spec.group_name,
        )

    def _require_loaded(self) -> LoadedPolicy:
        if self._loaded is None:
            raise RuntimeError("policy is not loaded")
        return self._loaded
