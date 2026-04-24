"""Shared tool definitions and serialization utilities for gRPC interfaces."""

from __future__ import annotations

import json
from typing import Any, Callable, Coroutine

from google.protobuf import json_format

from control_stubs import common_pb2, robot_core_pb2

from .client import RobosimClient


def _parse_pose(pose: dict[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Parse pose dict to (position, orientation) tuples."""
    pos = pose.get("position", {"x": 0, "y": 0, "z": 0})
    ori = pose.get("orientation", {"x": 0, "y": 0, "z": 0, "w": 1})
    return (pos["x"], pos["y"], pos["z"]), (ori["x"], ori["y"], ori["z"], ori["w"])


def _serialize_status(status: common_pb2.Status) -> dict[str, Any]:
    """Serialize Status proto to dict with enum as string name."""
    return {
        "code": common_pb2.StatusCode.Name(status.code),
        "message": status.message,
    }


def _serialize_sensor_meta_list(meta: Any) -> dict[str, Any]:
    """Wrap SensorMetaList entries in 'entries' key."""
    data = json_format.MessageToDict(meta)
    return {"entries": data.get("entries", [])}


def _serialize_step_response(resp: Any) -> dict[str, Any]:
    return {
        "header": {"seq": 0, "timestamp": 0.0, "frame_id": ""},
        "reward": resp.reward,
        "done": resp.done,
    }


def _serialize_navigate_feedback(fb: Any) -> dict[str, Any] | None:
    if fb is None:
        return None
    return {
        "task_id": fb.task_id,
        "status": _serialize_status(fb.status),
        "eta": fb.eta,
        "feedback_text": fb.feedback_text,
    }





# ============================================================================
# Tool Definitions (shared between local and MCP)
# ============================================================================

POS_ORI_SCHEMA = {
    "position": {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
    },
    "orientation": {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}, "w": {"type": "number"}},
    },
}

POSE_SCHEMA = {
    "type": "object",
    "required": ["position", "orientation"],
    "properties": POS_ORI_SCHEMA,
}

TOOL_DEFINITIONS = [
    {
        "name": "reset_world",
        "description": "Reset the simulation world to its initial state. Essential for starting new RL episodes.",
        "parameters": {
            "type": "object",
            "properties": {
                "seed": {"type": "integer", "description": "Random seed for reproducibility"},
                "randomization_params": {
                    "type": "object",
                    "description": "Optional randomization parameters (e.g., lighting, friction)",
                    "additionalProperties": {"type": "number"},
                },
            },
        },
    },
    {
        "name": "step_physics",
        "description": "Manually step the physics engine. Used for synchronous training control.",
        "parameters": {
            "type": "object",
            "properties": {
                "block": {"type": "boolean", "description": "Wait for physics step to complete"},
            },
        },
    },
    {
        "name": "set_object_pose",
        "description": "Set an object's pose in the simulation (cheat/editor functionality).",
        "parameters": {
            "type": "object",
            "required": ["object_name", "pose"],
            "properties": {
                "object_name": {"type": "string", "description": "Name of the object to manipulate"},
                "pose": POSE_SCHEMA,
            },
        },
    },
    {
        "name": "list_sensors",
        "description": "List all available sensors in the simulation with their types.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_sensors",
        "description": "Get a snapshot of specific sensor readings.",
        "parameters": {
            "type": "object",
            "required": ["sensor_names"],
            "properties": {
                "sensor_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of sensor names to read",
                },
            },
        },
    },
    {
        "name": "get_robot_state",
        "description": "Get the current state of all robot joints (position, velocity, effort).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_robot_spec",
        "description": "Get robot specifications including all joints, move groups, named states, and end effectors.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "set_joint_target",
        "description": "Set joint position/velocity targets for non-realtime trajectory execution.",
        "parameters": {
            "type": "object",
            "required": ["names", "data", "mode"],
            "properties": {
                "names": {"type": "array", "items": {"type": "string"}, "description": "Joint names to control"},
                "data": {"type": "array", "items": {"type": "number"}, "description": "Target values"},
                "mode": {"type": "string", "enum": ["POSITION", "VELOCITY", "TORQUE"], "description": "Control mode"},
                "jmg_name": {"type": "string", "description": "Optional move group name"},
            },
        },
    },
    {
        "name": "get_end_effector_state",
        "description": "Get the end-effector pose (forward kinematics result) for a specific move group.",
        "parameters": {
            "type": "object",
            "required": ["jmg_name"],
            "properties": {"jmg_name": {"type": "string", "description": "Move group name"}},
        },
    },
    {
        "name": "emergency_stop",
        "description": "Trigger emergency stop to halt all robot motion immediately.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_robot_pose_in_map",
        "description": "Get the current robot pose in the map coordinate frame.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "navigate_to",
        "description": "Navigate the robot to a target pose using SLAM-based navigation. Streams progress feedback.",
        "parameters": {
            "type": "object",
            "required": ["target_pose"],
            "properties": {
                "target_pose": POSE_SCHEMA,
                "target_frame": {"type": "string", "description": "Coordinate frame (default: 'map')"},
                "max_velocity": {"type": "number", "description": "Override max velocity"},
            },
        },
    },
]


def _to_mcp_tool(defn: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI-style tool definition to MCP format."""
    return {
        "name": defn["name"],
        "description": defn["description"],
        "inputSchema": defn["parameters"],
    }


MCP_TOOLS = [_to_mcp_tool(d) for d in TOOL_DEFINITIONS]


def create_tool_implementations(client: RobosimClient) -> dict[str, Callable[..., Coroutine[Any, Any, str]]]:
    """Create async tool implementations bound to a gRPC client."""

    async def reset_world(a: dict[str, Any]) -> str:
        return json.dumps(_serialize_status(
            client.simulation.reset_world(
                seed=a.get("seed", 0),
                randomization_params=a.get("randomization_params"),
            )
        ))

    async def step_physics(a: dict[str, Any]) -> str:
        resp = client.simulation.step_physics(block=a.get("block", True))
        return json.dumps(_serialize_step_response(resp))

    async def set_object_pose(a: dict[str, Any]) -> str:
        pos, ori = _parse_pose(a["pose"])
        return json.dumps(_serialize_status(
            client.simulation.set_object_pose(a["object_name"], pos, ori)
        ))

    async def list_sensors(_: dict[str, Any]) -> str:
        resp = client.sensing.list_sensors()
        return json.dumps(_serialize_sensor_meta_list(resp))

    async def get_sensors(a: dict[str, Any]) -> str:
        resp = client.sensing.get_sensors(a["sensor_names"])
        return json.dumps(json_format.MessageToDict(resp))

    async def get_robot_state(_: dict[str, Any]) -> str:
        resp = client.robot_core.get_robot_state()
        return json.dumps(json_format.MessageToDict(resp))

    async def get_robot_spec(_: dict[str, Any]) -> str:
        resp = client.robot_core.get_robot_spec()
        return json.dumps(json_format.MessageToDict(resp))

    async def set_joint_target(a: dict[str, Any]) -> str:
        return json.dumps(_serialize_status(
            client.robot_core.set_joint_target(
                names=a["names"],
                data=a["data"],
                mode=getattr(robot_core_pb2.JointCommand.ControlMode, a["mode"]),
                jmg_name=a.get("jmg_name"),
            )
        ))

    async def get_end_effector_state(a: dict[str, Any]) -> str:
        resp = client.robot_core.get_end_effector_state(a["jmg_name"])
        return json.dumps(json_format.MessageToDict(resp.pose_stamped))

    async def emergency_stop(_: dict[str, Any]) -> str:
        return json.dumps(_serialize_status(client.robot_core.emergency_stop()))

    async def get_robot_pose_in_map(_: dict[str, Any]) -> str:
        resp = client.mobility.get_robot_pose_in_map()
        return json.dumps(json_format.MessageToDict(resp))

    async def navigate_to(a: dict[str, Any]) -> str:
        tp = a["target_pose"]
        pos, ori = _parse_pose(tp)
        feedback_list = list(client.mobility.navigate_to(
            target_pose=(pos[0], pos[1], pos[2], ori[0], ori[1], ori[2]),
            target_frame=a.get("target_frame", "map"),
            max_velocity=a.get("max_velocity", 0.0),
        ))
        last = feedback_list[-1] if feedback_list else None
        return json.dumps(_serialize_navigate_feedback(last))

    return {
        "reset_world": reset_world,
        "step_physics": step_physics,
        "set_object_pose": set_object_pose,
        "list_sensors": list_sensors,
        "get_sensors": get_sensors,
        "get_robot_state": get_robot_state,
        "get_robot_spec": get_robot_spec,
        "set_joint_target": set_joint_target,
        "get_end_effector_state": get_end_effector_state,
        "emergency_stop": emergency_stop,
        "get_robot_pose_in_map": get_robot_pose_in_map,
        "navigate_to": navigate_to,
    }
