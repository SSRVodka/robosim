"""PyBullet cup scene mirroring the MuJoCo Franka cup scene.

Geometry parity with drivers_sim/mujoco/assets/robots/franka_panda/scene.xml:
table box at (0.6,-0.2,0) half extents (0.25,0.25,0.1); free cup cylinder
r=0.028 half-h=0.05 starting at (0.6,-0.2,0.151); container compound (base +
4 walls) at (0.4,0.3,0.15). Known lossy conversions (recorded in
scene_meta.json): the cup handle box is omitted (single cylinder, combined
mass) and contact parameters are tuned per-backend instead of copied from
MJCF solref/solimp.
"""

from __future__ import annotations

import pybullet as p
import pybullet_data


def load_scene(client_id: int) -> dict[str, object]:
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    table_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.1], physicsClientId=client_id
    )
    table_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=table_shape,
        basePosition=[0.6, -0.2, 0.0],
        physicsClientId=client_id,
    )
    p.changeDynamics(table_id, -1, lateralFriction=1.2, physicsClientId=client_id)

    cup_shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=0.028, height=0.1, physicsClientId=client_id
    )
    cup_id = p.createMultiBody(
        baseMass=0.14,  # MJCF cup_body 0.12 + cup_handle 0.02
        baseCollisionShapeIndex=cup_shape,
        basePosition=[0.6, -0.2, 0.151],
        physicsClientId=client_id,
    )
    p.changeDynamics(
        cup_id,
        -1,
        lateralFriction=2.2,
        spinningFriction=0.2,
        physicsClientId=client_id,
    )

    wall_half_extents = [
        [0.2, 0.2, 0.01],
        [0.2, 0.025, 0.1],
        [0.2, 0.025, 0.1],
        [0.025, 0.2, 0.1],
        [0.025, 0.2, 0.1],
    ]
    wall_positions = [
        [0.0, 0.0, -0.09],
        [0.0, 0.175, -0.09],
        [0.0, -0.175, -0.09],
        [-0.175, 0.0, -0.09],
        [0.175, 0.0, -0.09],
    ]
    container_shape = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX] * 5,
        halfExtents=wall_half_extents,
        collisionFramePositions=wall_positions,
        physicsClientId=client_id,
    )
    container_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=container_shape,
        basePosition=[0.4, 0.3, 0.15],
        physicsClientId=client_id,
    )

    # Per-backend grasp tuning: bump finger friction for a stable pinch.
    for joint_index in range(p.getNumJoints(robot_id, physicsClientId=client_id)):
        link_name = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)[
            12
        ].decode()
        if "finger" in link_name:
            p.changeDynamics(
                robot_id, joint_index, lateralFriction=2.0, physicsClientId=client_id
            )

    return {
        "bodies": {
            "panda": robot_id,
            "table": table_id,
            "cup": cup_id,
            "container": container_id,
        }
    }
