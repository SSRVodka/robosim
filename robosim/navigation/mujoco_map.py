"""Build a 2D occupancy grid from a MuJoCo scene."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robosim.navigation.grid import OccupancyGrid


@dataclass(frozen=True, slots=True)
class GridBuildOptions:
    resolution: float = 0.10
    robot_radius: float = 0.18
    safety_margin: float = 0.08
    bounds_padding: float = 0.20
    relevant_z_min: float = -0.05
    robot_top_z: float = 0.16
    bounds: tuple[float, float, float, float] | None = None
    ignored_geom_names: tuple[str, ...] = ()
    ignored_geom_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GridBuildStats:
    bounds: tuple[float, float, float, float]
    obstacle_geoms: int
    skipped_robot_geoms: int
    skipped_visual_geoms: int
    skipped_ignored_geoms: int
    skipped_height_geoms: int
    skipped_unsupported_geoms: int
    raw_occupied_cells: int
    inflated_occupied_cells: int


@dataclass(frozen=True, slots=True)
class BuiltGrid:
    grid: OccupancyGrid
    raw_grid: OccupancyGrid
    stats: GridBuildStats


def build_occupancy_grid(
    scene_path: str | Path,
    options: GridBuildOptions | None = None,
) -> BuiltGrid:
    opts = options or GridBuildOptions()
    model = mujoco.MjModel.from_xml_path(str(Path(scene_path)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bounds = opts.bounds or _infer_floor_bounds(model, data, opts.bounds_padding)
    x_min, x_max, y_min, y_max = bounds
    raw_grid = OccupancyGrid.empty(x_min, x_max, y_min, y_max, opts.resolution)
    robot_body_ids = _find_robot_body_ids(model)

    obstacle_geoms = 0
    skipped_robot_geoms = 0
    skipped_visual_geoms = 0
    skipped_ignored_geoms = 0
    skipped_height_geoms = 0
    skipped_unsupported_geoms = 0

    for geom_id in range(model.ngeom):
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        geom_name = model.geom(geom_id).name
        if _ignored_geom(geom_name, opts):
            skipped_ignored_geoms += 1
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if body_id in robot_body_ids:
            skipped_robot_geoms += 1
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            skipped_visual_geoms += 1
            continue

        aabb = _geom_world_aabb(model, data, geom_id)
        if aabb is None:
            skipped_unsupported_geoms += 1
            continue

        gx_min, gx_max, gy_min, gy_max, gz_min, gz_max = aabb
        if gz_max < opts.relevant_z_min or gz_min > opts.robot_top_z:
            skipped_height_geoms += 1
            continue
        if gx_max < x_min or gx_min > x_max or gy_max < y_min or gy_min > y_max:
            continue

        raw_grid.mark_rect(gx_min, gx_max, gy_min, gy_max)
        obstacle_geoms += 1

    inflated_grid = raw_grid.copy()
    inflated_grid.inflate(opts.robot_radius + opts.safety_margin)
    return BuiltGrid(
        grid=inflated_grid,
        raw_grid=raw_grid,
        stats=GridBuildStats(
            bounds=bounds,
            obstacle_geoms=obstacle_geoms,
            skipped_robot_geoms=skipped_robot_geoms,
            skipped_visual_geoms=skipped_visual_geoms,
            skipped_ignored_geoms=skipped_ignored_geoms,
            skipped_height_geoms=skipped_height_geoms,
            skipped_unsupported_geoms=skipped_unsupported_geoms,
            raw_occupied_cells=raw_grid.occupied_count(),
            inflated_occupied_cells=inflated_grid.occupied_count(),
        ),
    )


def _ignored_geom(geom_name: str, options: GridBuildOptions) -> bool:
    if geom_name in options.ignored_geom_names:
        return True
    return any(geom_name.startswith(prefix) for prefix in options.ignored_geom_prefixes)


def _infer_floor_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    padding: float,
) -> tuple[float, float, float, float]:
    bounds: list[tuple[float, float, float, float]] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        size = model.geom_size[geom_id]
        if not math.isfinite(float(size[0])) or not math.isfinite(float(size[1])):
            continue
        pos = data.geom_xpos[geom_id]
        bounds.append(
            (
                float(pos[0] - size[0]),
                float(pos[0] + size[0]),
                float(pos[1] - size[1]),
                float(pos[1] + size[1]),
            )
        )
    if not bounds:
        return _infer_geometry_bounds(model, data, padding)
    return (
        min(item[0] for item in bounds) - padding,
        max(item[1] for item in bounds) + padding,
        min(item[2] for item in bounds) - padding,
        max(item[3] for item in bounds) + padding,
    )


def _infer_geometry_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    padding: float,
) -> tuple[float, float, float, float]:
    aabbs = [
        aabb
        for geom_id in range(model.ngeom)
        if (aabb := _geom_world_aabb(model, data, geom_id)) is not None
    ]
    if not aabbs:
        return (-5.0, 5.0, -5.0, 5.0)
    return (
        min(item[0] for item in aabbs) - padding,
        max(item[1] for item in aabbs) + padding,
        min(item[2] for item in aabbs) - padding,
        max(item[3] for item in aabbs) + padding,
    )


def _find_robot_body_ids(model: mujoco.MjModel) -> set[int]:
    candidates: set[int] = set()
    for body_id in range(1, model.nbody):
        body_name = model.body(body_id).name
        if "robot_vacuum" in body_name or body_name.startswith("rv_"):
            candidates.add(body_id)
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        if joint_name.startswith("rv_"):
            candidates.add(int(model.jnt_bodyid[joint_id]))

    body_children: dict[int, list[int]] = {body_id: [] for body_id in range(model.nbody)}
    for body_id in range(1, model.nbody):
        body_children[int(model.body_parentid[body_id])].append(body_id)

    robot_ids: set[int] = set()

    def collect(body_id: int) -> None:
        robot_ids.add(body_id)
        for child_id in body_children[body_id]:
            collect(child_id)

    for body_id in candidates:
        collect(body_id)
    return robot_ids


def _geom_world_aabb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[float, float, float, float, float, float] | None:
    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        corners = np.array(
            [
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        return _points_world_aabb(data, geom_id, corners)

    if geom_type in (
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    ):
        radius = float(model.geom_rbound[geom_id])
        pos = data.geom_xpos[geom_id]
        return (
            float(pos[0] - radius),
            float(pos[0] + radius),
            float(pos[1] - radius),
            float(pos[1] + radius),
            float(pos[2] - radius),
            float(pos[2] + radius),
        )

    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        start = int(model.mesh_vertadr[mesh_id])
        end = start + int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:end], dtype=np.float64)
        if vertices.size == 0:
            return None
        return _points_world_aabb(data, geom_id, vertices)

    return None


def _points_world_aabb(
    data: mujoco.MjData,
    geom_id: int,
    local_points: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    world = local_points @ mat.T + pos
    mins = world.min(axis=0)
    maxs = world.max(axis=0)
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )
