"""Test-only authoring of legacy semantic fixtures as canonical OpenUSD stages."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


def author_openusd_csd(payload: Mapping[str, Any], root: Path) -> Path:
    """Author legacy test data into one standards-based composed-stage fixture."""
    csd_id = str(payload["csd_id"])
    scenario = _mapping(payload["scenario"])
    path = root / "generated_openusd" / csd_id / "csd.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.FindOrOpen(str(path)) if path.exists() else Sdf.Layer.CreateNew(str(path))
    assert layer is not None
    layer.Clear()
    stage = Usd.Stage.Open(layer)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageMetersPerUnit(stage, _meters_per_unit(str(scenario.get("units", "m"))))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    _schemas(world, "RobosimCsdRootAPI", "RobosimRandomizationAPI")
    _attr(world, "robosim:csd:id", Sdf.ValueTypeNames.String, csd_id)
    _attr(
        world,
        "robosim:csd:schemaVersion",
        Sdf.ValueTypeNames.String,
        str(payload.get("schema_version", "0.1")),
    )
    task_data = _mapping(scenario.get("task", {}))
    _attr(
        world,
        "robosim:csd:taskInstanceId",
        Sdf.ValueTypeNames.String,
        str(task_data.get("task_instance_id", f"task_{csd_id}")),
    )
    environment_data = _mapping(scenario.get("environment", {}))
    environment_id = str(environment_data.get("environment_id", "environment"))
    _attr(
        world,
        "robosim:csd:worldTemplateId",
        Sdf.ValueTypeNames.String,
        environment_id,
    )
    randomization = _mapping(scenario.get("randomization", {}))
    _attr(
        world,
        "robosim:randomization:seed",
        Sdf.ValueTypeNames.Int64,
        int(randomization.get("seed", 0)),
    )

    _author_environment(stage, environment_data)
    _author_robot(stage, scenario.get("robot"))
    object_paths = _author_objects(stage, scenario.get("objects", ()))
    _author_task(stage, task_data)
    _author_relationships(stage, scenario.get("relationships", ()), object_paths)

    variants = world.GetVariantSets().AddVariantSet("physicsBackend")
    for backend in ("mujoco", "pybullet", "gazebo"):
        variants.AddVariant(backend)
    variants.SetVariantSelection("mujoco")
    layer.Save()
    return path


def _author_environment(stage: Usd.Stage, data: Mapping[str, Any]) -> None:
    environment = UsdGeom.Xform.Define(stage, "/World/Environment").GetPrim()
    _entity(
        environment,
        entity_id="environment",
        asset_id=str(data.get("environment_id", "environment")),
        role="environment",
        static=True,
    )
    gravity = _vector(data.get("gravity", (0.0, 0.0, -9.81)))
    magnitude = gravity.GetLength()
    direction = gravity / magnitude if magnitude else Gf.Vec3d(0.0, 0.0, -1.0)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(*direction))
    scene.CreateGravityMagnitudeAttr(float(magnitude))

    for surface_value in data.get("surfaces", ()):
        surface = _mapping(surface_value)
        surface_id = str(surface["surface_id"])
        surface_path = f"/World/Environment/{_name(surface_id)}"
        if surface.get("type", "box") == "cylinder":
            prim = UsdGeom.Cylinder.Define(stage, surface_path).GetPrim()
        else:
            prim = UsdGeom.Cube.Define(stage, surface_path).GetPrim()
        _entity(
            prim,
            entity_id=surface_id,
            asset_id=str(surface.get("asset_id", data.get("environment_id", "environment"))),
            role="support_surface",
            static=True,
        )
        _pose(prim, _mapping(surface.get("pose", {})))
        size = _vector_mapping(surface.get("size", {}), default=(0.5, 0.5, 0.05))
        if prim.IsA(UsdGeom.Cylinder):
            UsdGeom.Cylinder(prim).CreateRadiusAttr(float(size[0]))
            UsdGeom.Cylinder(prim).CreateHeightAttr(float(size[2]) * 2.0)
        else:
            UsdGeom.Xformable(prim).AddScaleOp().Set(size)
        color = tuple(float(value) for value in surface.get("rgba", (0.42, 0.36, 0.28, 1.0)))
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color[:3])])
        UsdGeom.Gprim(prim).CreateDisplayOpacityAttr([color[3]])
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        _bind_physics_material(
            stage,
            prim,
            f"surface_{surface_id}",
            surface.get("friction", (1.2, 0.2, 0.2)),
        )

    for camera_value in data.get("cameras", ()):
        camera = _mapping(camera_value)
        camera_prim = UsdGeom.Camera.Define(
            stage, f"/World/{_name(str(camera['camera_id']))}"
        ).GetPrim()
        camera_pose = _mapping(camera.get("pose", {}))
        _pose(camera_prim, camera_pose)
        xyaxes = camera_pose.get("xyaxes")
        if isinstance(xyaxes, (list, tuple)) and len(xyaxes) == 6:
            _set_xyaxes(camera_prim, tuple(float(value) for value in xyaxes))

    for light_value in data.get("lighting", ()):
        light = _mapping(light_value)
        light_prim = UsdLux.DistantLight.Define(
            stage, f"/World/{_name(str(light['light_id']))}"
        ).GetPrim()
        position = _vector(light.get("position", (0.0, 0.0, 0.0)))
        direction = _vector(light.get("direction", (0.0, 0.0, -1.0)))
        xform = UsdGeom.Xformable(light_prim)
        xform.AddTranslateOp().Set(position)
        if direction.GetLength():
            rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), direction.GetNormalized())
            xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation.GetQuat())


def _author_robot(stage: Usd.Stage, value: object) -> None:
    if not isinstance(value, Mapping):
        return
    robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    _entity(
        robot,
        entity_id="robot",
        asset_id=str(value["asset_id"]),
        role="robot",
        static=True,
    )
    _pose(robot, _mapping(value.get("pose", {})))


def _author_objects(stage: Usd.Stage, values: object) -> dict[str, Sdf.Path]:
    UsdGeom.Scope.Define(stage, "/World/Objects")
    paths: dict[str, Sdf.Path] = {}
    if not isinstance(values, (list, tuple)):
        return paths
    for value in values:
        obj = _mapping(value)
        name = str(obj["name"])
        prim = UsdGeom.Xform.Define(stage, f"/World/Objects/{_name(name)}").GetPrim()
        paths[f"object:{name}"] = prim.GetPath()
        _entity(
            prim,
            entity_id=name,
            asset_id=str(obj["asset_id"]),
            role=str(obj.get("role", "interactive_object")),
            static=bool(obj.get("static", False)),
        )
        _pose(prim, _mapping(obj.get("pose", {})))
        initial = _mapping(obj.get("initial_state", {}))
        mass = float(initial.get("mass_kg", 0.1))
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        if not bool(obj.get("static", False)):
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        inertial = initial.get("inertial")
        if isinstance(inertial, Mapping):
            center = _vector_mapping(inertial.get("center_of_mass", {}))
            diagonal = _vector(inertial.get("diagonal_inertia_kg_m2", (0.0, 0.0, 0.0)))
            prim.CreateAttribute("physics:centerOfMass", Sdf.ValueTypeNames.Point3f).Set(
                Gf.Vec3f(*center)
            )
            prim.CreateAttribute("physics:diagonalInertia", Sdf.ValueTypeNames.Float3).Set(
                Gf.Vec3f(*diagonal)
            )
        _bind_physics_material(
            stage,
            prim,
            f"object_{name}",
            initial.get("friction", (0.7, 0.005, 0.0001)),
        )
        friction = initial.get("friction")
        if isinstance(friction, (list, tuple)) and len(friction) == 3:
            _attr(
                prim,
                "robosim:mujoco:friction",
                Sdf.ValueTypeNames.Double3,
                Gf.Vec3d(*(float(value) for value in friction)),
            )
        contact = initial.get("contact")
        if isinstance(contact, Mapping):
            _author_contact(prim, contact)
    return paths


def _author_task(stage: Usd.Stage, data: Mapping[str, Any]) -> None:
    task = stage.DefinePrim("/World/Task", "Scope")
    _schemas(task, "RobosimTaskAPI")
    _attr(
        task,
        "robosim:task:objective",
        Sdf.ValueTypeNames.String,
        str(data.get("objective", "test objective")),
    )


def _author_relationships(
    stage: Usd.Stage,
    values: object,
    object_paths: Mapping[str, Sdf.Path],
) -> None:
    stage.DefinePrim("/World/Relationships", "Scope")
    if not isinstance(values, (list, tuple)):
        return
    for value in values:
        relationship = _mapping(value)
        relation_id = str(relationship["relation_id"])
        prim = stage.DefinePrim(f"/World/Relationships/{_name(relation_id)}", "RobosimRelationship")
        relation_type = str(relationship["type"])
        _attr(prim, "robosim:relationship:type", Sdf.ValueTypeNames.Token, relation_type)
        subject = str(relationship["subject"])
        prim.CreateRelationship("robosim:relationship:subject").SetTargets(
            [_relationship_path(subject, object_paths)]
        )
        target = str(relationship["object"])
        prim.CreateRelationship("robosim:relationship:object").SetTargets(
            [_relationship_path(target, object_paths)]
        )
        parameters = _mapping(relationship.get("parameters", {}))
        _attr(
            prim,
            "robosim:relationship:minDistanceM",
            Sdf.ValueTypeNames.Double,
            float(parameters.get("min_distance_m", 0.0)),
        )
        _attr(
            prim,
            "robosim:relationship:tolerance",
            Sdf.ValueTypeNames.Double,
            float(parameters.get("position_tolerance_m", 0.0)),
        )


def _author_contact(prim: Usd.Prim, contact: Mapping[str, Any]) -> None:
    for source, name, value_type in (
        ("margin_m", "mjc:margin", Sdf.ValueTypeNames.Double),
        ("gap_m", "mjc:gap", Sdf.ValueTypeNames.Double),
        ("solref", "mjc:solref", Sdf.ValueTypeNames.DoubleArray),
        ("solimp", "mjc:solimp", Sdf.ValueTypeNames.DoubleArray),
    ):
        if contact.get(source) is not None:
            _attr(prim, name, value_type, contact[source])


def _relationship_path(
    entity_ref: str,
    object_paths: Mapping[str, Sdf.Path],
) -> Sdf.Path:
    if entity_ref in object_paths:
        return object_paths[entity_ref]
    entity_type, _, entity_id = entity_ref.partition(":")
    parent = "Environment" if entity_type == "surface" else "Objects"
    return Sdf.Path(f"/World/{parent}/{_name(entity_id)}")


def _bind_physics_material(
    stage: Usd.Stage,
    prim: Usd.Prim,
    name: str,
    friction: object,
) -> None:
    if isinstance(friction, (int, float)):
        sliding = float(friction)
    elif isinstance(friction, (list, tuple)) and friction:
        sliding = float(friction[0])
    else:
        sliding = 0.7
    material = UsdShade.Material.Define(stage, f"/World/Materials/{_name(name)}")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateDynamicFrictionAttr(sliding)
    physics.CreateStaticFrictionAttr(sliding)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _entity(
    prim: Usd.Prim,
    *,
    entity_id: str,
    asset_id: str,
    role: str,
    static: bool,
) -> None:
    _schemas(prim, "RobosimEntityAPI")
    _attr(prim, "robosim:entity:id", Sdf.ValueTypeNames.String, entity_id)
    _attr(prim, "robosim:entity:assetId", Sdf.ValueTypeNames.String, asset_id)
    _attr(prim, "robosim:entity:role", Sdf.ValueTypeNames.Token, role)
    _attr(prim, "robosim:entity:static", Sdf.ValueTypeNames.Bool, static)


def _pose(prim: Usd.Prim, value: Mapping[str, Any]) -> None:
    position = _vector_mapping(value.get("position", {}))
    orientation = _mapping(value.get("orientation", {}))
    quaternion = Gf.Quatd(
        float(orientation.get("w", 1.0)),
        Gf.Vec3d(
            float(orientation.get("x", 0.0)),
            float(orientation.get("y", 0.0)),
            float(orientation.get("z", 0.0)),
        ),
    )
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(position)
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quaternion)


def _set_xyaxes(prim: Usd.Prim, values: tuple[float, ...]) -> None:
    x_axis = Gf.Vec3d(*values[:3]).GetNormalized()
    y_axis = Gf.Vec3d(*values[3:]).GetNormalized()
    z_axis = Gf.Cross(x_axis, y_axis).GetNormalized()
    rotation = Gf.Matrix3d(1.0)
    rotation.SetRow(0, x_axis)
    rotation.SetRow(1, y_axis)
    rotation.SetRow(2, z_axis)
    UsdGeom.Xformable(prim).AddOrientOp(UsdGeom.XformOp.PrecisionDouble, opSuffix="xyaxes").Set(
        rotation.ExtractRotation().GetQuat()
    )


def _schemas(prim: Usd.Prim, *names: str) -> None:
    for name in names:
        prim.AddAppliedSchema(name)


def _attr(
    prim: Usd.Prim,
    name: str,
    value_type: Sdf.ValueTypeName,
    value: object,
) -> None:
    prim.CreateAttribute(name, value_type).Set(value)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _vector(value: object) -> Gf.Vec3d:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return Gf.Vec3d(0.0)
    return Gf.Vec3d(*(float(item) for item in value))


def _vector_mapping(
    value: object,
    *,
    default: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Gf.Vec3d:
    data = _mapping(value)
    return Gf.Vec3d(
        float(data.get("x", default[0])),
        float(data.get("y", default[1])),
        float(data.get("z", default[2])),
    )


def _meters_per_unit(units: str) -> float:
    return {"m": 1.0, "cm": 0.01, "mm": 0.001}.get(units, 1.0)


def _name(value: str) -> str:
    return "_".join(value.strip().replace("-", "_").split())
