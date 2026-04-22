## Robot Definition Naming Specification

- Root directory Name: `<robot-name>`

- Robot Description (URDF) Package Name: `<robot-name>_desc`

    - Navigable Extension: Navigation2 Configurations `config/nav2_params*.yaml` (NOTE: This is robot relative config: check `robot_radius` / `motion_model` or other entries)

- Moveit2 Configuration Package Name: `<robot-name>_moveit`

- MuJoCo Description (MJCF) Package Name: `<robot-name>_mujoco`
