# Robot asset semantic parity

## Franka asset-backed control semantics（进行中）

- [ ] Treat the robot asset selected by CSD `robot:id` as the single source of
      Franka control semantics; do not add duplicate per-scene control fields
      to CSD.
- [ ] Define the robot asset resolver input and cache hash for the complete
      URDF, SRDF, controller, joint-limit, mesh, texture, and metadata closure.
- [ ] Materialize that closure into every backend realization package without
      runtime references to `drivers_sim` or an external asset cache.
- [ ] Make Gazebo consume the copied Franka URDF/SRDF/controller artifacts and
      expose arm, hand, arm+hand groups, named states, joint limits, and end
      effector metadata through its manifest and `GetRobotSpec`.
- [ ] Make MuJoCo consume the same robot semantic profile and verify that its
      `GetRobotSpec`, group-scoped control, and command-state semantics match
      Gazebo where their supported control modes overlap.
- [ ] Add focused tests for closure portability, cache invalidation, robot spec
      parity, arm position control, and gripper position control; reject any
      unsupported profile/backend mapping with a typed blocker.
