cmake_minimum_required(VERSION 3.10.2)
project(franka_panda_mujoco)
find_package(ament_cmake REQUIRED)

ament_package()

install(DIRECTORY franka_panda DESTINATION share/${PROJECT_NAME})
