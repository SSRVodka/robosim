cmake_minimum_required(VERSION 3.10.2)

project(ipads_desc)

find_package(ament_cmake REQUIRED)

ament_package()

# install(DIRECTORY config launch meshes urdf DESTINATION share/${PROJECT_NAME})
install(DIRECTORY meshes urdf DESTINATION share/${PROJECT_NAME})
