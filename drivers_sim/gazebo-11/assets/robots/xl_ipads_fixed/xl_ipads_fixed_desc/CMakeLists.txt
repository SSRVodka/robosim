cmake_minimum_required(VERSION 3.10.2)

project(xl_ipads_fixed_desc)

find_package(ament_cmake REQUIRED)

ament_package()

# install(DIRECTORY config launch meshes urdf DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config launch meshes urdf DESTINATION share/${PROJECT_NAME})
