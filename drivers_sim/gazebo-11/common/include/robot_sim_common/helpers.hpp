#pragma once

#include <geometry_msgs/msg/pose.hpp>


namespace robot_sim {

bool is_valid_double(double x);
bool is_valid_pose(const geometry_msgs::msg::Pose &pose);
bool is_valid_position(double x, double y, double z);
bool is_valid_position2d(double x, double y, double z);
bool is_valid_quaternion(double x, double y, double z, double w);
// NOTE: throws if invalid so you should check `is_valid_quaternion`
void normalize_quaternion(double &x, double &y, double &z, double &w);

}   // namespace robot_sim
