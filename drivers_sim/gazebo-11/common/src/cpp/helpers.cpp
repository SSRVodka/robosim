
#include <cmath>
#include <exception>
#include <tf2/LinearMath/Quaternion.hpp>

#include "robot_sim_common/helpers.hpp"

namespace robot_sim {

bool is_valid_double(double x) {
    return std::isfinite(x);
}
bool is_valid_pose(const geometry_msgs::msg::Pose &pose) {
    // check position
    const auto &pos = pose.position;
    const auto &ori = pose.orientation;
    return is_valid_position(pos.x, pos.y, pos.z)
        && is_valid_quaternion(ori.x, ori.y, ori.z, ori.w);
}
bool is_valid_position(double x, double y, double z) {
    return is_valid_double(x) && is_valid_double(y) && is_valid_double(z);
}
bool is_valid_position2d(double x, double y, double z) {
    return is_valid_position(x, y, z) && std::abs(z) < 1e-5;
}
bool is_valid_quaternion(double x, double y, double z, double w) {
    tf2::Quaternion q(x, y, z, w);
    return q.length2() != 0.0;
}
void normalize_quaternion(double &x, double &y, double &z, double &w) {
    tf2::Quaternion q(x, y, z, w);
    if (!is_valid_quaternion(x, y, z, w)) {
        throw std::runtime_error("normalize_quaternion: invalid quaternion");
    }
    q.normalize();
    x = q.x(), y = q.y(), z = q.z(), w = q.w();
}

}   // namespace robot_sim