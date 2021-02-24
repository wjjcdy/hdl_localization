#ifndef POSE_SYSTEM_HPP
#define POSE_SYSTEM_HPP

#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief Definition of system to be estimated by ukf
 * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
 */
class PoseSystem {
public:
  typedef float T;
  typedef Eigen::Matrix<T, 3, 1> Vector3t;
  typedef Eigen::Matrix<T, 4, 4> Matrix4t;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Quaternion<T> Quaterniont;
public:
  PoseSystem() {
    dt = 0.01;
  }


  // system equation
  // 控制量
  VectorXt f(const VectorXt& state, const VectorXt& control) const {
    VectorXt next_state(16);

    // 上刻位置
    Vector3t pt = state.middleRows(0, 3);
    // 上刻线速度
    Vector3t vt = state.middleRows(3, 3);
    // 上刻姿态
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    // 加速度静态偏移
    Vector3t acc_bias = state.middleRows(10, 3);
    // 角速度静态偏移
    Vector3t gyro_bias = state.middleRows(13, 3);

    // 新的线加速度
    Vector3t raw_acc = control.middleRows(0, 3);
    // 新的角速度
    Vector3t raw_gyro = control.middleRows(3, 3);

    // position， 采用上刻的线速度计算
    next_state.middleRows(0, 3) = pt + vt * dt;					//

    // velocity， 由于imu线加速一般误差较大，最好不用，否则反而会代入误差
    // 同时该代码表明imu严格水平安装
    // 静态加速度
    Vector3t g(0.0f, 0.0f, -9.80665f);
    // 获取相对加速度
    Vector3t acc_ = raw_acc - acc_bias;
    // 转换到姿态坐标系下的线加速度
    Vector3t acc = qt * acc_;
    // 预测速度
    next_state.middleRows(3, 3) = vt; // + (acc - g) * dt;		// acceleration didn't contribute to accuracy due to large noise

    // orientation
    // 计算角速度变换量
    Vector3t gyro = raw_gyro - gyro_bias;
    // 更换成4元数
    Quaterniont dq(1, gyro[0] * dt / 2, gyro[1] * dt / 2, gyro[2] * dt / 2);
    dq.normalize();
    // 更新新的位置
    Quaterniont qt_ = (qt * dq).normalized();
    next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();

    // 线角速度和角度静态偏移
    next_state.middleRows(10, 3) = state.middleRows(10, 3);		// constant bias on acceleration
    next_state.middleRows(13, 3) = state.middleRows(13, 3);		// constant bias on angular velocity

    return next_state;
  }

  // observation equation
  // 观测方程
  VectorXt h(const VectorXt& state) const {
    VectorXt observation(7);
    observation.middleRows(0, 3) = state.middleRows(0, 3);
    observation.middleRows(3, 4) = state.middleRows(6, 4).normalized();

    return observation;
  }

  double dt;
};

}

#endif // POSE_SYSTEM_HPP
