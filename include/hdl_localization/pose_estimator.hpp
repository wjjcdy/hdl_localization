#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <memory>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pclomp/ndt_omp.h>
#include <pcl/filters/voxel_grid.h>

#include <hdl_localization/pose_system.hpp>
#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
  using PointT = pcl::PointXYZI;

  /**
   * @brief constructor
   * @param registration        registration method
   * @param stamp               timestamp
   * @param pos                 initial position
   * @param quat                initial orientation
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const ros::Time& stamp, const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, double cool_time_duration = 1.0)
    : init_stamp(stamp),
      registration(registration),
      cool_time_duration(cool_time_duration)
  {
    // 预测单位时间噪声,即控制噪声，
    process_noise = Eigen::MatrixXf::Identity(16, 16);
    process_noise.middleRows(0, 3) *= 1.0;
    process_noise.middleRows(3, 3) *= 1.0;
    process_noise.middleRows(6, 4) *= 0.5;
    process_noise.middleRows(10, 3) *= 1e-6;
    process_noise.middleRows(13, 3) *= 1e-6;

    // 测量噪声
    Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
    // 点云结果测量姿态的协防差
    measurement_noise.middleRows(0, 3) *= 0.01;
    // 点云匹配结果测量
    measurement_noise.middleRows(3, 4) *= 0.001;

    // ukf 状态量
    Eigen::VectorXf mean(16);
    // 0,1,2 pose
    mean.middleRows(0, 3) = pos;
    // 3,4,5 为3个方向线速度
    mean.middleRows(3, 3).setZero();
    // 6,7,8,9 为姿态
    mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());
    // 
    mean.middleRows(10, 3).setZero();
    mean.middleRows(13, 3).setZero();

    // 状态协方差
    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

    // 位置系统类
    PoseSystem system;
    ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>(system, 16, 6, 7, process_noise, measurement_noise, mean, cov));
  }

  /**
   * @brief predict
   * @param stamp    timestamp
   * @param acc      acceleration
   * @param gyro     angular velocity
   * 采用ukf 滤波器预测下刻位置
   */
  void predict(const ros::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
    // 判断时间戳是否更新
    if((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
      prev_stamp = stamp;
      return;
    }

    // 计算两次时间差
    double dt = (stamp - prev_stamp).toSec();
    // 更新时间戳
    prev_stamp = stamp;

    // 设置控制量噪声，即预测噪声
    ukf->setProcessNoiseCov(process_noise * dt);
    // 设置滤波器中间隔时间
    ukf->system.dt = dt;

    // 设置控制量
    Eigen::VectorXf control(6);
    control.head<3>() = acc;   // 线加速
    control.tail<3>() = gyro;  // 角速度

    // 根据线速度和加速度预测此刻状态量
    ukf->predict(control);     // ukf 预测,主要更新的是均值和协方差
  }

  /**
   * @brief correct
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  pcl::PointCloud<PointT>::Ptr correct(const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    // 定义初始位置矩阵，即滤波器的均值
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    // 姿态
    init_guess.block<3, 3>(0, 0) = quat().toRotationMatrix();
    // 位置
    init_guess.block<3, 1>(0, 3) = pos();

    // 将输入的点云进行点云匹配
    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    registration->setInputSource(cloud);
    // 包含初始值，输出对齐点云
    registration->align(*aligned, init_guess);

    // 获取点云匹配结果
    Eigen::Matrix4f trans = registration->getFinalTransformation();
    //获取位置和姿态
    Eigen::Vector3f p = trans.block<3, 1>(0, 3);
    Eigen::Quaternionf q(trans.block<3, 3>(0, 0));

    // 保证初始的和匹配后的姿态系数方向一致
    if(quat().coeffs().dot(q.coeffs()) < 0.0f) {
      q.coeffs() *= -1.0f;
    }

    // 定义观测值，即位置和姿态
    Eigen::VectorXf observation(7);
    observation.middleRows(0, 3) = p;
    observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());

    // ukf观测方程
    ukf->correct(observation);
    return aligned;
  }

  /* getters */
  // 滤波器中的x，y，z三轴位置
  Eigen::Vector3f pos() const {
    return Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
  }

  // 滤波器中三轴线速度
  Eigen::Vector3f vel() const {
    return Eigen::Vector3f(ukf->mean[3], ukf->mean[4], ukf->mean[5]);
  }

  // 滤波器输出的姿态
  Eigen::Quaternionf quat() const {
    return Eigen::Quaternionf(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]).normalized();
  }

  // 当前点位置和姿态
  Eigen::Matrix4f matrix() const {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3, 3>(0, 0) = quat().toRotationMatrix();
    m.block<3, 1>(0, 3) = pos();
    return m;
  }

private:
  ros::Time init_stamp;         // when the estimator was initialized
  ros::Time prev_stamp;         // when the estimator was updated last time
  double cool_time_duration;    //

  Eigen::MatrixXf process_noise;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>> ukf;

  pcl::Registration<PointT, PointT>::Ptr registration;
};

}

#endif // POSE_ESTIMATOR_HPP
