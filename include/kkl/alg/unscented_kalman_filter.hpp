/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>

namespace kkl {
  namespace alg {

/**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class UnscentedKalmanFilterX {
  // 重定义动态维度向量 和 矩阵
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension, 状态量维度
   * @param input_dim            input vector dimension，控制量维度
   * @param measurement_dim      measurement vector dimension， 测量量维度
   * @param process_noise        process noise covariance (state_dim x state_dim)， 过程噪声协方差
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)， 测量噪声协防差
   * @param mean                 initial mean， 状态量
   * @param cov                  initial covariance， 状态协方差
   */
  UnscentedKalmanFilterX(const System& system, int state_dim, int input_dim, int measurement_dim, const MatrixXt& process_noise, const MatrixXt& measurement_noise, const VectorXt& mean, const MatrixXt& cov)
    : state_dim(state_dim),
    input_dim(input_dim),
    measurement_dim(measurement_dim),
    N(state_dim),
    M(input_dim),
    K(measurement_dim),
    S(2 * state_dim + 1),
    mean(mean),
    cov(cov),
    system(system),
    process_noise(process_noise),
    measurement_noise(measurement_noise),
    lambda(1),                   // sigma采样点中的可调参数K
    normal_dist(0.0, 1.0)
  {
    // 初始化采样点权重列表，S*1
    weights.resize(S, 1);
    // sigma采样点，共采2*N+1个, 其中N为状态维度，根据mean对称分布
    sigma_points.resize(S, N);
    ext_weights.resize(2 * (N + K) + 1, 1);
    // 采样点个数为， 状态维度为N+K
    ext_sigma_points.resize(2 * (N + K) + 1, N + K);
    // 个数为sigma采样点个数，状态维度为K个测量值，
    expected_measurements.resize(2 * (N + K) + 1, K);

    // initialize weights for unscented filter
    // 初始化权重
    weights[0] = lambda / (N + lambda);
    for (int i = 1; i < 2 * N + 1; i++) {
      weights[i] = 1 / (2 * (N + lambda));
    }

    // weights for extended state space which includes error variances
    // 观测方程中的权重系数
    ext_weights[0] = lambda / (N + K + lambda);
    for (int i = 1; i < 2 * (N + K) + 1; i++) {
      ext_weights[i] = 1 / (2 * (N + K + lambda));
    }
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict(const VectorXt& control) {
    // calculate sigma points
    // 修正协方差，
    ensurePositiveFinite(cov);
    // 根据mean和协方差获取sigma采样点
    computeSigmaPoints(mean, cov, sigma_points);
    // 对所有采样点进行控制量预测，获取预测后的采样点
    // 即非线性变换过程
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i), control);
    }

    // 过程噪声协防差
    const auto& R = process_noise;

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());
    // 新的状态均值和协方差
    mean_pred.setZero();
    cov_pred.setZero();
    // 无损变换，更新后验采样点的均值与方差
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    // 方差
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    // 增加噪声协方差
    cov_pred += R;

    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct
   * @param measurement  measurement vector
   */
  void correct(const VectorXt& measurement) {
    // create extended state space which includes error variances
    // 状态维度增加K个，即测量状态维度,增加的为测量噪声
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
    // 前面N个状态为状态均值
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
    // 前面N×N为状态协方差
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
    // 右下角为测量噪声
    ext_cov_pred.bottomRightCorner(K, K) = measurement_noise;

    // 确保为正定矩阵
    ensurePositiveFinite(ext_cov_pred);
    // 与预测一致进行采样sigma点
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points);

    // unscented transform， 无损观测方程
    // 测量值仅有7个值pose 和 姿态， 计算每个采样点后的测量值
    expected_measurements.setZero();
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      // 观测方程，仅更新前n个，即真正状态量
      expected_measurements.row(i) = system.h(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));
      // 增加观测误差，即后K个
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
    }

    // 测量维度仅有K个
    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    // 根据权重计算状态均值
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }
    // 根据权重计算协方差
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance
    // 计算协方差变换和增益
    MatrixXt sigma = MatrixXt::Zero(N + K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      // 预测采样的测量状态与预测均值状态 差
      auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
      // 测量采样状态 与 测量均值状态差
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      // 统计交叉协方差
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    // 获得增益
    kalman_gain = sigma * expected_measurement_cov.inverse();
    const auto& K = kalman_gain;

    // 计算更新后的状态均值
    VectorXt ext_mean = ext_mean_pred + K * (measurement - expected_measurement_mean);
    // 计算更新后的协方差
    MatrixXt ext_cov = ext_cov_pred - K * expected_measurement_cov * K.transpose();

    // 获取关心的状态和协方差
    mean = ext_mean.topLeftCorner(N, 1);
    cov = ext_cov.topLeftCorner(N, N);
  }

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }
  const MatrixXt& getSigmaPoints() const { return sigma_points; }

  System& getSystem() { return system; }
  const System& getSystem() const { return system; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }

  const MatrixXt& getKalmanGain() const { return kalman_gain; }

  /*			setter			*/
  UnscentedKalmanFilterX& setMean(const VectorXt& m) { mean = m;			return *this; }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s) { cov = s;			return *this; }

  UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& p) { process_noise = p;			return *this; }
  UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m) { measurement_noise = m;	return *this; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim;
  const int input_dim;
  const int measurement_dim;

  const int N;
  const int M;
  const int K;
  const int S;

public:
  VectorXt mean;
  MatrixXt cov;

  System system;
  MatrixXt process_noise;		//
  MatrixXt measurement_noise;	//

  T lambda;
  VectorXt weights;

  MatrixXt sigma_points;

  VectorXt ext_weights;
  MatrixXt ext_sigma_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  computeSigmaPointscalculated sigma points
   * 根据状态量和协方差， 进行采样
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
    // 获取高斯维度
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);
    // 需要正定的才可Cholesky分解 A = L*L_T
    Eigen::LLT<MatrixXt> llt;
    // Cholesky分解, lambda为可调参数，这里取1，与mean的距离通常为标准差的倍数
    llt.compute((n + lambda) * cov);
    MatrixXt l = llt.matrixL();

    // 第一个点为均值，共取2n+1个
    sigma_points.row(0) = mean;
    // 其他为对称点，间距为协方差的整数倍
    for (int i = 0; i < n; i++) {
      sigma_points.row(1 + i * 2) = mean + l.col(i);
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
    }
  }

  /**
   * @brief make covariance matrix positive finite 确保协方差正定
   * 即判断特征值为0或为负，并修正
   * @param cov  covariance matrix
   */
  void ensurePositiveFinite(MatrixXt& cov) {
    return;
    const double eps = 1e-9;

    Eigen::EigenSolver<MatrixXt> solver(cov);
    MatrixXt D = solver.pseudoEigenvalueMatrix();
    MatrixXt V = solver.pseudoEigenvectors();
    for (int i = 0; i < D.rows(); i++) {
      if (D(i, i) < eps) {
        D(i, i) = eps;
      }
    }

    cov = V * D * V.inverse();
  }

public:
  MatrixXt kalman_gain;

  std::mt19937 mt;
  std::normal_distribution<T> normal_dist;
};

  }
}


#endif
