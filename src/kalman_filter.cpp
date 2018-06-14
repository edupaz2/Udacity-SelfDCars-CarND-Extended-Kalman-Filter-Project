#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

//#define LOG_DEBUG

#ifdef LOG_DEBUG
#include <iostream>
using namespace std;
#define DEBUG(x) do { cout << x << endl; } while (0)
#else
#define DEBUG(x)
#endif

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  DEBUG("KalmanFilter::Predict");
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
  DEBUG("KalmanFilter::Predict end");
}

void KalmanFilter::Update(const VectorXd &z) {
  DEBUG("KalmanFilter::Update");
  // y = z − H*x'
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

  DEBUG("KalmanFilter::Update end");
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  DEBUG("KalmanFilter::UpdateEKF");
  // y = z − h(x')

  // where h(x') is Vector(3)
  // rho = √(px**2 + py**2)
  // phi = tan-1 (y/x)
  // rho_dot = (px*vy + py*vx) / ( √(px**2+py**2) )
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  float rho = sqrt(px*px + py*py);      // Range
  float phi = atan2(py, px);                  // Bearing
  float rho_dot = (px*vx + py*vy)/rho;  // Range rate

  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

  DEBUG("KalmanFilter::UpdateEKF end");
}
