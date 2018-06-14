#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

//#define LOG_DEBUG

#ifdef LOG_DEBUG
#include <iostream>
using namespace std;
#define DEBUG(x) do { cout << x << endl; } while (0)
#else
#define DEBUG(x)
#endif


/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  this->is_initialized_ = false;

  this->previous_timestamp_ = 0;

  // initializing matrices
  this->R_laser_ = MatrixXd(2, 2);
  this->R_radar_ = MatrixXd(3, 3);
  this->H_laser_ = MatrixXd(2, 4);
  this->H_radar_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  this->R_laser_ << 0.0225, 0,
                    0, 0.0225;

  //measurement covariance matrix - radar
  this->R_radar_ << 0.09, 0, 0,
                    0, 0.0009, 0,
                    0, 0, 0.09;

  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  //measurement matrix - laser
  this->H_laser_ << 1, 0, 0, 0,
                    0, 1, 0, 0;

  //measurement matrix - radar
  this->H_radar_ << 1, 1, 0, 0,
                    1, 1, 0, 0,
                    1, 1, 1, 1;

  //state covariance matrix P
  this->ekf_.P_ = MatrixXd(4, 4);
  this->ekf_.P_ <<  1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1000, 0,
                    0, 0, 0, 1000;

  //the initial transition matrix F_
  this->ekf_.F_ = MatrixXd(4, 4);
  this->ekf_.F_ <<  1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    this->ekf_.x_ = VectorXd(4);
    this->ekf_.x_ << 1, 1, 1, 1;

    //set the state with the initial location and zero velocity
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      Input: #R(for radar) meas_rho meas_phi meas_rho_dot timestamp gt_px gt_py gt_vx gt_vy
      */
      DEBUG("FusionEKF::ProcessMeasurement Init Radar");
      // px = rho * cod(phi)
      // py = rho * sin(phi)

      this->ekf_.x_(0) = measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]);
      this->ekf_.x_(1) = measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      Input: #L(for laser) meas_px meas_py timestamp gt_px gt_py gt_vx gt_vy
      */
      DEBUG("FusionEKF::ProcessMeasurement Init Lidar");
      this->ekf_.x_(0) = measurement_pack.raw_measurements_[0]; // meas_px
      this->ekf_.x_(1) = measurement_pack.raw_measurements_[1]; // meas_py
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  DEBUG("FusionEKF::ProcessMeasurement Measurements Compute time");

  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  //1. Modify the F matrix so that the time is integrated
  //2. Set the process covariance matrix Q
  //3. Call the Kalman Filter predict() function
  //4. Call the Kalman Filter update() function
  // with the most recent raw measurements_
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  float noise_ax = 9;
  float noise_ay = 9;

  //Modify the F matrix so that the time is integrated
  this->ekf_.F_(0, 2) = dt;
  this->ekf_.F_(1, 3) = dt;

  //set the process covariance matrix Q
  this->ekf_.Q_ = MatrixXd(4, 4);
  this->ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
		   0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
		   dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
		   0, dt_3/2*noise_ay, 0, dt_2*noise_ay;


  this->ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  DEBUG("FusionEKF::ProcessMeasurement Compute H and R");
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    this->H_radar_ = this->tools.CalculateJacobian(this->ekf_.x_);
    this->ekf_.H_ = this->H_radar_;
    this->ekf_.R_ = this->R_radar_;
    this->ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    this->ekf_.H_ = this->H_laser_;
    this->ekf_.R_ = this->R_laser_;
    this->ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
