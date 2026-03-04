#ifndef SIGNBRIDGE_KALMAN_FILTER_H
#define SIGNBRIDGE_KALMAN_FILTER_H

/**
 * 1D Kalman Filter for landmark coordinate smoothing.
 *
 * Applied independently to each of the 63 coordinates
 * (21 landmarks × 3 axes) to reduce jitter from MediaPipe.
 *
 * Default hyperparameters:
 *   Q (process noise)     = 0.001
 *   R (measurement noise) = 0.01
 *   P (initial covariance)= 1.0
 */
struct KalmanFilter1D {
    float Q = 0.001f;   // process noise covariance
    float R = 0.01f;    // measurement noise covariance
    float P = 1.0f;     // estimate error covariance
    float x = 0.0f;     // state estimate

    /**
     * Update the filter with a new measurement.
     * @param z  New measurement value
     * @return   Smoothed estimate
     */
    float update(float z) {
        // Predict step
        P = P + Q;

        // Kalman gain
        float K = P / (P + R);

        // Update step
        x = x + K * (z - x);
        P = (1.0f - K) * P;

        return x;
    }

    /**
     * Reset the filter to initial state.
     */
    void reset() {
        P = 1.0f;
        x = 0.0f;
    }
};

/**
 * Bank of Kalman filters for all 21 hand landmarks (3 axes each = 63 filters).
 */
struct LandmarkKalmanBank {
    KalmanFilter1D filters[21][3];

    /**
     * Update all 63 filters with new landmark measurements.
     *
     * @param landmarks  Input:  float[21][3] raw (normalized) landmarks
     * @param smoothed   Output: float[21][3] smoothed landmarks
     */
    void update(float landmarks[21][3], float smoothed[21][3]) {
        for (int i = 0; i < 21; i++) {
            for (int j = 0; j < 3; j++) {
                smoothed[i][j] = filters[i][j].update(landmarks[i][j]);
            }
        }
    }

    /**
     * Reset all filters.
     */
    void reset() {
        for (int i = 0; i < 21; i++) {
            for (int j = 0; j < 3; j++) {
                filters[i][j].reset();
            }
        }
    }
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * C-compatible wrapper: Update Kalman filter bank with new landmarks.
 * Uses a static internal bank instance.
 *
 * @param landmarks  Input:  float[63] flattened landmarks (21*3)
 * @param smoothed   Output: float[63] smoothed landmarks
 */
void kalman_update(float* landmarks, float* smoothed);

/**
 * C-compatible wrapper: Reset the Kalman filter bank.
 */
void kalman_reset();

#ifdef __cplusplus
}
#endif

#endif // SIGNBRIDGE_KALMAN_FILTER_H
