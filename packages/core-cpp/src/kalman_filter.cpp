/**
 * kalman_filter.cpp
 * SignBridge Core - 1D Kalman Filter for Landmark Smoothing
 *
 * Implements scalar 1D Kalman filter applied independently to each
 * of the 63 coordinates (21 landmarks × 3 axes).
 *
 * Default hyperparameters: Q = 0.001, R = 0.01, initial P = 1.0
 */

#include "kalman_filter.h"

// Static instance for C-compatible API
static LandmarkKalmanBank g_kalman_bank;

void kalman_update(float* landmarks, float* smoothed) {
    // Reinterpret flat arrays as [21][3]
    float(*lm)[3]  = reinterpret_cast<float(*)[3]>(landmarks);
    float(*sm)[3]  = reinterpret_cast<float(*)[3]>(smoothed);
    g_kalman_bank.update(lm, sm);
}

void kalman_reset() {
    g_kalman_bank.reset();
}
