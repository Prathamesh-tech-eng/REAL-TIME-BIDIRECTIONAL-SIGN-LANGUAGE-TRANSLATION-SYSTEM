/**
 * landmark_normalizer.cpp
 * SignBridge Core - Hand Landmark Normalization
 *
 * Input:  float[21][3] raw landmarks from MediaPipe Hand Landmarker
 * Output: float[21][3] normalized landmarks
 *
 * Algorithm:
 *   1. Translate: subtract wrist position (landmark 0) from all landmarks
 *   2. Scale: divide by wrist-to-MCP9 distance for scale invariance
 */

#include "landmark_normalizer.h"
#include <cmath>

void normalize_landmarks(float raw[21][3], float out[21][3]) {
    // 1. Translate: subtract wrist position so landmark 0 = origin
    float wx = raw[0][0], wy = raw[0][1], wz = raw[0][2];
    for (int i = 0; i < 21; i++) {
        out[i][0] = raw[i][0] - wx;
        out[i][1] = raw[i][1] - wy;
        out[i][2] = raw[i][2] - wz;
    }

    // 2. Scale: divide by distance from wrist (0) to middle-finger MCP (9)
    float dx = out[9][0], dy = out[9][1], dz = out[9][2];
    float scale = std::sqrt(dx * dx + dy * dy + dz * dz);

    // Guard against division by zero (degenerate hand pose)
    if (scale < 1e-6f) {
        scale = 1.0f;
    }

    for (int i = 0; i < 21; i++) {
        out[i][0] /= scale;
        out[i][1] /= scale;
        out[i][2] /= scale;
    }
}
