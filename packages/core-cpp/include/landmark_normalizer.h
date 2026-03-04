#ifndef SIGNBRIDGE_LANDMARK_NORMALIZER_H
#define SIGNBRIDGE_LANDMARK_NORMALIZER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Normalize 21 hand landmarks from MediaPipe.
 *
 * Algorithm:
 *   1. Translate all landmarks so wrist (landmark 0) is at origin.
 *   2. Scale by the distance between wrist (0) and middle-finger MCP (9)
 *      to achieve scale invariance.
 *
 * @param raw   Input:  float[21][3] raw landmarks from MediaPipe
 * @param out   Output: float[21][3] normalized landmarks
 */
void normalize_landmarks(float raw[21][3], float out[21][3]);

#ifdef __cplusplus
}
#endif

#endif // SIGNBRIDGE_LANDMARK_NORMALIZER_H
