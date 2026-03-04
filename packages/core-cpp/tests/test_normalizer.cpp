/**
 * test_normalizer.cpp
 * Tests for landmark_normalizer.cpp
 */

#include "landmark_normalizer.h"
#include <cstdio>
#include <cmath>
#include <cstring>

/**
 * Test 1: All-zeros input.
 * Expected: Output all zeros. No crash or division-by-zero.
 */
int test_normalizer_all_zeros() {
    float raw[21][3] = {};   // all zeros
    float out[21][3] = {};

    // Should not crash (division-by-zero guarded)
    normalize_landmarks(raw, out);

    // All outputs should be zero
    for (int i = 0; i < 21; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::fabs(out[i][j]) > 1e-6f) {
                printf("  ERROR: out[%d][%d] = %f, expected 0.0\n", i, j, out[i][j]);
                return 1;
            }
        }
    }
    printf("  All outputs are zero. No crash. Division-by-zero guard OK.\n");
    return 0;
}

/**
 * Test 2: Known hand pose.
 * Expected: Landmark 0 = [0,0,0]. Distance from landmark 0 to landmark 9 = 1.0.
 */
int test_normalizer_known_pose() {
    float raw[21][3];

    // Create a known pose:
    // Wrist at (10, 20, 30)
    raw[0][0] = 10.0f; raw[0][1] = 20.0f; raw[0][2] = 30.0f;

    // MCP9 at (10, 20, 35) → distance from wrist = 5.0 in Z
    raw[9][0] = 10.0f; raw[9][1] = 20.0f; raw[9][2] = 35.0f;

    // Fill other landmarks with various positions
    for (int i = 1; i < 21; i++) {
        if (i == 9) continue;
        raw[i][0] = 10.0f + (float)i * 0.5f;
        raw[i][1] = 20.0f + (float)i * 0.3f;
        raw[i][2] = 30.0f + (float)i * 0.2f;
    }

    float out[21][3];
    normalize_landmarks(raw, out);

    // Check 1: Landmark 0 (wrist) should be at origin
    float wrist_dist = std::sqrt(out[0][0]*out[0][0] + out[0][1]*out[0][1] + out[0][2]*out[0][2]);
    if (wrist_dist > 1e-6f) {
        printf("  ERROR: Wrist not at origin. Distance = %f\n", wrist_dist);
        return 1;
    }
    printf("  Landmark 0 (wrist) = [%f, %f, %f] ✓\n", out[0][0], out[0][1], out[0][2]);

    // Check 2: Distance from landmark 0 to landmark 9 should be 1.0
    float d9 = std::sqrt(out[9][0]*out[9][0] + out[9][1]*out[9][1] + out[9][2]*out[9][2]);
    if (std::fabs(d9 - 1.0f) > 1e-4f) {
        printf("  ERROR: Distance to landmark 9 = %f, expected 1.0\n", d9);
        return 1;
    }
    printf("  Distance wrist→MCP9 = %f ✓\n", d9);

    return 0;
}
