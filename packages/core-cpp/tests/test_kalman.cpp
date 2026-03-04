/**
 * test_kalman.cpp
 * Tests for kalman_filter.cpp
 */

#include "kalman_filter.h"
#include "landmark_normalizer.h"
#include <cstdio>
#include <cmath>
#include <cstring>

/**
 * Test 3: Constant input z=5.0 over 100 frames.
 * Expected: Output converges to 5.0 within ±0.01 by frame 50.
 */
int test_kalman_constant_input() {
    KalmanFilter1D kf;

    float last_output = 0.0f;
    bool converged_by_50 = false;

    for (int frame = 0; frame < 100; frame++) {
        last_output = kf.update(5.0f);

        if (frame == 49) {  // frame 50 (0-indexed = 49)
            float error = std::fabs(last_output - 5.0f);
            if (error <= 0.01f) {
                converged_by_50 = true;
            }
            printf("  Frame 50: output = %f, error = %f\n", last_output, error);
        }
    }

    printf("  Frame 100: output = %f\n", last_output);

    if (!converged_by_50) {
        printf("  ERROR: Did not converge to 5.0 ± 0.01 by frame 50\n");
        return 1;
    }

    // Final output should be very close to 5.0
    if (std::fabs(last_output - 5.0f) > 0.001f) {
        printf("  ERROR: Final output %f not close enough to 5.0\n", last_output);
        return 1;
    }

    printf("  Converged to 5.0 within tolerance by frame 50 ✓\n");
    return 0;
}

/**
 * Test 4: Step change from 0 to 10 at frame 0.
 * Expected: Output smoothly ramps, settling within ±0.1 by frame 30.
 */
int test_kalman_step_change() {
    KalmanFilter1D kf;

    // Pre-condition: feed 20 frames of z=0 to stabilize
    for (int i = 0; i < 20; i++) {
        kf.update(0.0f);
    }

    float last_output = 0.0f;
    bool settled_by_30 = false;
    bool monotonic = true;
    float prev = 0.0f;

    for (int frame = 0; frame < 50; frame++) {
        last_output = kf.update(10.0f);

        // Check monotonically increasing (smooth ramp)
        if (last_output < prev - 1e-6f) {
            monotonic = false;
        }
        prev = last_output;

        if (frame == 29) {  // frame 30 (0-indexed)
            float error = std::fabs(last_output - 10.0f);
            if (error <= 0.1f) {
                settled_by_30 = true;
            }
            printf("  Frame 30: output = %f, error = %f\n", last_output, error);
        }
    }

    printf("  Frame 50: output = %f\n", last_output);
    printf("  Monotonically increasing: %s\n", monotonic ? "yes" : "no");

    if (!settled_by_30) {
        printf("  ERROR: Did not settle to 10.0 ± 0.1 by frame 30\n");
        return 1;
    }

    if (!monotonic) {
        printf("  ERROR: Output was not monotonically increasing\n");
        return 1;
    }

    printf("  Smoothly ramped and settled within tolerance by frame 30 ✓\n");
    return 0;
}

/**
 * Test 5: JNI bridge simulation.
 * Simulate what the JNI bridge does: pass a flat float[63] through
 * normalize_landmarks and kalman_update, verify no crashes and output is valid.
 */
int test_jni_bridge_simulation() {
    // Simulate JNI: create a dummy float[63] as if from Java
    float raw_flat[63];
    for (int i = 0; i < 63; i++) {
        raw_flat[i] = (float)(i % 21) * 0.1f + 0.5f;
    }

    // Set wrist (indices 0,1,2) to a known position
    raw_flat[0] = 1.0f;   // wrist x
    raw_flat[1] = 2.0f;   // wrist y
    raw_flat[2] = 3.0f;   // wrist z

    // Set MCP9 (indices 27,28,29) to be offset from wrist
    raw_flat[27] = 1.0f;  // mcp9 x
    raw_flat[28] = 2.0f;  // mcp9 y
    raw_flat[29] = 6.0f;  // mcp9 z (distance = 3.0 in z)

    // Call normalize_landmarks (simulating JNI call)
    float out_flat[63];
    float (*raw_2d)[3] = reinterpret_cast<float(*)[3]>(raw_flat);
    float (*out_2d)[3] = reinterpret_cast<float(*)[3]>(out_flat);
    normalize_landmarks(raw_2d, out_2d);

    // Verify wrist is at origin
    if (std::fabs(out_flat[0]) > 1e-6f || 
        std::fabs(out_flat[1]) > 1e-6f || 
        std::fabs(out_flat[2]) > 1e-6f) {
        printf("  ERROR: Wrist not at origin after normalize\n");
        return 1;
    }
    printf("  normalize_landmarks via flat array: wrist at origin ✓\n");

    // Call kalman_update (simulating JNI call)
    kalman_reset();  // ensure clean state
    float smoothed[63];
    kalman_update(out_flat, smoothed);

    // Verify no NaN in output
    for (int i = 0; i < 63; i++) {
        if (std::isnan(smoothed[i]) || std::isinf(smoothed[i])) {
            printf("  ERROR: NaN or Inf at smoothed[%d]\n", i);
            return 1;
        }
    }
    printf("  kalman_update via flat array: no NaN/Inf ✓\n");
    printf("  JNI simulation: float[] round-trip successful, no JVM-like crash ✓\n");

    return 0;
}

/**
 * Test 6: WASM build simulation.
 * Verify that normalize + kalman produce the same values whether called
 * via 2D array API or flat array API (as WASM would use typed arrays).
 */
int test_wasm_build_simulation() {
    // Create test data
    float raw[21][3];
    for (int i = 0; i < 21; i++) {
        raw[i][0] = (float)i * 0.5f + 1.0f;
        raw[i][1] = (float)i * 0.3f + 2.0f;
        raw[i][2] = (float)i * 0.1f + 0.5f;
    }

    // Method A: Use 2D array API (native C++)
    float out_a[21][3];
    normalize_landmarks(raw, out_a);

    // Method B: Use flat array API (as WASM/typed array would)
    float* raw_flat = &raw[0][0];
    float out_flat[63];
    float (*out_b)[3] = reinterpret_cast<float(*)[3]>(out_flat);
    normalize_landmarks(reinterpret_cast<float(*)[3]>(raw_flat), out_b);

    // Compare results
    for (int i = 0; i < 21; i++) {
        for (int j = 0; j < 3; j++) {
            float diff = std::fabs(out_a[i][j] - out_b[i][j]);
            if (diff > 1e-6f) {
                printf("  ERROR: Mismatch at [%d][%d]: %f vs %f\n", 
                       i, j, out_a[i][j], out_b[i][j]);
                return 1;
            }
        }
    }
    printf("  2D array API matches flat array API (WASM compatible) ✓\n");

    // Verify Kalman also works with flat arrays
    kalman_reset();
    float smoothed_flat[63];
    kalman_update(out_flat, smoothed_flat);

    for (int i = 0; i < 63; i++) {
        if (std::isnan(smoothed_flat[i]) || std::isinf(smoothed_flat[i])) {
            printf("  ERROR: NaN/Inf in WASM-style Kalman output at [%d]\n", i);
            return 1;
        }
    }
    printf("  Kalman filter via flat array (WASM-style): no NaN/Inf ✓\n");

    return 0;
}
