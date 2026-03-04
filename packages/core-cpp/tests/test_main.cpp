/**
 * test_main.cpp
 * SignBridge Core - Test Harness (minimal, no external deps)
 */

#include <cstdio>
#include <cstdlib>

// Defined in test_normalizer.cpp and test_kalman.cpp
int test_normalizer_all_zeros();
int test_normalizer_known_pose();
int test_kalman_constant_input();
int test_kalman_step_change();
int test_jni_bridge_simulation();
int test_wasm_build_simulation();

int main() {
    int passed = 0;
    int failed = 0;
    int total = 6;

    printf("==============================================\n");
    printf("  SignBridge Core - C++ Unit Tests\n");
    printf("==============================================\n\n");

    // Test 1
    printf("Test 1: normalize_landmarks - all zeros input\n");
    if (test_normalizer_all_zeros() == 0) {
        printf("  PASSED\n\n"); passed++;
    } else {
        printf("  FAILED\n\n"); failed++;
    }

    // Test 2
    printf("Test 2: normalize_landmarks - known hand pose\n");
    if (test_normalizer_known_pose() == 0) {
        printf("  PASSED\n\n"); passed++;
    } else {
        printf("  FAILED\n\n"); failed++;
    }

    // Test 3
    printf("Test 3: KalmanFilter1D - constant input z=5.0\n");
    if (test_kalman_constant_input() == 0) {
        printf("  PASSED\n\n"); passed++;
    } else {
        printf("  FAILED\n\n"); failed++;
    }

    // Test 4
    printf("Test 4: KalmanFilter1D - step change 0 to 10\n");
    if (test_kalman_step_change() == 0) {
        printf("  PASSED\n\n"); passed++;
    } else {
        printf("  FAILED\n\n"); failed++;
    }

    // Test 5
    printf("Test 5: JNI bridge simulation (float[] round-trip)\n");
    if (test_jni_bridge_simulation() == 0) {
        printf("  PASSED\n\n"); passed++;
    } else {
        printf("  FAILED\n\n"); failed++;
    }

    // Test 6
    printf("Test 6: WASM build simulation (typed array verification)\n");
    if (test_wasm_build_simulation() == 0) {
        printf("  PASSED\n\n"); passed++;
    } else {
        printf("  FAILED\n\n"); failed++;
    }

    // Summary
    printf("==============================================\n");
    printf("  Results: %d/%d passed, %d failed\n", passed, total, failed);
    printf("==============================================\n");

    if (failed == 0) {
        printf("\nALL TESTS PASSED\n");
        return 0;
    } else {
        printf("\nSOME TESTS FAILED\n");
        return 1;
    }
}
