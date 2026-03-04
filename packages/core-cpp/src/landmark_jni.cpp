/**
 * landmark_jni.cpp
 * SignBridge Core - JNI Bridge for Android
 *
 * Exposes normalize_landmarks() and kalman_update() to Java via JNI.
 * On non-Android platforms, these are compiled as regular C++ functions
 * (JNI code is guarded by #ifdef __ANDROID__).
 */

#include "landmark_normalizer.h"
#include "kalman_filter.h"

#ifdef __ANDROID__

#include <jni.h>
#include <android/log.h>

#define LOG_TAG "SignBridgeCore"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

/**
 * JNI: Normalize landmarks.
 * Input:  jfloatArray of 63 floats (21 landmarks × 3 axes)
 * Output: jfloatArray of 63 floats (normalized)
 */
JNIEXPORT jfloatArray JNICALL
Java_com_signbridge_LandmarkProcessor_normalize(
    JNIEnv* env, jobject /* this */, jfloatArray rawArray) {

    jsize len = env->GetArrayLength(rawArray);
    if (len != 63) {
        LOGE("normalize: expected 63 floats, got %d", len);
        return nullptr;
    }

    jfloat* rawPtr = env->GetFloatArrayElements(rawArray, nullptr);
    if (!rawPtr) return nullptr;

    float out[21][3];
    float (*raw)[3] = reinterpret_cast<float(*)[3]>(rawPtr);

    normalize_landmarks(raw, out);

    env->ReleaseFloatArrayElements(rawArray, rawPtr, JNI_ABORT);

    jfloatArray result = env->NewFloatArray(63);
    if (result) {
        env->SetFloatArrayRegion(result, 0, 63, &out[0][0]);
    }
    return result;
}

/**
 * JNI: Kalman filter update.
 * Input:  jfloatArray of 63 floats (normalized landmarks)
 * Output: jfloatArray of 63 floats (smoothed)
 */
JNIEXPORT jfloatArray JNICALL
Java_com_signbridge_LandmarkProcessor_kalmanUpdate(
    JNIEnv* env, jobject /* this */, jfloatArray landmarksArray) {

    jsize len = env->GetArrayLength(landmarksArray);
    if (len != 63) {
        LOGE("kalmanUpdate: expected 63 floats, got %d", len);
        return nullptr;
    }

    jfloat* lmPtr = env->GetFloatArrayElements(landmarksArray, nullptr);
    if (!lmPtr) return nullptr;

    float smoothed[63];
    kalman_update(lmPtr, smoothed);

    env->ReleaseFloatArrayElements(landmarksArray, lmPtr, JNI_ABORT);

    jfloatArray result = env->NewFloatArray(63);
    if (result) {
        env->SetFloatArrayRegion(result, 0, 63, smoothed);
    }
    return result;
}

/**
 * JNI: Reset Kalman filter bank.
 */
JNIEXPORT void JNICALL
Java_com_signbridge_LandmarkProcessor_kalmanReset(
    JNIEnv* env, jobject /* this */) {
    kalman_reset();
    LOGI("Kalman filter bank reset");
}

} // extern "C"

#endif // __ANDROID__
