cmake_minimum_required(VERSION 3.22)
project(signbridge_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core source files
set(CORE_SOURCES
    src/landmark_normalizer.cpp
    src/kalman_filter.cpp
)

# Include directory
include_directories(include/)

if(ANDROID)
    # ===== Android NDK Build: Shared library with JNI =====
    add_library(signbridge_core SHARED
        ${CORE_SOURCES}
        src/landmark_jni.cpp
    )
    find_library(log-lib log)
    target_link_libraries(signbridge_core ${log-lib})

else()
    # ===== Desktop Build: Static library + test executable =====
    add_library(signbridge_core STATIC ${CORE_SOURCES})

    # Test executable (build separately, run manually)
    add_executable(run_tests
        tests/test_normalizer.cpp
        tests/test_kalman.cpp
        tests/test_main.cpp
    )
    target_link_libraries(run_tests signbridge_core)
endif()
