/**
 * HandLandmarker — Wrapper around MediaPipe Tasks Vision Hand Landmarker.
 *
 * On device this uses the GPU delegate for real-time 30 FPS detection.
 * For desktop tests, a mock implementation is provided that returns
 * known-good landmark data for test images.
 *
 * Reference: Step 4.3 of the execution document.
 */

import { Landmark } from '../jni/NativeModule';

/**
 * Configuration for the Hand Landmarker.
 */
export interface HandLandmarkerConfig {
  modelAssetPath: string;
  delegate: 'GPU' | 'CPU';
  numHands: number;
  minHandDetectionConfidence: number;
  minHandPresenceConfidence: number;
  minTrackingConfidence: number;
  runningMode: 'IMAGE' | 'VIDEO';
}

export const DEFAULT_CONFIG: HandLandmarkerConfig = {
  modelAssetPath: './hand_landmarker.task',
  delegate: 'GPU',
  numHands: 2,
  minHandDetectionConfidence: 0.5,
  minHandPresenceConfidence: 0.5,
  minTrackingConfidence: 0.5,
  runningMode: 'VIDEO',
};

/**
 * Detection result for a single frame.
 */
export interface HandDetectionResult {
  landmarks: Landmark[][];   // [numHands][21] landmarks
  handedness: string[];       // 'Left' | 'Right' per hand
  timestamp: number;
  detectionTimeMs: number;
}

/**
 * Abstract HandLandmarker interface.
 * Platform implementations extend this.
 */
export abstract class HandLandmarkerBase {
  protected config: HandLandmarkerConfig;

  constructor(config?: Partial<HandLandmarkerConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  abstract init(): Promise<void>;
  abstract detect(frameData: unknown, timestampMs: number): Promise<HandDetectionResult>;
  abstract close(): void;
}

/**
 * Ground-truth hand landmark data for known ASL poses.
 * Used in integration tests to verify detection accuracy.
 *
 * Each entry contains 21 (x, y, z) landmarks for a canonical pose.
 * Values are normalized to [0, 1] image coordinates.
 */
export const GROUND_TRUTH_LANDMARKS: Record<string, Landmark[]> = {
  // "HELLO" — open palm, fingers spread
  HELLO: [
    { x: 0.500, y: 0.850, z: 0.000 },  // 0: WRIST
    { x: 0.480, y: 0.750, z: -0.010 },  // 1: THUMB_CMC
    { x: 0.430, y: 0.650, z: -0.020 },  // 2: THUMB_MCP
    { x: 0.380, y: 0.570, z: -0.025 },  // 3: THUMB_IP
    { x: 0.340, y: 0.500, z: -0.030 },  // 4: THUMB_TIP
    { x: 0.470, y: 0.600, z: -0.015 },  // 5: INDEX_MCP
    { x: 0.460, y: 0.450, z: -0.020 },  // 6: INDEX_PIP
    { x: 0.455, y: 0.340, z: -0.025 },  // 7: INDEX_DIP
    { x: 0.450, y: 0.250, z: -0.030 },  // 8: INDEX_TIP
    { x: 0.510, y: 0.580, z: -0.012 },  // 9: MIDDLE_MCP
    { x: 0.515, y: 0.420, z: -0.018 },  // 10: MIDDLE_PIP
    { x: 0.518, y: 0.310, z: -0.022 },  // 11: MIDDLE_DIP
    { x: 0.520, y: 0.220, z: -0.026 },  // 12: MIDDLE_TIP
    { x: 0.555, y: 0.600, z: -0.010 },  // 13: RING_MCP
    { x: 0.565, y: 0.450, z: -0.016 },  // 14: RING_PIP
    { x: 0.570, y: 0.350, z: -0.020 },  // 15: RING_DIP
    { x: 0.575, y: 0.270, z: -0.024 },  // 16: RING_TIP
    { x: 0.600, y: 0.640, z: -0.008 },  // 17: PINKY_MCP
    { x: 0.615, y: 0.510, z: -0.014 },  // 18: PINKY_PIP
    { x: 0.625, y: 0.430, z: -0.018 },  // 19: PINKY_DIP
    { x: 0.635, y: 0.360, z: -0.022 },  // 20: PINKY_TIP
  ],

  // "THANK-YOU" — flat hand from chin forward
  'THANK-YOU': [
    { x: 0.500, y: 0.900, z: 0.000 },
    { x: 0.460, y: 0.800, z: -0.015 },
    { x: 0.420, y: 0.700, z: -0.025 },
    { x: 0.400, y: 0.620, z: -0.030 },
    { x: 0.380, y: 0.550, z: -0.035 },
    { x: 0.480, y: 0.650, z: -0.010 },
    { x: 0.475, y: 0.500, z: -0.015 },
    { x: 0.472, y: 0.400, z: -0.020 },
    { x: 0.470, y: 0.310, z: -0.025 },
    { x: 0.520, y: 0.640, z: -0.008 },
    { x: 0.525, y: 0.490, z: -0.012 },
    { x: 0.528, y: 0.390, z: -0.016 },
    { x: 0.530, y: 0.300, z: -0.020 },
    { x: 0.560, y: 0.660, z: -0.006 },
    { x: 0.568, y: 0.520, z: -0.010 },
    { x: 0.573, y: 0.420, z: -0.014 },
    { x: 0.578, y: 0.340, z: -0.018 },
    { x: 0.600, y: 0.690, z: -0.004 },
    { x: 0.612, y: 0.560, z: -0.008 },
    { x: 0.620, y: 0.470, z: -0.012 },
    { x: 0.628, y: 0.390, z: -0.016 },
  ],
};

/**
 * MockHandLandmarker — Returns ground-truth data for testing.
 * Maps pose names to known landmark arrays.
 */
export class MockHandLandmarker extends HandLandmarkerBase {
  private currentPose: string = 'HELLO';

  async init(): Promise<void> {
    // No-op for mock
  }

  setPose(poseName: string): void {
    this.currentPose = poseName;
  }

  async detect(_frameData: unknown, timestampMs: number): Promise<HandDetectionResult> {
    const t0 = performance.now();
    const landmarks = GROUND_TRUTH_LANDMARKS[this.currentPose];

    if (!landmarks) {
      return {
        landmarks: [],
        handedness: [],
        timestamp: timestampMs,
        detectionTimeMs: performance.now() - t0,
      };
    }

    // Small noise to simulate real detection
    const noisyLandmarks = landmarks.map((l) => ({
      x: l.x + (Math.random() - 0.5) * 0.002,
      y: l.y + (Math.random() - 0.5) * 0.002,
      z: l.z + (Math.random() - 0.5) * 0.001,
    }));

    return {
      landmarks: [noisyLandmarks],
      handedness: ['Right'],
      timestamp: timestampMs,
      detectionTimeMs: performance.now() - t0,
    };
  }

  close(): void {
    // No-op
  }
}
