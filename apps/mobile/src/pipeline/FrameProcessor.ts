/**
 * FrameProcessor — Camera frame processing loop manager.
 *
 * Implements the real-time pipeline described in Step 4.4:
 *   1. Camera frame → MediaPipe Hand Landmarker (GPU delegate)
 *   2. Normalize landmarks via JNI / NativeModule
 *   3. Kalman-filter landmarks
 *   4. Append to circular buffer (size 64 frames)
 *   5. Every 10 frames → TFLite TMS inference
 *   6. CTC decode → emit gloss token if confidence > 0.7
 *
 * The loop target is < 33 ms per frame to maintain 30 FPS.
 */

import { Landmark, normalizeLandmarks, KalmanSmoother } from '../jni/NativeModule';
import { greedyDecode, DecodedToken, ASL_VOCAB } from './CTCDecoder';
import { LanguageModel } from './LanguageModel';

/** Circular buffer entry = smoothed landmark array for one frame. */
export interface FrameEntry {
  landmarks: Landmark[];
  timestamp: number;
}

export interface FrameProcessorConfig {
  bufferSize: number;       // circular buffer capacity (default: 64)
  inferenceStride: number;  // run inference every N frames (default: 10)
  confidenceThreshold: number;
  numHands: number;
}

const DEFAULT_CONFIG: FrameProcessorConfig = {
  bufferSize: 64,
  inferenceStride: 10,
  confidenceThreshold: 0.7,
  numHands: 2,
};

export type InferenceCallback = (sequence: Landmark[][]) => Promise<number[][]>;
export type TokenCallback = (tokens: DecodedToken[]) => void;

export class FrameProcessor {
  private config: FrameProcessorConfig;
  private buffer: FrameEntry[];
  private bufferIndex: number;
  private frameCount: number;
  private smoother: KalmanSmoother;
  private languageModel: LanguageModel;
  private onInference: InferenceCallback | null;
  private onToken: TokenCallback | null;
  private lastTokens: DecodedToken[];
  private processingTimes: number[];
  private dropped: number;

  constructor(config?: Partial<FrameProcessorConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.buffer = new Array(this.config.bufferSize).fill(null);
    this.bufferIndex = 0;
    this.frameCount = 0;
    this.smoother = new KalmanSmoother(21);
    this.languageModel = new LanguageModel();
    this.onInference = null;
    this.onToken = null;
    this.lastTokens = [];
    this.processingTimes = [];
    this.dropped = 0;
  }

  /** Register the TFLite inference callback. */
  setInferenceCallback(cb: InferenceCallback): void {
    this.onInference = cb;
  }

  /** Register the token emission callback. */
  setTokenCallback(cb: TokenCallback): void {
    this.onToken = cb;
  }

  /**
   * Process a single camera frame.
   * @param rawLandmarks  21 landmarks from MediaPipe (may be null if no hand)
   * @returns Processing time in milliseconds
   */
  async processFrame(rawLandmarks: Landmark[] | null): Promise<number> {
    const t0 = performance.now();

    if (!rawLandmarks || rawLandmarks.length === 0) {
      return performance.now() - t0;
    }

    // Step 1: Normalize (C++ JNI bridge equivalent)
    const normalized = normalizeLandmarks(rawLandmarks);

    // Step 2: Kalman filter
    const smoothed = this.smoother.update(normalized);

    // Step 3: Append to circular buffer
    const idx = this.bufferIndex % this.config.bufferSize;
    this.buffer[idx] = {
      landmarks: smoothed,
      timestamp: performance.now(),
    };
    this.bufferIndex++;
    this.frameCount++;

    // Step 4: Run inference every `inferenceStride` frames
    if (this.frameCount % this.config.inferenceStride === 0 && this.onInference) {
      const sequence = this.getBufferedSequence();
      if (sequence.length >= this.config.inferenceStride) {
        try {
          const logits = await this.onInference(sequence);
          const tokens = greedyDecode(logits, ASL_VOCAB, this.config.confidenceThreshold);
          const rescored = this.languageModel.rescore(tokens);

          if (rescored.length > 0) {
            this.lastTokens = rescored;
            if (this.onToken) {
              this.onToken(rescored);
            }
          }
        } catch {
          // Inference error — skip this window
        }
      }
    }

    const elapsed = performance.now() - t0;
    this.processingTimes.push(elapsed);
    if (elapsed > 33) this.dropped++;

    return elapsed;
  }

  /** Get the current buffered sequence (ordered oldest → newest). */
  private getBufferedSequence(): Landmark[][] {
    const result: Landmark[][] = [];
    const count = Math.min(this.bufferIndex, this.config.bufferSize);
    const start = this.bufferIndex >= this.config.bufferSize
      ? this.bufferIndex % this.config.bufferSize
      : 0;

    for (let i = 0; i < count; i++) {
      const idx = (start + i) % this.config.bufferSize;
      if (this.buffer[idx]) {
        result.push(this.buffer[idx].landmarks);
      }
    }
    return result;
  }

  /** Get performance statistics. */
  getStats(): {
    totalFrames: number;
    droppedFrames: number;
    avgProcessingMs: number;
    maxProcessingMs: number;
    fps: number;
  } {
    const avg =
      this.processingTimes.length > 0
        ? this.processingTimes.reduce((a, b) => a + b, 0) / this.processingTimes.length
        : 0;
    const max =
      this.processingTimes.length > 0
        ? Math.max(...this.processingTimes)
        : 0;

    return {
      totalFrames: this.frameCount,
      droppedFrames: this.dropped,
      avgProcessingMs: avg,
      maxProcessingMs: max,
      fps: avg > 0 ? 1000 / avg : 0,
    };
  }

  /** Get last decoded tokens. */
  getLastTokens(): DecodedToken[] {
    return this.lastTokens;
  }

  /** Reset all state. */
  reset(): void {
    this.buffer.fill(null);
    this.bufferIndex = 0;
    this.frameCount = 0;
    this.smoother.reset();
    this.lastTokens = [];
    this.processingTimes = [];
    this.dropped = 0;
  }
}
