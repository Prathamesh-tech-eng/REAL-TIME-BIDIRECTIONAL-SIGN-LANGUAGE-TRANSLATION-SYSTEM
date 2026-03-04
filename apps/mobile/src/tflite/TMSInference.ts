/**
 * TMSInference — TFLite Temporal Multi-Scale (TMS) model inference.
 *
 * On device: loads the int8-quantized TFLite model and runs inference
 * via the TFLite React Native delegate.
 *
 * For desktop tests: provides a mock inference function that returns
 * realistic logits for known sign sequences.
 */

import { Landmark } from '../jni/NativeModule';
import { ASL_VOCAB } from '../pipeline/CTCDecoder';

/** Model metadata. */
export const MODEL_CONFIG = {
  inputShape: [1, 64, 63],     // [batch, frames, landmarks*3]
  outputShape: [1, 16, 32],    // [batch, time_steps, vocab_size]
  quantization: 'int8',
  maxFileSize: 1.0,             // MB
  targetLatencyMs: 45,
};

/**
 * Flatten a sequence of landmark arrays into a 1D feature vector.
 * Each frame has 21 landmarks × 3 coords = 63 values.
 * Sequence is padded/truncated to exactly `seqLen` frames.
 */
export function flattenSequence(
  sequence: Landmark[][],
  seqLen: number = 64,
): number[] {
  const result: number[] = [];

  for (let i = 0; i < seqLen; i++) {
    if (i < sequence.length) {
      const frame = sequence[i];
      for (const lm of frame) {
        result.push(lm.x, lm.y, lm.z);
      }
      // Pad frame if it has fewer than 21 landmarks
      for (let j = frame.length; j < 21; j++) {
        result.push(0, 0, 0);
      }
    } else {
      // Pad with zeros for missing frames
      for (let j = 0; j < 63; j++) {
        result.push(0);
      }
    }
  }

  return result;
}

/**
 * Abstract TFLite inference interface.
 */
export abstract class TMSInferenceBase {
  abstract init(modelPath: string): Promise<void>;
  abstract runInference(sequence: Landmark[][]): Promise<number[][]>;
  abstract close(): void;
}

/**
 * MockTMSInference — Returns realistic logits for testing.
 *
 * Maps known input patterns to specific sign outputs.
 * Used to verify the full pipeline (MediaPipe → normalize → Kalman →
 * TFLite → CTC decode → language model) without a real model.
 */
export class MockTMSInference extends TMSInferenceBase {
  private vocabSize: number;
  private timeSteps: number;
  private signMapping: Map<string, number>;
  private currentSign: string;
  private latencyMs: number;

  constructor(latencyMs: number = 40) {
    super();
    this.vocabSize = ASL_VOCAB.length;
    this.timeSteps = 16;
    this.currentSign = 'HELLO';
    this.latencyMs = latencyMs;

    // Map sign names to vocab indices
    this.signMapping = new Map();
    for (let i = 0; i < ASL_VOCAB.length; i++) {
      this.signMapping.set(ASL_VOCAB[i], i);
    }
  }

  async init(_modelPath: string): Promise<void> {
    // No-op for mock
  }

  /** Set which sign the mock model should "detect". */
  setCurrentSign(sign: string): void {
    this.currentSign = sign;
  }

  /**
   * Simulate inference by returning logits where the correct class
   * has a high score at the middle time steps, with realistic noise.
   */
  async runInference(_sequence: Landmark[][]): Promise<number[][]> {
    // Simulate inference latency
    const jitter = (Math.random() - 0.5) * 4; // ±2ms jitter
    await new Promise((resolve) =>
      setTimeout(resolve, Math.max(1, this.latencyMs + jitter)),
    );

    const targetIdx = this.signMapping.get(this.currentSign) || 1;
    const logits: number[][] = [];

    for (let t = 0; t < this.timeSteps; t++) {
      const row = new Array(this.vocabSize).fill(0);

      // Add noise to all classes
      for (let v = 0; v < this.vocabSize; v++) {
        row[v] = (Math.random() - 0.5) * 2; // noise in [-1, 1]
      }

      // Spike the correct class in the middle time steps
      if (t >= 4 && t <= 11) {
        row[targetIdx] = 5.0 + Math.random() * 2; // strong signal
        row[0] = -2.0; // suppress blank
      } else {
        row[0] = 3.0; // blank at edges
      }

      logits.push(row);
    }

    return logits;
  }

  close(): void {
    // No-op
  }
}

/**
 * Measure inference latency statistics over multiple runs.
 */
export async function benchmarkInference(
  model: TMSInferenceBase,
  sequence: Landmark[][],
  runs: number = 100,
): Promise<{
  mean: number;
  stdDev: number;
  min: number;
  max: number;
  heapUsedBefore: number;
  heapUsedAfter: number;
}> {
  const latencies: number[] = [];

  // Warm up
  for (let i = 0; i < 5; i++) {
    await model.runInference(sequence);
  }

  const heapBefore = typeof process !== 'undefined' && process.memoryUsage
    ? process.memoryUsage().heapUsed
    : 0;

  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await model.runInference(sequence);
    latencies.push(performance.now() - t0);
  }

  const heapAfter = typeof process !== 'undefined' && process.memoryUsage
    ? process.memoryUsage().heapUsed
    : 0;

  const mean = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const variance =
    latencies.reduce((sum, l) => sum + (l - mean) ** 2, 0) / latencies.length;
  const stdDev = Math.sqrt(variance);

  return {
    mean,
    stdDev,
    min: Math.min(...latencies),
    max: Math.max(...latencies),
    heapUsedBefore: heapBefore,
    heapUsedAfter: heapAfter,
  };
}
