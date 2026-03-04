/**
 * SignToTextPipeline — High-level orchestrator for the sign-to-text
 * real-time translation pipeline.
 *
 * Wires together:
 *   • HandLandmarker (MediaPipe)
 *   • FrameProcessor (normalize → Kalman → buffer → inference)
 *   • DataChannel (WebRTC text relay)
 *
 * Usage:
 *   const pipeline = new SignToTextPipeline(signalingUrl, roomId);
 *   await pipeline.init();
 *   pipeline.start();          // starts camera + frame loop
 *   pipeline.onCaption(text => display(text));
 *   pipeline.stop();
 */

import { FrameProcessor, FrameProcessorConfig } from './FrameProcessor';
import { DecodedToken } from './CTCDecoder';
import { Landmark } from '../jni/NativeModule';

export interface PipelineConfig extends Partial<FrameProcessorConfig> {
  signalingUrl: string;
  roomId: string;
  modelPath: string;
}

export type CaptionHandler = (text: string, tokens: DecodedToken[]) => void;

export class SignToTextPipeline {
  private frameProcessor: FrameProcessor;
  private config: PipelineConfig;
  private running: boolean;
  private captionHandlers: CaptionHandler[];
  private inferenceLatencies: number[];

  constructor(config: PipelineConfig) {
    this.config = config;
    this.frameProcessor = new FrameProcessor(config);
    this.running = false;
    this.captionHandlers = [];
    this.inferenceLatencies = [];

    // Wire token callback → caption handlers
    this.frameProcessor.setTokenCallback((tokens) => {
      const text = tokens.map((t) => t.gloss).join(' ');
      for (const handler of this.captionHandlers) {
        handler(text, tokens);
      }
    });
  }

  /** Register a caption listener. */
  onCaption(handler: CaptionHandler): void {
    this.captionHandlers.push(handler);
  }

  /**
   * Set the TFLite inference function.
   * This is injected from the platform-specific TFLite module.
   */
  setInferenceFunction(fn: (sequence: Landmark[][]) => Promise<number[][]>): void {
    this.frameProcessor.setInferenceCallback(fn);
  }

  /**
   * Feed a single frame of landmarks (from MediaPipe or a mock source).
   * Returns processing time in ms.
   */
  async feedFrame(landmarks: Landmark[] | null): Promise<number> {
    if (!this.running) return 0;
    const t0 = performance.now();
    await this.frameProcessor.processFrame(landmarks);
    const elapsed = performance.now() - t0;
    this.inferenceLatencies.push(elapsed);
    return elapsed;
  }

  /** Start the pipeline. */
  start(): void {
    this.running = true;
    this.frameProcessor.reset();
    this.inferenceLatencies = [];
  }

  /** Stop the pipeline. */
  stop(): void {
    this.running = false;
  }

  /** Check if pipeline is running. */
  isRunning(): boolean {
    return this.running;
  }

  /** Get pipeline performance stats. */
  getStats() {
    return {
      ...this.frameProcessor.getStats(),
      inferenceCount: this.inferenceLatencies.length,
    };
  }

  /** Get last decoded tokens. */
  getLastTokens(): DecodedToken[] {
    return this.frameProcessor.getLastTokens();
  }
}
