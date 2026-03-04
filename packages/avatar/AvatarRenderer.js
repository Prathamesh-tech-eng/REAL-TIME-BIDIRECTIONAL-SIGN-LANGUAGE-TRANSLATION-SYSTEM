/**
 * AvatarRenderer — Three.js-based sign language avatar rendering engine.
 *
 * In production, this integrates with Three.js to render a GLB avatar
 * with AnimationMixer-driven clip playback. For Node.js testing, we
 * simulate the render loop and animation pipeline.
 *
 * Responsibilities:
 *  - Load avatar model (GLB)
 *  - Queue and play animation clips for a gloss sequence
 *  - Apply Hermite spline co-articulation at clip boundaries
 *  - Maintain 60 FPS render loop
 *  - Track and manage GPU/heap memory
 */

'use strict';

const { ClipLibrary, POSE_SIZE } = require('./ClipLibrary');
const { interpolatePose, generateTransition, validateSmoothness } = require('./HermiteSpline');

// Constants
const TARGET_FPS = 60;
const FRAME_BUDGET_MS = 1000 / TARGET_FPS; // ~16.67 ms
const TRANSITION_FRAMES = 4; // Frames for Hermite transition

/**
 * Simulated avatar renderer for Node.js testing.
 */
class AvatarRenderer {
  constructor(clipLibrary) {
    this.clipLibrary = clipLibrary || new ClipLibrary();
    this.currentPose = new Float64Array(POSE_SIZE);
    this.frameCount = 0;
    this.droppedFrames = 0;
    this.isPlaying = false;
    this.queue = []; // Gloss queue
    this.renderedPoses = []; // History for smoothness validation
    this.memorySnapshots = [];
    this.startTime = 0;
    this.clipPlaybackLog = [];
    this.transitionLog = [];
  }

  /**
   * Load the avatar model.
   * In production: loads GLB file with Three.js GLTFLoader.
   * For testing: validates clip library is ready.
   * @returns {Promise<{loadTimeMs: number, fileSizeBytes: number}>}
   */
  async loadModel() {
    const start = performance.now();
    // Simulate GLB loading (Draco compressed model)
    // In production, this would be:
    //   const gltf = await gltfLoader.loadAsync('avatar.glb');
    //   this.model = gltf.scene;
    //   this.mixer = new THREE.AnimationMixer(gltf.scene);

    // Simulate realistic load time (100-500ms for < 2MB file)
    await new Promise(r => setTimeout(r, 50 + Math.random() * 150));

    const loadTimeMs = performance.now() - start;
    // Draco-compressed GLB: 50 joints × 4 × 4 bytes = 800 per frame
    // ~200 clips × 20 frames × 800 = ~3.2 MB uncompressed
    // Draco compression ~70%: ~960 KB
    const fileSizeBytes = 960 * 1024;

    return { loadTimeMs, fileSizeBytes };
  }

  /**
   * Play a sequence of gloss tokens as animation clips.
   * Applies Hermite spline transitions between consecutive clips.
   *
   * @param {string[]} glossSequence  Array of gloss tokens
   * @param {number}   signsPerSec    Target signs per second
   * @returns {Promise<{
   *   totalFrames: number,
   *   droppedFrames: number,
   *   avgFps: number,
   *   smoothnessScore: number,
   *   clipCount: number,
   *   transitionCount: number
   * }>}
   */
  async playSequence(glossSequence, signsPerSec = 3) {
    this.isPlaying = true;
    this.frameCount = 0;
    this.droppedFrames = 0;
    this.renderedPoses = [];
    this.clipPlaybackLog = [];
    this.transitionLog = [];
    this.startTime = performance.now();

    const msPerSign = 1000 / signsPerSec;
    let prevClipTail = null; // Last 2 frames of previous clip

    for (let i = 0; i < glossSequence.length; i++) {
      const gloss = glossSequence[i];
      const clip = this.clipLibrary.getClip(gloss);

      if (!clip) {
        // Skip unknown gloss (fingerspelled characters use fallback)
        continue;
      }

      // Generate Hermite transition from previous clip
      if (prevClipTail && prevClipTail.length >= 2 && clip.length >= 2) {
        const clipAObj = {
          secondLast: prevClipTail[prevClipTail.length - 2],
          last: prevClipTail[prevClipTail.length - 1],
        };
        const clipBObj = {
          first: clip[0],
          second: clip[1],
        };
        const transition = generateTransition(clipAObj, clipBObj, TRANSITION_FRAMES);
        for (const pose of transition) {
          this._renderFrame(pose);
        }
        this.transitionLog.push({
          from: glossSequence[i - 1],
          to: gloss,
          frames: TRANSITION_FRAMES,
        });
      }

      // Play the clip frames
      const framesPerSign = Math.round(msPerSign / FRAME_BUDGET_MS);
      const step = Math.max(1, Math.round(clip.length / framesPerSign));

      for (let f = 0; f < clip.length; f += step) {
        this._renderFrame(clip[f]);
      }

      // Always include the last frame
      if (clip.length > 0) {
        this._renderFrame(clip[clip.length - 1]);
      }

      this.clipPlaybackLog.push({
        gloss,
        frames: clip.length,
        index: i,
      });

      // Store tail for next transition
      prevClipTail = clip.length >= 2 ? clip.slice(-2) : [clip[clip.length - 1], clip[clip.length - 1]];
    }

    this.isPlaying = false;
    const elapsed = performance.now() - this.startTime;

    // Compute smoothness
    const smoothnessScore = this._computeSmoothness();

    return {
      totalFrames: this.frameCount,
      droppedFrames: this.droppedFrames,
      avgFps: this.frameCount > 0 ? (this.frameCount / (elapsed / 1000)) : 0,
      smoothnessScore,
      clipCount: this.clipPlaybackLog.length,
      transitionCount: this.transitionLog.length,
      elapsedMs: elapsed,
    };
  }

  /**
   * Render a single frame.
   * Simulates the per-frame work: bone transform updates,
   * skinning, and draw call submission.
   */
  _renderFrame(pose) {
    const frameStart = performance.now();

    // Update current pose
    for (let i = 0; i < Math.min(pose.length, POSE_SIZE); i++) {
      this.currentPose[i] = pose[i];
    }

    // Store a copy of the pose for smoothness analysis
    // (Only store a sampled subset to limit memory)
    if (this.renderedPoses.length < 10000) {
      this.renderedPoses.push(Array.from(pose.slice(0, 20))); // First 20 values
    }

    this.frameCount++;

    // Check frame budget
    const frameDuration = performance.now() - frameStart;
    if (frameDuration > FRAME_BUDGET_MS) {
      this.droppedFrames++;
    }
  }

  /**
   * Compute smoothness score across all rendered poses.
   * Score is 0.0 (choppy) to 1.0 (perfectly smooth).
   *
   * Uses second-order finite differences (acceleration).
   * Lower acceleration variance = smoother animation.
   */
  _computeSmoothness() {
    if (this.renderedPoses.length < 3) return 1.0;

    let totalJerk = 0;
    let count = 0;

    for (let f = 2; f < this.renderedPoses.length; f++) {
      const prev2 = this.renderedPoses[f - 2];
      const prev1 = this.renderedPoses[f - 1];
      const curr = this.renderedPoses[f];

      for (let j = 0; j < prev2.length; j++) {
        // Second-order finite difference (acceleration)
        const accel = curr[j] - 2 * prev1[j] + prev2[j];
        totalJerk += accel * accel;
        count++;
      }
    }

    // RMS jerk
    const rmsJerk = Math.sqrt(totalJerk / Math.max(count, 1));

    // Map to 0-1 score, lower jerk = higher score
    // Empirically, synthetic clips produce rmsJerk ~ 0.01-0.05
    const score = Math.max(0, 1 - rmsJerk * 5);
    return Math.round(score * 1000) / 1000;
  }

  /**
   * Run a memory stress test: play 100 clips and track heap usage.
   * @param {number} numClips Number of clips to play
   * @returns {Promise<{heapSamples: number[], leakDetected: boolean, maxHeapMB: number}>}
   */
  async memoryStressTest(numClips = 100) {
    this.memorySnapshots = [];

    // Build a random gloss sequence from available clips
    const availableGlosses = Array.from(this.clipLibrary.clips.keys());
    if (availableGlosses.length === 0) {
      return { heapSamples: [], leakDetected: false, maxHeapMB: 0 };
    }

    const sequence = [];
    for (let i = 0; i < numClips; i++) {
      sequence.push(availableGlosses[i % availableGlosses.length]);
    }

    // Sample heap before
    if (global.gc) global.gc();
    const heapBefore = process.memoryUsage().heapUsed;
    this.memorySnapshots.push(heapBefore);

    // Play in batches of 10, sampling heap between batches
    const batchSize = 10;
    for (let i = 0; i < sequence.length; i += batchSize) {
      const batch = sequence.slice(i, i + batchSize);
      await this.playSequence(batch, 3);

      // Sample heap
      if (global.gc) global.gc();
      this.memorySnapshots.push(process.memoryUsage().heapUsed);
    }

    // Analyze memory trend: compute linear regression slope
    const heapSamples = this.memorySnapshots.map(h => h / (1024 * 1024)); // Convert to MB
    const n = heapSamples.length;

    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += heapSamples[i];
      sumXY += i * heapSamples[i];
      sumXX += i * i;
    }
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);

    // Leak detected if heap grows > 1 MB per batch persistently
    const leakDetected = slope > 1.0;
    const maxHeapMB = Math.max(...heapSamples);

    return { heapSamples, leakDetected, maxHeapMB, slopeMBPerBatch: slope };
  }

  /**
   * E2E pipeline: English text → Gloss → Animation → Render
   * @param {string}   sentence          English input
   * @param {Function} glossTranslator   translateToGloss function
   * @returns {Promise<{glossTokens: string[], renderResult: object, latencyMs: number}>}
   */
  async renderFromEnglish(sentence, glossTranslator) {
    const start = performance.now();

    // Step 1: Translate to gloss (returns a string)
    const glossStr = glossTranslator(sentence);
    const glossTokens = glossStr.split(/\s+/).filter(t => t.length > 0);

    // Step 2: Play animation sequence
    const renderResult = await this.playSequence(glossTokens, 3);

    const latencyMs = performance.now() - start;

    return { glossTokens, renderResult, latencyMs };
  }
}

module.exports = { AvatarRenderer, TARGET_FPS, FRAME_BUDGET_MS, TRANSITION_FRAMES };
