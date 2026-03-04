/**
 * ClipLibrary — Animation clip management for the sign avatar.
 *
 * Each gloss has a corresponding animation clip stored as a set
 * of keyframes. In production, these come from GLB files.
 * For testing, we generate synthetic animation data.
 *
 * Format: Each clip is an array of keyframe poses.
 * Each pose is a flat array of joint values (e.g., 50 joints × 4 quaternion = 200 values).
 */

'use strict';

const NUM_JOINTS = 50;           // Skeleton joint count
const VALUES_PER_JOINT = 4;      // Quaternion (x, y, z, w)
const POSE_SIZE = NUM_JOINTS * VALUES_PER_JOINT; // 200

/**
 * Generate a synthetic animation clip for a given gloss.
 * Each clip has 15-30 keyframes at 30 FPS (0.5 - 1.0 seconds).
 *
 * The clip is deterministic given the gloss name (seeded pseudo-random).
 *
 * @param {string} gloss       Gloss name
 * @param {number} numFrames   Number of keyframes
 * @returns {number[][]}       Array of pose vectors
 */
function generateSyntheticClip(gloss, numFrames = 20) {
  // Seed from gloss name for determinism
  let seed = 0;
  for (let i = 0; i < gloss.length; i++) {
    seed = ((seed << 5) - seed + gloss.charCodeAt(i)) | 0;
  }

  function seededRandom() {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  }

  // Generate base pose
  const basePose = new Array(POSE_SIZE);
  for (let i = 0; i < POSE_SIZE; i++) {
    basePose[i] = seededRandom() * 2 - 1; // [-1, 1]
  }

  // Generate keyframes with smooth variation
  const clip = [];
  for (let f = 0; f < numFrames; f++) {
    const pose = new Array(POSE_SIZE);
    const phase = (f / numFrames) * Math.PI * 2;
    for (let i = 0; i < POSE_SIZE; i++) {
      // Smooth sinusoidal variation around base
      const freq = 1 + (i % 7) * 0.3;
      pose[i] = basePose[i] + 0.2 * Math.sin(phase * freq + i * 0.1);
    }
    clip.push(pose);
  }

  return clip;
}

/**
 * ClipLibrary — stores and retrieves animation clips.
 */
class ClipLibrary {
  constructor() {
    this.clips = new Map();
    this.metadata = new Map();
  }

  /**
   * Load a clip into the library.
   * @param {string} gloss     Gloss name (e.g., 'HELLO')
   * @param {number[][]} clip  Keyframe array
   */
  loadClip(gloss, clip) {
    this.clips.set(gloss, clip);
    this.metadata.set(gloss, {
      name: gloss,
      numFrames: clip.length,
      poseSize: clip[0]?.length || 0,
      durationMs: (clip.length / 30) * 1000,
      loadedAt: Date.now(),
    });
  }

  /**
   * Generate and load synthetic clips for a vocabulary.
   * @param {string[]} vocab   Array of gloss strings
   */
  loadSyntheticVocab(vocab) {
    for (const gloss of vocab) {
      const numFrames = 15 + Math.floor(Math.random() * 16); // 15-30
      const clip = generateSyntheticClip(gloss, numFrames);
      this.loadClip(gloss, clip);
    }
  }

  /**
   * Get a clip by gloss name.
   * @param {string} gloss
   * @returns {number[][] | null}
   */
  getClip(gloss) {
    return this.clips.get(gloss) || null;
  }

  /**
   * Get the last N frames of a clip (for transition computation).
   */
  getClipTail(gloss, n = 2) {
    const clip = this.clips.get(gloss);
    if (!clip || clip.length < n) return null;
    return clip.slice(-n);
  }

  /**
   * Get the first N frames of a clip (for transition computation).
   */
  getClipHead(gloss, n = 2) {
    const clip = this.clips.get(gloss);
    if (!clip || clip.length < n) return null;
    return clip.slice(0, n);
  }

  /**
   * Get the estimated GLB file size in bytes.
   * Each float32 = 4 bytes. Plus header overhead.
   */
  estimateFileSize(gloss) {
    const clip = this.clips.get(gloss);
    if (!clip) return 0;
    return clip.length * clip[0].length * 4 + 1024; // data + overhead
  }

  /**
   * Get total memory usage estimate for all loaded clips.
   * @returns {number} Bytes
   */
  totalMemoryUsage() {
    let total = 0;
    for (const clip of this.clips.values()) {
      total += clip.length * (clip[0]?.length || 0) * 8; // float64 in JS
    }
    return total;
  }

  /** Get number of loaded clips. */
  get size() {
    return this.clips.size;
  }

  /** Check if a gloss clip exists. */
  has(gloss) {
    return this.clips.has(gloss);
  }
}

module.exports = {
  ClipLibrary,
  generateSyntheticClip,
  NUM_JOINTS,
  VALUES_PER_JOINT,
  POSE_SIZE,
};
