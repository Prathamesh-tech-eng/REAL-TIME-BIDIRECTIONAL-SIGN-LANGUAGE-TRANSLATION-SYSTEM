/**
 * HermiteSpline — Cubic Hermite spline co-articulation for smooth
 * transitions between sign animation clips.
 *
 * Formula (Step 5.4 of execution document):
 *   p(t) = (2t³ - 3t² + 1)·pA + (t³ - 2t² + t)·tA +
 *          (-2t³ + 3t²)·pB + (t³ - t²)·tB
 *
 *   where t ∈ [0, 1], duration = 80ms (2-3 frames at 30 FPS)
 *
 * pA = ending pose of clip A
 * pB = starting pose of clip B
 * tA = tangent at end of clip A (computed from adjacent keyframes)
 * tB = tangent at start of clip B (computed from adjacent keyframes)
 */

'use strict';

/**
 * Evaluate cubic Hermite spline at parameter t.
 *
 * @param {number} t      Parameter ∈ [0, 1]
 * @param {number} pA     Start value
 * @param {number} tA     Start tangent
 * @param {number} pB     End value
 * @param {number} tB     End tangent
 * @returns {number}       Interpolated value
 */
function hermite(t, pA, tA, pB, tB) {
  const t2 = t * t;
  const t3 = t2 * t;

  const h00 = 2 * t3 - 3 * t2 + 1;   // (2t³ - 3t² + 1)
  const h10 = t3 - 2 * t2 + t;        // (t³ - 2t² + t)
  const h01 = -2 * t3 + 3 * t2;       // (-2t³ + 3t²)
  const h11 = t3 - t2;                // (t³ - t²)

  return h00 * pA + h10 * tA + h01 * pB + h11 * tB;
}

/**
 * Interpolate between two pose vectors using cubic Hermite.
 *
 * @param {number} t           Parameter ∈ [0, 1]
 * @param {number[]} poseA     End pose of clip A (flat array of joint values)
 * @param {number[]} tangentA  End tangent of clip A
 * @param {number[]} poseB     Start pose of clip B
 * @param {number[]} tangentB  Start tangent of clip B
 * @returns {number[]}         Interpolated pose
 */
function interpolatePose(t, poseA, tangentA, poseB, tangentB) {
  const result = new Array(poseA.length);
  for (let i = 0; i < poseA.length; i++) {
    result[i] = hermite(t, poseA[i], tangentA[i], poseB[i], tangentB[i]);
  }
  return result;
}

/**
 * Compute tangent vector from adjacent keyframes.
 * Uses Catmull-Rom tangent formula: t = (p[i+1] - p[i-1]) / 2
 *
 * @param {number[]} prev     Previous keyframe pose
 * @param {number[]} next     Next keyframe pose
 * @returns {number[]}        Tangent vector
 */
function computeTangent(prev, next) {
  const tangent = new Array(prev.length);
  for (let i = 0; i < prev.length; i++) {
    tangent[i] = (next[i] - prev[i]) / 2;
  }
  return tangent;
}

/**
 * Generate transition frames between two clips using cubic Hermite.
 *
 * @param {object} clipA                Last 2 keyframes of clip A
 * @param {number[]} clipA.secondLast   Second-to-last keyframe pose
 * @param {number[]} clipA.last         Last keyframe pose
 * @param {object} clipB                First 2 keyframes of clip B
 * @param {number[]} clipB.first        First keyframe pose
 * @param {number[]} clipB.second       Second keyframe pose
 * @param {number} numFrames            Number of transition frames (default: 3)
 * @returns {number[][]}                Array of interpolated pose frames
 */
function generateTransition(clipA, clipB, numFrames = 3) {
  const tangentA = computeTangent(clipA.secondLast, clipA.last);
  const tangentB = computeTangent(clipB.first, clipB.second);

  const frames = [];
  for (let i = 0; i < numFrames; i++) {
    const t = (i + 1) / (numFrames + 1); // Evenly spaced, excluding endpoints
    frames.push(interpolatePose(t, clipA.last, tangentA, clipB.first, tangentB));
  }

  return frames;
}

/**
 * Validate that a transition is smooth (no discontinuities).
 * Checks that the maximum per-joint velocity doesn't spike.
 *
 * @param {number[][]} sequence  Full sequence: [... clipA frames, transition, clipB frames ...]
 * @param {number} maxVelocity   Maximum allowed per-joint velocity per frame
 * @returns {{ smooth: boolean, maxVel: number, spikeIndex: number }}
 */
function validateSmoothness(sequence, maxVelocity = 2.0) {
  let maxVel = 0;
  let spikeIndex = -1;

  for (let i = 1; i < sequence.length; i++) {
    for (let j = 0; j < sequence[i].length; j++) {
      const vel = Math.abs(sequence[i][j] - sequence[i - 1][j]);
      if (vel > maxVel) {
        maxVel = vel;
        if (vel > maxVelocity) {
          spikeIndex = i;
        }
      }
    }
  }

  return {
    smooth: spikeIndex === -1,
    maxVel,
    spikeIndex,
  };
}

module.exports = {
  hermite,
  interpolatePose,
  computeTangent,
  generateTransition,
  validateSmoothness,
};
