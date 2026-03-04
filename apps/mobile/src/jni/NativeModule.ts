/**
 * LandmarkNormalizer — Pure TypeScript port of the C++ normalize_landmarks.
 *
 * Centres landmarks on the wrist (index 0), scales by the maximum distance
 * from the wrist, and returns values in roughly [-1, 1] (may extend to
 * [-3, 3] with filtering). Used both in the mobile pipeline and in
 * desktop-side integration tests.
 */
export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/**
 * Normalize 21 hand landmarks:
 *   1. Translate so wrist (index 0) is at origin.
 *   2. Scale by max Euclidean distance from wrist.
 */
export function normalizeLandmarks(raw: Landmark[]): Landmark[] {
  if (raw.length === 0) return [];

  const wrist = raw[0];

  // Translate
  const translated = raw.map((l) => ({
    x: l.x - wrist.x,
    y: l.y - wrist.y,
    z: l.z - wrist.z,
  }));

  // Scale factor = max distance from origin (wrist)
  let maxDist = 0;
  for (const l of translated) {
    const d = Math.sqrt(l.x * l.x + l.y * l.y + l.z * l.z);
    if (d > maxDist) maxDist = d;
  }

  if (maxDist < 1e-8) return translated; // degenerate hand

  return translated.map((l) => ({
    x: l.x / maxDist,
    y: l.y / maxDist,
    z: l.z / maxDist,
  }));
}

/**
 * Simple 1D Kalman filter state for one coordinate dimension.
 */
interface KalmanState {
  x: number;  // estimate
  p: number;  // error covariance
  q: number;  // process noise
  r: number;  // measurement noise
}

function createKalmanState(q = 0.01, r = 0.1): KalmanState {
  return { x: 0, p: 1, q, r };
}

function kalmanUpdate(state: KalmanState, measurement: number): number {
  // Predict
  state.p += state.q;
  // Update
  const k = state.p / (state.p + state.r);
  state.x += k * (measurement - state.x);
  state.p *= 1 - k;
  return state.x;
}

/**
 * KalmanSmoother — maintains a bank of Kalman filters, one per
 * coordinate per landmark (21 landmarks × 3 axes = 63 filters).
 */
export class KalmanSmoother {
  private states: KalmanState[][];

  constructor(numLandmarks = 21, processNoise = 0.01, measurementNoise = 0.1) {
    this.states = Array.from({ length: numLandmarks }, () => [
      createKalmanState(processNoise, measurementNoise),
      createKalmanState(processNoise, measurementNoise),
      createKalmanState(processNoise, measurementNoise),
    ]);
  }

  /**
   * Smooth a set of normalized landmarks through the Kalman filter bank.
   * Returns filtered landmarks.
   */
  update(landmarks: Landmark[]): Landmark[] {
    return landmarks.map((l, i) => {
      const s = this.states[i] || this.states[0];
      return {
        x: kalmanUpdate(s[0], l.x),
        y: kalmanUpdate(s[1], l.y),
        z: kalmanUpdate(s[2], l.z),
      };
    });
  }

  /** Reset all filter states to zero. */
  reset(): void {
    for (const bank of this.states) {
      for (const s of bank) {
        s.x = 0;
        s.p = 1;
      }
    }
  }
}
