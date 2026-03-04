/**
 * CTCDecoder — Greedy CTC decoding for TMS model output.
 *
 * The TMS model produces a [T, V] logits tensor where T = time steps
 * and V = vocabulary size. This decoder:
 *   1. Takes argmax at each time step.
 *   2. Collapses repeated indices.
 *   3. Removes blank tokens (index 0 by convention).
 *   4. Maps remaining indices to gloss tokens.
 */

/** Standard ASL gloss vocabulary (subset for the mini-project demo). */
export const ASL_VOCAB: string[] = [
  '<blank>',   // 0 — CTC blank
  'HELLO',     // 1
  'THANK-YOU', // 2
  'YES',       // 3
  'NO',        // 4
  'PLEASE',    // 5
  'SORRY',     // 6
  'HELP',      // 7
  'LOVE',      // 8
  'FRIEND',    // 9
  'GOOD',      // 10
  'BAD',       // 11
  'EAT',       // 12
  'DRINK',     // 13
  'WATER',     // 14
  'MORE',      // 15
  'STOP',      // 16
  'GO',        // 17
  'COME',      // 18
  'WANT',      // 19
  'NAME',      // 20
  'DEAF',      // 21
  'HEARING',   // 22
  'SIGN',      // 23
  'LANGUAGE',  // 24
  'UNDERSTAND', // 25
  'HOUSE',     // 26
  'SCHOOL',    // 27
  'WORK',      // 28
  'FAMILY',    // 29
  'MOTHER',    // 30
  'FATHER',    // 31
];

export const BLANK_INDEX = 0;
export const CONFIDENCE_THRESHOLD = 0.7;

export interface DecodedToken {
  gloss: string;
  confidence: number;
  timeStep: number;
}

/**
 * Greedy CTC decode.
 * @param logits  2D array [T][V] of raw logits (pre-softmax is fine)
 * @param vocab   Vocabulary mapping index → gloss string
 * @param confidenceThreshold  Minimum softmax probability to emit a token
 * @returns Array of decoded gloss tokens with confidence and time step
 */
export function greedyDecode(
  logits: number[][],
  vocab: string[] = ASL_VOCAB,
  confidenceThreshold = CONFIDENCE_THRESHOLD,
): DecodedToken[] {
  const tokens: DecodedToken[] = [];
  let prevIndex = BLANK_INDEX;

  for (let t = 0; t < logits.length; t++) {
    const row = logits[t];

    // Softmax
    const maxVal = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sumExps);

    // Argmax
    let bestIdx = 0;
    let bestProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > bestProb) {
        bestIdx = i;
        bestProb = probs[i];
      }
    }

    // CTC: skip blanks and repeated
    if (bestIdx !== BLANK_INDEX && bestIdx !== prevIndex && bestProb >= confidenceThreshold) {
      tokens.push({
        gloss: vocab[bestIdx] || `<unk:${bestIdx}>`,
        confidence: bestProb,
        timeStep: t,
      });
    }

    prevIndex = bestIdx;
  }

  return tokens;
}

/**
 * Batch-decode: given a 3D tensor [B, T, V], decode each batch element.
 */
export function batchDecode(
  batchLogits: number[][][],
  vocab: string[] = ASL_VOCAB,
  confidenceThreshold = CONFIDENCE_THRESHOLD,
): DecodedToken[][] {
  return batchLogits.map((logits) =>
    greedyDecode(logits, vocab, confidenceThreshold),
  );
}
