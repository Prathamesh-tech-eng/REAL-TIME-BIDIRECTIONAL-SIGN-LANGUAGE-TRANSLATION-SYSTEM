/**
 * LanguageModel — Simple n-gram language model for post-processing
 * CTC-decoded gloss sequences.
 *
 * Uses bigram probabilities to correct common recognition errors
 * (e.g., swapping "NO" → "KNOW" based on context).
 *
 * In production this wraps KenLM; here we use a lightweight
 * frequency table suitable for the 32-token demo vocabulary.
 */

import { DecodedToken, ASL_VOCAB } from './CTCDecoder';

/**
 * Bigram probability table.  bigrams[A][B] = P(B | A).
 * Only stores non-zero entries; missing pairs get a small backoff probability.
 */
type BigramTable = Map<string, Map<string, number>>;

const BACKOFF_PROB = 0.01;

/** Build default bigram table from common ASL phrase patterns. */
function buildDefaultBigrams(): BigramTable {
  const table: BigramTable = new Map();

  const patterns: [string, string, number][] = [
    // Greetings / courtesy
    ['HELLO', 'FRIEND', 0.35],
    ['HELLO', 'NAME', 0.25],
    ['THANK-YOU', 'FRIEND', 0.30],
    ['PLEASE', 'HELP', 0.40],
    ['PLEASE', 'MORE', 0.20],
    ['SORRY', 'HELP', 0.20],

    // Questions / responses
    ['YES', 'THANK-YOU', 0.25],
    ['NO', 'SORRY', 0.25],
    ['WANT', 'EAT', 0.30],
    ['WANT', 'DRINK', 0.25],
    ['WANT', 'WATER', 0.20],
    ['WANT', 'HELP', 0.15],

    // Daily life
    ['GO', 'SCHOOL', 0.25],
    ['GO', 'WORK', 0.25],
    ['GO', 'HOUSE', 0.20],
    ['COME', 'HOUSE', 0.30],
    ['GOOD', 'FRIEND', 0.20],
    ['GOOD', 'FAMILY', 0.20],

    // Identity
    ['NAME', 'SIGN', 0.35],
    ['DEAF', 'SIGN', 0.30],
    ['DEAF', 'LANGUAGE', 0.25],
    ['SIGN', 'LANGUAGE', 0.50],
    ['HEARING', 'UNDERSTAND', 0.30],

    // Family
    ['MOTHER', 'FATHER', 0.25],
    ['FATHER', 'MOTHER', 0.25],
    ['LOVE', 'FAMILY', 0.35],
    ['LOVE', 'FRIEND', 0.25],

    // Needs
    ['MORE', 'WATER', 0.30],
    ['MORE', 'EAT', 0.25],
    ['STOP', 'PLEASE', 0.30],
    ['HELP', 'PLEASE', 0.25],
  ];

  for (const [a, b, p] of patterns) {
    if (!table.has(a)) table.set(a, new Map());
    table.get(a)!.set(b, p);
  }

  return table;
}

export class LanguageModel {
  private bigrams: BigramTable;

  constructor(bigrams?: BigramTable) {
    this.bigrams = bigrams || buildDefaultBigrams();
  }

  /**
   * Score a bigram P(next | prev).
   * Returns the stored probability or the backoff probability.
   */
  bigramScore(prev: string, next: string): number {
    return this.bigrams.get(prev)?.get(next) ?? BACKOFF_PROB;
  }

  /**
   * Re-rank decoded tokens using bigram context.
   *
   * For each position, if the CTC confidence is below a "confusion threshold"
   * (default 0.85), look at the previous token and check whether swapping
   * to a high-bigram-probability alternative improves the sequence score.
   *
   * This is a simple 1-best beam re-scoring, not a full beam search.
   */
  rescore(
    tokens: DecodedToken[],
    confusionThreshold = 0.85,
  ): DecodedToken[] {
    if (tokens.length <= 1) return tokens;

    const result = [tokens[0]];

    for (let i = 1; i < tokens.length; i++) {
      const prev = result[i - 1];
      const curr = tokens[i];

      if (curr.confidence >= confusionThreshold) {
        // High confidence — keep as-is
        result.push(curr);
        continue;
      }

      // Low confidence — see if a different gloss fits better
      const bestAlt = this.findBestAlternative(prev.gloss, curr);
      result.push(bestAlt || curr);
    }

    return result;
  }

  private findBestAlternative(
    prevGloss: string,
    curr: DecodedToken,
  ): DecodedToken | null {
    const prevMap = this.bigrams.get(prevGloss);
    if (!prevMap) return null;

    let bestGloss = curr.gloss;
    let bestScore = this.bigramScore(prevGloss, curr.gloss) * curr.confidence;

    for (const [candidate, bigramP] of prevMap) {
      const score = bigramP * 0.5; // weigh bigram lower than CTC confidence
      if (score > bestScore) {
        bestScore = score;
        bestGloss = candidate;
      }
    }

    if (bestGloss !== curr.gloss) {
      return { ...curr, gloss: bestGloss };
    }
    return null;
  }
}
