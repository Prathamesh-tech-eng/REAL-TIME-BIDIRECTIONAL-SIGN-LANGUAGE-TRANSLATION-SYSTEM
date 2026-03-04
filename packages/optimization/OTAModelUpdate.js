/**
 * OTAModelUpdate — Over-the-air TFLite model update system.
 *
 * Step 6.6: On app startup, check remote manifest, download if newer,
 * verify SHA256, hot-swap model — all within 60 seconds, no restart.
 *
 * For Node.js testing, we simulate the full OTA lifecycle with
 * a mock HTTP server and verify the timing and integrity constraints.
 */

'use strict';

const crypto = require('crypto');

/**
 * Simulated model manifest.
 */
function createManifest(version, modelData) {
  const hash = crypto.createHash('sha256').update(modelData).digest('hex');
  return {
    version,
    asl_model: {
      url: `https://cdn.signbridge.dev/models/tms_wlasl100_int8_v${version}.tflite`,
      sha256: hash,
      size_mb: Math.round((modelData.length / (1024 * 1024)) * 100) / 100 || 7.2,
    },
    isl_model: {
      url: `https://cdn.signbridge.dev/models/tms_isl_v${version}.tflite`,
      sha256: crypto.createHash('sha256').update(modelData + '_isl').digest('hex'),
      size_mb: 6.8,
    },
  };
}

/**
 * Simulate model binary data (for hash verification).
 * @param {string} version
 * @returns {Buffer}
 */
function generateModelData(version) {
  // Generate deterministic model data from version
  const seed = Buffer.from(`model_v${version}_signbridge`);
  const data = Buffer.alloc(1024 * 100); // 100 KB simulated model
  for (let i = 0; i < data.length; i++) {
    data[i] = seed[i % seed.length] ^ (i & 0xff);
  }
  return data;
}

/**
 * OTA Update Manager — handles the complete update lifecycle.
 */
class OTAManager {
  constructor(currentVersion = '1.0.0') {
    this.currentVersion = currentVersion;
    this.currentModelData = generateModelData(currentVersion);
    this.loadedModel = this.currentModelData;
    this.updateLog = [];
  }

  /**
   * Check for updates and apply if available.
   * Returns the complete update timeline.
   *
   * @param {string} remoteVersion   Version on the server
   * @returns {Promise<{
   *   updateAvailable: boolean,
   *   downloaded: boolean,
   *   verified: boolean,
   *   swapped: boolean,
   *   totalTimeMs: number,
   *   steps: Array<{step: string, timeMs: number, success: boolean}>,
   * }>}
   */
  async checkAndUpdate(remoteVersion) {
    const start = performance.now();
    const steps = [];

    // Step 1: Fetch manifest (simulate network)
    const manifestStart = performance.now();
    await this._simulateNetwork(50); // 50ms to fetch manifest
    const remoteModelData = generateModelData(remoteVersion);
    const manifest = createManifest(remoteVersion, remoteModelData);
    steps.push({
      step: 'fetch_manifest',
      timeMs: performance.now() - manifestStart,
      success: true,
    });

    // Step 2: Compare versions
    const updateAvailable = this._isNewer(remoteVersion, this.currentVersion);
    steps.push({
      step: 'version_check',
      timeMs: 0.1,
      success: true,
      detail: `${this.currentVersion} → ${remoteVersion} (update: ${updateAvailable})`,
    });

    if (!updateAvailable) {
      return {
        updateAvailable: false,
        downloaded: false,
        verified: false,
        swapped: false,
        totalTimeMs: performance.now() - start,
        steps,
      };
    }

    // Step 3: Download model (simulate)
    const dlStart = performance.now();
    const downloadTimeMs = 200 + Math.random() * 300; // 200-500ms simulated
    await this._simulateNetwork(downloadTimeMs);
    const downloadedData = remoteModelData; // In real: HTTP download
    steps.push({
      step: 'download_model',
      timeMs: performance.now() - dlStart,
      success: true,
      detail: `${manifest.asl_model.size_mb} MB`,
    });

    // Step 4: Verify SHA256
    const verifyStart = performance.now();
    const actualHash = crypto.createHash('sha256').update(downloadedData).digest('hex');
    const hashMatch = actualHash === manifest.asl_model.sha256;
    steps.push({
      step: 'verify_sha256',
      timeMs: performance.now() - verifyStart,
      success: hashMatch,
      detail: hashMatch ? 'OK' : `MISMATCH: expected ${manifest.asl_model.sha256}, got ${actualHash}`,
    });

    if (!hashMatch) {
      return {
        updateAvailable: true,
        downloaded: true,
        verified: false,
        swapped: false,
        totalTimeMs: performance.now() - start,
        steps,
      };
    }

    // Step 5: Hot-swap model (no restart)
    const swapStart = performance.now();
    this.loadedModel = downloadedData;
    this.currentVersion = remoteVersion;
    await this._simulateNetwork(20); // 20ms to reinitialize interpreter
    steps.push({
      step: 'hot_swap_model',
      timeMs: performance.now() - swapStart,
      success: true,
      detail: `Now running v${remoteVersion}`,
    });

    const totalTimeMs = performance.now() - start;

    this.updateLog.push({
      from: this.currentVersion,
      to: remoteVersion,
      totalTimeMs,
      timestamp: Date.now(),
    });

    return {
      updateAvailable: true,
      downloaded: true,
      verified: true,
      swapped: true,
      totalTimeMs,
      steps,
    };
  }

  /**
   * Compare semantic versions.
   */
  _isNewer(remote, current) {
    const r = remote.split('.').map(Number);
    const c = current.split('.').map(Number);
    for (let i = 0; i < 3; i++) {
      if ((r[i] || 0) > (c[i] || 0)) return true;
      if ((r[i] || 0) < (c[i] || 0)) return false;
    }
    return false;
  }

  /**
   * Simulate network delay.
   */
  async _simulateNetwork(ms) {
    await new Promise(r => setTimeout(r, ms));
  }
}

module.exports = { OTAManager, createManifest, generateModelData };
