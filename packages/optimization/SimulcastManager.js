/**
 * SimulcastManager — WebRTC simulcast for multi-layer video.
 *
 * Step 6.1: Enable 3 layers (360p / 720p / 1080p). Recipient selects
 * based on available bandwidth. Codec preference: H264 > VP8 > AV1.
 *
 * For Node.js testing, we simulate sender/receiver with bandwidth
 * constraints and verify automatic layer selection.
 */

'use strict';

// Simulcast layer definitions
const LAYERS = [
  { rid: 'low',  width: 640,  height: 360, maxBitrate: 500_000,   scaleResolutionDownBy: 4.0 },
  { rid: 'mid',  width: 1280, height: 720, maxBitrate: 1_500_000, scaleResolutionDownBy: 2.0 },
  { rid: 'high', width: 1920, height: 1080, maxBitrate: 2_500_000, scaleResolutionDownBy: 1.0 },
];

const CODEC_PRIORITY = ['H264', 'VP8', 'AV1'];

/**
 * Select the best simulcast layer given available bandwidth.
 * @param {number} availableBps  Available bandwidth in bits/second
 * @returns {{ layer: object, index: number }}
 */
function selectLayer(availableBps) {
  // Select highest layer that fits within bandwidth
  for (let i = LAYERS.length - 1; i >= 0; i--) {
    if (availableBps >= LAYERS[i].maxBitrate * 0.8) {
      return { layer: LAYERS[i], index: i };
    }
  }
  // Fall back to lowest layer
  return { layer: LAYERS[0], index: 0 };
}

/**
 * Simulate a receiver on throttled bandwidth.
 * Returns metrics about the selected layer and buffering.
 *
 * @param {number} bandwidthKbps  Bandwidth in kbps
 * @param {number} durationSec    Simulation duration
 * @returns {{ selectedLayer: string, resolution: string, bufferingEvents: number, avgFps: number }}
 */
function simulateReceiver(bandwidthKbps, durationSec = 10) {
  const bandwidthBps = bandwidthKbps * 1000;
  const { layer } = selectLayer(bandwidthBps);

  // The video encoder targets the layer's maxBitrate.
  // Effective FPS = bandwidth / layer bitrate × 30 (capped at 30)
  const effectiveBitrate = Math.min(bandwidthBps, layer.maxBitrate);
  const theoreticalFps = (effectiveBitrate / layer.maxBitrate) * 30;
  const actualFps = Math.min(30, Math.max(0, theoreticalFps));

  // Buffering occurs when actual FPS < 15
  const bufferingEvents = actualFps < 15 ? Math.ceil(durationSec / 2) : 0;

  return {
    selectedLayer: layer.rid,
    resolution: `${layer.width}x${layer.height}`,
    bufferingEvents,
    avgFps: Math.round(actualFps * 10) / 10,
    bitrateKbps: Math.round(layer.maxBitrate / 1000),
  };
}

/**
 * Get preferred codec ordering.
 * @param {string[]} available  Array of available codecs
 * @returns {string}  Selected codec
 */
function preferCodec(available) {
  for (const pref of CODEC_PRIORITY) {
    if (available.includes(pref)) return pref;
  }
  return available[0] || 'VP8';
}

module.exports = {
  LAYERS,
  CODEC_PRIORITY,
  selectLayer,
  simulateReceiver,
  preferCodec,
};
