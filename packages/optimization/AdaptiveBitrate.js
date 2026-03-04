/**
 * AdaptiveBitrate — Monitor connection stats and adjust quality.
 *
 * Step 6.1: Monitor connection.getStats() every 2s. If packetLossRate > 5%,
 * downscale resolution. Auto-recover when conditions improve.
 *
 * For Node.js testing, we simulate network conditions and verify
 * the adaptation engine downscales within the required 4-second window.
 */

'use strict';

const { selectLayer, LAYERS } = require('./SimulcastManager');

const POLL_INTERVAL_MS = 2000;  // 2 seconds
const LOSS_THRESHOLD = 0.05;    // 5% packet loss triggers downscale
const RECOVERY_THRESHOLD = 0.02; // < 2% packet loss to upscale
const DOWNSCALE_DEADLINE_MS = 4000; // Must adapt within 4 seconds

/**
 * Simulated network stats.
 */
class NetworkCondition {
  constructor() {
    this.packetLossRate = 0;
    this.rttMs = 50;
    this.availableBandwidthKbps = 2500;
    this.jitterMs = 5;
  }

  injectPacketLoss(rate) {
    this.packetLossRate = rate;
    // Packet loss typically also reduces effective bandwidth
    this.availableBandwidthKbps = Math.round(this.availableBandwidthKbps * (1 - rate * 3));
  }

  recover() {
    this.packetLossRate = 0.01;
    this.availableBandwidthKbps = 2500;
  }
}

/**
 * Adaptive bitrate controller.
 */
class AdaptiveBitrateController {
  constructor() {
    this.currentLayerIndex = 2; // Start at highest
    this.history = []; // Array of { timestamp, layerIndex, reason }
    this.frozen = false;
    this.lastChangeTime = 0;
  }

  /**
   * Process a stats sample and decide whether to change layer.
   * @param {NetworkCondition} stats
   * @param {number} timestamp  Simulated timestamp in ms
   * @returns {{ changed: boolean, layer: object, reason: string }}
   */
  processStats(stats, timestamp) {
    let reason = 'stable';
    let changed = false;

    if (stats.packetLossRate > LOSS_THRESHOLD && this.currentLayerIndex > 0) {
      // Downscale
      this.currentLayerIndex = Math.max(0, this.currentLayerIndex - 1);
      reason = `packet_loss_${(stats.packetLossRate * 100).toFixed(1)}%`;
      changed = true;
      this.frozen = false;
    } else if (stats.packetLossRate < RECOVERY_THRESHOLD && this.currentLayerIndex < LAYERS.length - 1) {
      // Only upscale if stable for > 10 seconds
      if (timestamp - this.lastChangeTime > 10000) {
        this.currentLayerIndex = Math.min(LAYERS.length - 1, this.currentLayerIndex + 1);
        reason = 'recovery';
        changed = true;
      }
    }

    // Check for freeze: if loss > 20%, mark frozen
    this.frozen = stats.packetLossRate > 0.20;

    if (changed) {
      this.lastChangeTime = timestamp;
    }

    const entry = {
      timestamp,
      layerIndex: this.currentLayerIndex,
      layer: LAYERS[this.currentLayerIndex],
      reason,
      changed,
      frozen: this.frozen,
    };
    this.history.push(entry);

    return entry;
  }

  /**
   * Simulate adaptive bitrate over a timeline of network conditions.
   *
   * @param {Array<{timeMs: number, lossRate: number}>} events
   *   Sorted array of events: at timeMs, set loss rate
   * @param {number} totalDurationMs  Total simulation duration
   * @returns {{
   *   history: object[],
   *   downscaledWithinDeadline: boolean,
   *   frozenDuringAdaptation: boolean,
   *   finalLayer: string
   * }}
   */
  simulate(events, totalDurationMs = 30000) {
    this.currentLayerIndex = 2;
    this.history = [];
    this.lastChangeTime = 0;

    const condition = new NetworkCondition();
    let eventIdx = 0;
    let downscaleDeadlineCheck = null;
    let downscaledInTime = true;
    let frozenDuringAdapt = false;
    let lossInjectedAt = null;

    for (let t = 0; t <= totalDurationMs; t += POLL_INTERVAL_MS) {
      // Apply any events at this time
      while (eventIdx < events.length && events[eventIdx].timeMs <= t) {
        condition.injectPacketLoss(events[eventIdx].lossRate);
        if (events[eventIdx].lossRate > LOSS_THRESHOLD && lossInjectedAt === null) {
          lossInjectedAt = events[eventIdx].timeMs;
        }
        if (events[eventIdx].lossRate < RECOVERY_THRESHOLD) {
          condition.recover();
          lossInjectedAt = null;
        }
        eventIdx++;
      }

      const result = this.processStats(condition, t);

      // Check deadline: if loss was injected and we haven't downscaled within 4s
      if (lossInjectedAt !== null && t - lossInjectedAt <= DOWNSCALE_DEADLINE_MS) {
        if (result.changed && result.reason.startsWith('packet_loss')) {
          // Good: adapted in time
          downscaleDeadlineCheck = t - lossInjectedAt;
        }
        if (result.frozen) {
          frozenDuringAdapt = true;
        }
      }
    }

    // If loss was injected but never downscaled in time
    if (lossInjectedAt !== null && downscaleDeadlineCheck === null) {
      // Check if any downscale happened within deadline
      const lossEvents = this.history.filter(h =>
        h.reason.startsWith('packet_loss') && h.timestamp - lossInjectedAt <= DOWNSCALE_DEADLINE_MS
      );
      if (lossEvents.length === 0) {
        downscaledInTime = false;
      }
    }

    return {
      history: this.history,
      downscaledWithinDeadline: downscaledInTime,
      downscaleLatencyMs: downscaleDeadlineCheck,
      frozenDuringAdaptation: frozenDuringAdapt,
      finalLayer: LAYERS[this.currentLayerIndex].rid,
    };
  }
}

module.exports = {
  AdaptiveBitrateController,
  NetworkCondition,
  POLL_INTERVAL_MS,
  LOSS_THRESHOLD,
  RECOVERY_THRESHOLD,
  DOWNSCALE_DEADLINE_MS,
};
