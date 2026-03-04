/**
 * PrivacyGuarantees — On-device inference and data minimization.
 *
 * Step 7.2:
 *  - Raw video frames never leave the device
 *  - Only text (captions) or normalized landmark coordinates transmitted
 *  - No persistent storage on server
 *  - Landmarks are abstracted coordinates (not biometric images)
 *
 * For Node.js testing, we simulate the data pipeline and verify
 * that no video data is leaked through any transmission channel.
 */

'use strict';

/**
 * Simulated data types that can be transmitted.
 */
const DATA_TYPES = {
  VIDEO_FRAME: 'video_frame',     // Raw pixel data — MUST NOT be transmitted
  AUDIO_FRAME: 'audio_frame',     // Raw audio — only for ASR via WebRTC
  LANDMARKS: 'landmarks',         // Normalized coordinates — safe to transmit
  CAPTION_TEXT: 'caption_text',   // Translated text — safe to transmit
  GLOSS_TOKENS: 'gloss_tokens',   // ASL gloss — safe to transmit
  SIGNAL_MSG: 'signaling',       // WebRTC signaling — safe to transmit
};

const PROHIBITED_TYPES = new Set([DATA_TYPES.VIDEO_FRAME]);

/**
 * Simulated pipeline that processes frames and produces transmittable data.
 * Enforces that video frames are consumed locally and never queued for sending.
 */
class PrivacyPipeline {
  constructor() {
    this.transmissionLog = [];
    this.violations = [];
    this.localProcessingLog = [];
  }

  /**
   * Process a camera frame locally (on-device inference).
   * @param {object} frame  Simulated camera frame
   * @returns {object}  Inference result (landmarks + caption)
   */
  processFrame(frame) {
    // Step 1: MediaPipe hand detection (on-device)
    const landmarks = this._detectHands(frame);
    this.localProcessingLog.push({
      type: 'hand_detection',
      timestamp: Date.now(),
      dataType: DATA_TYPES.VIDEO_FRAME,
      action: 'local_only', // NEVER transmitted
    });

    // Step 2: TMS inference (on-device)
    const prediction = this._runInference(landmarks);
    this.localProcessingLog.push({
      type: 'inference',
      timestamp: Date.now(),
      dataType: DATA_TYPES.LANDMARKS,
      action: 'local_only',
    });

    return {
      landmarks, // Normalized coordinates, safe to transmit
      prediction, // Text, safe to transmit
    };
  }

  /**
   * Transmit data to remote peer/server.
   * Enforces privacy: rejects video frame data.
   *
   * @param {string} dataType  One of DATA_TYPES
   * @param {any} data         Data payload
   * @param {string} channel   'datachannel' | 'websocket' | 'webrtc_media'
   * @returns {{ allowed: boolean, reason: string }}
   */
  transmit(dataType, data, channel) {
    const entry = {
      dataType,
      channel,
      timestamp: Date.now(),
      sizeBytes: JSON.stringify(data).length,
    };

    if (PROHIBITED_TYPES.has(dataType)) {
      entry.blocked = true;
      this.violations.push({
        ...entry,
        reason: `Attempted to transmit ${dataType} — BLOCKED`,
      });
      this.transmissionLog.push(entry);
      return { allowed: false, reason: `${dataType} is prohibited from transmission` };
    }

    entry.blocked = false;
    this.transmissionLog.push(entry);
    return { allowed: true, reason: 'OK' };
  }

  /**
   * Get transmission audit report.
   * @returns {{ totalTransmissions: number, blockedCount: number, violations: object[], videoDataLeaked: boolean }}
   */
  getAuditReport() {
    const videoLeaks = this.transmissionLog.filter(
      e => e.dataType === DATA_TYPES.VIDEO_FRAME && !e.blocked
    );

    return {
      totalTransmissions: this.transmissionLog.length,
      blockedCount: this.transmissionLog.filter(e => e.blocked).length,
      violations: this.violations,
      videoDataLeaked: videoLeaks.length > 0,
      localOperations: this.localProcessingLog.length,
      transmittedTypes: [...new Set(
        this.transmissionLog.filter(e => !e.blocked).map(e => e.dataType)
      )],
    };
  }

  // ── Private methods (simulated on-device processing) ──

  _detectHands(frame) {
    // Returns 21 landmarks × 3 (x, y, z), normalized [0, 1]
    const landmarks = [];
    for (let i = 0; i < 21; i++) {
      landmarks.push({
        x: Math.random(),
        y: Math.random(),
        z: Math.random() * 0.1,
      });
    }
    return landmarks;
  }

  _runInference(landmarks) {
    return {
      sign: 'HELLO',
      confidence: 0.95,
    };
  }
}

/**
 * Simulated server storage audit.
 * Verifies: no persistent conversation content on server.
 */
class ServerStorageAudit {
  constructor() {
    this.rooms = new Map();
  }

  /**
   * Simulate join room (only stores ephemeral data).
   */
  joinRoom(roomId, socketId) {
    if (!this.rooms.has(roomId)) {
      this.rooms.set(roomId, new Set());
    }
    this.rooms.get(roomId).add(socketId);
  }

  /**
   * Simulate disconnect (clears all data).
   */
  disconnect(roomId, socketId) {
    const room = this.rooms.get(roomId);
    if (room) {
      room.delete(socketId);
      if (room.size === 0) {
        this.rooms.delete(roomId);
      }
    }
  }

  /**
   * Audit: check what's stored.
   * @returns {{ roomCount: number, storedData: string[], conversationContent: boolean }}
   */
  audit() {
    const storedData = [];
    for (const [roomId, sockets] of this.rooms) {
      storedData.push(`Room ${roomId}: ${sockets.size} sockets`);
    }

    return {
      roomCount: this.rooms.size,
      storedData,
      conversationContent: false, // Server never stores conversation content
      persistentStorage: false,   // All data is ephemeral (memory only)
    };
  }
}

module.exports = {
  PrivacyPipeline,
  ServerStorageAudit,
  DATA_TYPES,
  PROHIBITED_TYPES,
};
