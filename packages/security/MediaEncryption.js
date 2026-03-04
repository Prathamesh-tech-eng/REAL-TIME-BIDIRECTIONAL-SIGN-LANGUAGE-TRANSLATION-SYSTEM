/**
 * MediaEncryption — DTLS-SRTP, SCTP/DTLS, and WSS enforcement.
 *
 * Step 7.1: WebRTC media uses DTLS-SRTP (enforced by default).
 * RTCDataChannel uses SCTP over DTLS. Signaling uses WSS (TLS 1.3).
 * TURN credentials: HMAC-SHA256 with 24h expiry.
 *
 * For Node.js testing, we simulate WebRTC stats, verify DTLS state,
 * and validate encryption properties.
 */

'use strict';

const crypto = require('crypto');

// ── DTLS-SRTP Verification ──────────────────────────────────

/**
 * Simulated RTCPeerConnection stats.
 * In production: pc.getStats() returns these values.
 */
function simulateWebRTCStats(dtlsConnected = true) {
  const fingerprint = crypto.randomBytes(32).toString('hex').replace(/(.{2})/g, '$1:').slice(0, -1).toUpperCase();

  return {
    transport: {
      type: 'transport',
      dtlsState: dtlsConnected ? 'connected' : 'failed',
      dtlsCipher: 'TLS_AES_256_GCM_SHA384',
      srtpCipher: 'AEAD_AES_128_GCM',
      tlsVersion: 'DTLS 1.2',
      selectedCandidatePairId: 'candidate-pair-0',
      localCertificateId: 'cert-local',
      remoteCertificateId: 'cert-remote',
    },
    localCertificate: {
      type: 'certificate',
      fingerprint,
      fingerprintAlgorithm: 'sha-256',
      base64Certificate: crypto.randomBytes(256).toString('base64'),
    },
    candidatePair: {
      type: 'candidate-pair',
      state: 'succeeded',
      localCandidateId: 'candidate-local',
      remoteCandidateId: 'candidate-remote',
      nominated: true,
      bytesReceived: 1024000,
      bytesSent: 1024000,
    },
    dataChannel: {
      label: 'landmarks',
      protocol: 'sctp',
      state: 'open',
      // SCTP is always encrypted over DTLS
      encrypted: true,
    },
  };
}

/**
 * Verify DTLS-SRTP state from WebRTC stats.
 * @param {object} stats  Simulated peer connection stats
 * @returns {{ dtlsConnected: boolean, srtpEnabled: boolean, fingerprint: string }}
 */
function verifyDTLS(stats) {
  return {
    dtlsConnected: stats.transport.dtlsState === 'connected',
    dtlsCipher: stats.transport.dtlsCipher,
    srtpEnabled: !!stats.transport.srtpCipher,
    srtpCipher: stats.transport.srtpCipher,
    tlsVersion: stats.transport.tlsVersion,
    fingerprint: stats.localCertificate.fingerprint,
    dataChannelEncrypted: stats.dataChannel.encrypted,
  };
}

// ── WSS Enforcement ─────────────────────────────────────────

/**
 * Validate signaling connection security.
 * Rejects plain WS connections (only WSS with TLS 1.3).
 *
 * @param {string} url  WebSocket URL
 * @returns {{ allowed: boolean, reason: string, statusCode: number }}
 */
function validateSignalingConnection(url) {
  if (!url.startsWith('wss://')) {
    return { allowed: false, reason: 'Plain WS connection rejected', statusCode: 403 };
  }

  // Validate TLS version (simulated)
  return {
    allowed: true,
    reason: 'WSS connection with TLS 1.3',
    statusCode: 101, // Upgrade successful
    tlsVersion: 'TLSv1.3',
    cipher: 'TLS_AES_256_GCM_SHA384',
  };
}

// ── TURN Credential Management ──────────────────────────────

const TURN_SECRET = 'signbridge-turn-secret-key-2024';
const TURN_EXPIRY_HOURS = 24;

/**
 * Generate time-limited TURN credentials using HMAC-SHA256.
 * @param {string} username  Base username
 * @param {number} [expiryHours]  Hours until expiry
 * @returns {{ username: string, credential: string, expiresAt: number }}
 */
function generateTURNCredential(username, expiryHours = TURN_EXPIRY_HOURS) {
  const expiresAt = Math.floor(Date.now() / 1000) + expiryHours * 3600;
  const timestampedUsername = `${expiresAt}:${username}`;
  const credential = crypto.createHmac('sha256', TURN_SECRET)
    .update(timestampedUsername)
    .digest('base64');

  return { username: timestampedUsername, credential, expiresAt };
}

/**
 * Validate a TURN credential.
 * @param {string} username  Timestamped username
 * @param {string} credential  HMAC credential
 * @returns {{ valid: boolean, expired: boolean, reason: string }}
 */
function validateTURNCredential(username, credential) {
  // Check expiry
  const parts = username.split(':');
  const expiresAt = parseInt(parts[0], 10);
  const now = Math.floor(Date.now() / 1000);

  if (now > expiresAt) {
    return { valid: false, expired: true, reason: 'Credential expired', statusCode: 401 };
  }

  // Verify HMAC
  const expected = crypto.createHmac('sha256', TURN_SECRET)
    .update(username)
    .digest('base64');

  if (credential !== expected) {
    return { valid: false, expired: false, reason: 'Invalid HMAC', statusCode: 401 };
  }

  return { valid: true, expired: false, reason: 'Valid', statusCode: 200 };
}

module.exports = {
  simulateWebRTCStats,
  verifyDTLS,
  validateSignalingConnection,
  generateTURNCredential,
  validateTURNCredential,
  TURN_SECRET,
  TURN_EXPIRY_HOURS,
};
