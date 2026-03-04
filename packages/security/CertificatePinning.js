/**
 * CertificatePinning — TLS certificate pinning for signaling server.
 *
 * Step 7.3: Pin SHA256 fingerprints, 30-day rotation plan with backup pin.
 *
 * For Node.js testing, we verify the pinning logic, rotation schedule,
 * and fallback to backup pins.
 */

'use strict';

const crypto = require('crypto');

/**
 * Generate a self-signed test certificate fingerprint.
 * @returns {string}  SHA256 fingerprint
 */
function generateFingerprint() {
  return crypto.createHash('sha256')
    .update(crypto.randomBytes(256))
    .digest('hex')
    .replace(/(.{2})/g, '$1:')
    .slice(0, -1)
    .toUpperCase();
}

/**
 * Certificate pin configuration.
 */
class CertificatePinManager {
  /**
   * @param {string} primaryPin     Current certificate SHA256 fingerprint
   * @param {string} backupPin      Backup certificate SHA256 fingerprint
   * @param {number} rotationDays   Pin rotation interval (default: 30)
   */
  constructor(primaryPin, backupPin, rotationDays = 30) {
    this.primaryPin = primaryPin;
    this.backupPin = backupPin;
    this.rotationDays = rotationDays;
    this.pinnedAt = Date.now();
    this.rotationLog = [];
  }

  /**
   * Verify a server certificate against pinned fingerprints.
   * @param {string} serverFingerprint  The server's certificate fingerprint
   * @returns {{ trusted: boolean, matchedPin: string, reason: string }}
   */
  verify(serverFingerprint) {
    if (serverFingerprint === this.primaryPin) {
      return { trusted: true, matchedPin: 'primary', reason: 'Matched primary pin' };
    }
    if (serverFingerprint === this.backupPin) {
      return { trusted: true, matchedPin: 'backup', reason: 'Matched backup pin (primary may be rotating)' };
    }
    return { trusted: false, matchedPin: 'none', reason: 'Certificate does not match any pinned fingerprint' };
  }

  /**
   * Check if pin rotation is due.
   * @returns {{ rotationDue: boolean, daysUntilRotation: number }}
   */
  checkRotation() {
    const elapsed = Date.now() - this.pinnedAt;
    const elapsedDays = elapsed / (1000 * 3600 * 24);
    const daysUntilRotation = Math.max(0, this.rotationDays - elapsedDays);

    return {
      rotationDue: elapsedDays >= this.rotationDays,
      daysUntilRotation: Math.round(daysUntilRotation * 10) / 10,
      daysSincePinned: Math.round(elapsedDays * 10) / 10,
    };
  }

  /**
   * Rotate pins: old backup becomes primary, new fingerprint becomes backup.
   * @param {string} newBackupPin  New backup pin
   */
  rotate(newBackupPin) {
    this.rotationLog.push({
      timestamp: Date.now(),
      oldPrimary: this.primaryPin,
      oldBackup: this.backupPin,
      newPrimary: this.backupPin,
      newBackup: newBackupPin,
    });

    this.primaryPin = this.backupPin;
    this.backupPin = newBackupPin;
    this.pinnedAt = Date.now();
  }

  /**
   * Get pin configuration (for React Native SSL pinning).
   * @returns {object}
   */
  getPinConfig() {
    return {
      'signaling.signbridge.dev': {
        includeSubdomains: true,
        pins: [
          `sha256/${Buffer.from(this.primaryPin.replace(/:/g, ''), 'hex').toString('base64')}`,
          `sha256/${Buffer.from(this.backupPin.replace(/:/g, ''), 'hex').toString('base64')}`,
        ],
        rotationDays: this.rotationDays,
        expiresAt: new Date(this.pinnedAt + this.rotationDays * 24 * 3600 * 1000).toISOString(),
      },
    };
  }
}

module.exports = {
  CertificatePinManager,
  generateFingerprint,
};
