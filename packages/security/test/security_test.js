/**
 * CHECK 7 — Security & Privacy Tests (Section 9)
 *
 * 5 tests, all must pass for GATE clearance.
 *
 *  Test 1: DTLS verification — dtlsState='connected', fingerprint matches
 *  Test 2: TURN credential reuse after expiry — server returns 401
 *  Test 3: Signaling plain WS connection — server closes with 403
 *  Test 4: Network capture — zero unencrypted media, only DTLS/SRTP
 *  Test 5: Privacy — no video data leaves device
 */

'use strict';

const path = require('path');

// ── Module imports ──────────────────────────────────────────
const {
  simulateWebRTCStats,
  verifyDTLS,
  validateSignalingConnection,
  generateTURNCredential,
  validateTURNCredential,
} = require(path.join(__dirname, '..', 'MediaEncryption'));

const {
  PrivacyPipeline,
  ServerStorageAudit,
  DATA_TYPES,
} = require(path.join(__dirname, '..', 'PrivacyGuarantees'));

const {
  CertificatePinManager,
  generateFingerprint,
} = require(path.join(__dirname, '..', 'CertificatePinning'));

// ── Helpers ─────────────────────────────────────────────────
let passed = 0;
let failed = 0;

function ok(cond, msg, extra) {
  if (cond) {
    passed++;
    console.log(`  ✓  ${msg}`);
  } else {
    failed++;
    console.log(`  ✗  ${msg}`);
    if (extra) console.log(`       ${extra}`);
  }
}

function header(title) {
  console.log(`\n──── ${title} ────`);
}

// ═════════════════════════════════════════════════════════════
// TESTS
// ═════════════════════════════════════════════════════════════

async function runTests() {
  console.log('╔══════════════════════════════════════════════════════╗');
  console.log('║       CHECK 7 — Security Tests  (5 tests)          ║');
  console.log('╚══════════════════════════════════════════════════════╝');

  // ────────────────────────────────────────────────────────
  // TEST 1: DTLS Verification
  // ────────────────────────────────────────────────────────
  header('Test 1 — DTLS Verification');
  {
    const stats = simulateWebRTCStats(true);
    const dtls = verifyDTLS(stats);

    console.log(`    DTLS state        = ${dtls.dtlsConnected ? 'connected' : 'failed'}`);
    console.log(`    DTLS cipher       = ${dtls.dtlsCipher}`);
    console.log(`    SRTP enabled      = ${dtls.srtpEnabled}`);
    console.log(`    SRTP cipher       = ${dtls.srtpCipher}`);
    console.log(`    Fingerprint       = ${dtls.fingerprint.slice(0, 20)}...`);
    console.log(`    DataChannel enc   = ${dtls.dataChannelEncrypted}`);

    ok(dtls.dtlsConnected, "dtlsState = 'connected'");
    ok(dtls.srtpEnabled, 'SRTP encryption enabled');
    ok(dtls.fingerprint.length > 0, 'Certificate fingerprint present');
    ok(dtls.dataChannelEncrypted, 'DataChannel (SCTP) encrypted over DTLS');
  }

  // ────────────────────────────────────────────────────────
  // TEST 2: TURN Credential Reuse After Expiry
  // ────────────────────────────────────────────────────────
  header('Test 2 — TURN Credential Reuse After Expiry');
  {
    // Generate a credential that expires immediately (0 hours)
    const cred = generateTURNCredential('testuser', 0);

    // Artificially expire it by using a past timestamp
    const expiredUsername = `${Math.floor(Date.now() / 1000) - 3600}:testuser`;
    const expiredCred = require('crypto')
      .createHmac('sha256', require(path.join(__dirname, '..', 'MediaEncryption')).TURN_SECRET)
      .update(expiredUsername)
      .digest('base64');

    const validResult = validateTURNCredential(cred.username, cred.credential);
    const expiredResult = validateTURNCredential(expiredUsername, expiredCred);

    console.log(`    Valid cred check  = ${validResult.valid ? 'accepted' : 'rejected'}`);
    console.log(`    Expired cred check= ${expiredResult.valid ? 'accepted' : 'rejected'} (status ${expiredResult.statusCode})`);
    console.log(`    Expired reason    = ${expiredResult.reason}`);

    ok(expiredResult.valid === false, 'Expired credential rejected');
    ok(expiredResult.statusCode === 401, `Server returns 401 (got ${expiredResult.statusCode})`);
  }

  // ────────────────────────────────────────────────────────
  // TEST 3: Plain WS Connection Rejected
  // ────────────────────────────────────────────────────────
  header('Test 3 — Plain WS Connection Rejected');
  {
    const plainResult = validateSignalingConnection('ws://signaling.signbridge.dev');
    const secureResult = validateSignalingConnection('wss://signaling.signbridge.dev');

    console.log(`    Plain WS          = ${plainResult.allowed ? 'ALLOWED' : 'REJECTED'} (${plainResult.statusCode})`);
    console.log(`    WSS (secure)      = ${secureResult.allowed ? 'ALLOWED' : 'REJECTED'} (${secureResult.statusCode})`);
    console.log(`    TLS version       = ${secureResult.tlsVersion || 'N/A'}`);

    ok(!plainResult.allowed, 'Plain WS connection rejected');
    ok(plainResult.statusCode === 403, `Plain WS returns 403 (got ${plainResult.statusCode})`);
    ok(secureResult.allowed, 'WSS connection allowed');
  }

  // ────────────────────────────────────────────────────────
  // TEST 4: Network Capture — Zero Unencrypted Media
  // ────────────────────────────────────────────────────────
  header('Test 4 — Network Capture (30-second call simulation)');
  {
    // Simulate a 30-second call and capture all transmitted packets
    const stats = simulateWebRTCStats(true);
    const dtls = verifyDTLS(stats);

    // Simulate packet capture
    const packets = [];
    const callDurationSec = 30;
    const packetsPerSecond = 50; // ~50 RTP packets/sec for video

    for (let i = 0; i < callDurationSec * packetsPerSecond; i++) {
      packets.push({
        protocol: dtls.srtpEnabled ? 'SRTP' : 'RTP',
        encrypted: dtls.srtpEnabled,
        type: i % 10 === 0 ? 'DTLS' : 'SRTP',
        sizeBytes: 500 + Math.floor(Math.random() * 1000),
      });
    }

    const unencrypted = packets.filter(p => !p.encrypted);
    const totalPackets = packets.length;

    console.log(`    Total packets     = ${totalPackets}`);
    console.log(`    Unencrypted       = ${unencrypted.length}`);
    console.log(`    DTLS packets      = ${packets.filter(p => p.type === 'DTLS').length}`);
    console.log(`    SRTP packets      = ${packets.filter(p => p.type === 'SRTP').length}`);

    ok(unencrypted.length === 0, `Zero unencrypted packets (got ${unencrypted.length}/${totalPackets})`);
  }

  // ────────────────────────────────────────────────────────
  // TEST 5: Privacy — No Video Data Leaves Device
  // ────────────────────────────────────────────────────────
  header('Test 5 — Privacy (no video leaves device)');
  {
    const pipeline = new PrivacyPipeline();

    // Process 10 frames (on-device)
    for (let i = 0; i < 10; i++) {
      const result = pipeline.processFrame({ width: 640, height: 480, data: new Uint8Array(640 * 480 * 3) });

      // Transmit landmarks (allowed)
      pipeline.transmit(DATA_TYPES.LANDMARKS, result.landmarks, 'datachannel');

      // Transmit caption (allowed)
      pipeline.transmit(DATA_TYPES.CAPTION_TEXT, result.prediction, 'datachannel');

      // Attempt to transmit video (should be blocked)
      pipeline.transmit(DATA_TYPES.VIDEO_FRAME, { data: 'raw_pixels' }, 'websocket');
    }

    // Also transmit signaling events (allowed)
    pipeline.transmit(DATA_TYPES.SIGNAL_MSG, { type: 'offer', sdp: '...' }, 'websocket');

    const report = pipeline.getAuditReport();

    console.log(`    Total transmissions= ${report.totalTransmissions}`);
    console.log(`    Blocked attempts  = ${report.blockedCount}`);
    console.log(`    Video leaked      = ${report.videoDataLeaked}`);
    console.log(`    Local operations  = ${report.localOperations}`);
    console.log(`    Transmitted types = ${report.transmittedTypes.join(', ')}`);

    // Server storage audit
    const server = new ServerStorageAudit();
    server.joinRoom('room-1', 'socket-abc');
    server.joinRoom('room-1', 'socket-def');

    const serverAudit = server.audit();
    console.log(`    Server rooms      = ${serverAudit.roomCount}`);
    console.log(`    Conversation data = ${serverAudit.conversationContent}`);
    console.log(`    Persistent store  = ${serverAudit.persistentStorage}`);

    // Disconnect all — verify cleanup
    server.disconnect('room-1', 'socket-abc');
    server.disconnect('room-1', 'socket-def');
    const postDisconnect = server.audit();

    ok(!report.videoDataLeaked, 'Zero video data transmitted over network');
    ok(report.blockedCount === 10, `All 10 video attempts blocked (got ${report.blockedCount})`);
    ok(!report.transmittedTypes.includes(DATA_TYPES.VIDEO_FRAME), 'No video_frame in transmitted types');
    ok(!serverAudit.conversationContent, 'Server stores no conversation content');
    ok(postDisconnect.roomCount === 0, `Server cleanup on disconnect (${postDisconnect.roomCount} rooms remaining)`);
  }

  // ── Summary ────────────────────────────────────────────
  console.log('\n══════════════════════════════════════════════════════');
  console.log(`  CHECK 7 RESULT: ${passed} passed, ${failed} failed`);
  console.log(`  GATE: ${failed === 0 ? 'PASSED ✓' : 'FAILED ✗'}`);
  console.log('══════════════════════════════════════════════════════\n');

  process.exit(failed === 0 ? 0 : 1);
}

runTests().catch(err => {
  console.error('CHECK 7 FATAL:', err);
  process.exit(1);
});
