/**
 * CHECK 8 — E2E Testing & Deployment Gate
 *
 * Runs all 7 E2E scenarios (E1-E7), the performance regression suite,
 * and the deployment checklist. Every assertion must pass.
 *
 * Usage:  node packages/e2e/test/e2e_test.js
 */

'use strict';

const { runE1, runE2, runE3, runE4, runE5, runE6, runE7 } = require('../E2EScenarios');
const {
  runInferenceBenchmark,
  runAccuracyCheck,
  runSignalingLoadTest,
  runDeploymentChecklist,
} = require('../RegressionSuite');

// ── Minimal test harness ────────────────────────────────────
let passed = 0;
let failed = 0;
let total  = 0;

function assert(condition, label) {
  total++;
  if (condition) {
    passed++;
    console.log(`  [PASS] ${label}`);
  } else {
    failed++;
    console.log(`  [FAIL] ${label}`);
  }
}

// ── Main ────────────────────────────────────────────────────
(async () => {
  console.log('╔═══════════════════════════════════════════════════╗');
  console.log('║         CHECK 8 — E2E & Deployment Gate          ║');
  console.log('╚═══════════════════════════════════════════════════╝\n');

  // ── E1: ASL ↔ English, good WiFi ──────────────────────────
  console.log('─── E1: ASL ↔ English, good WiFi ───');
  const e1 = await runE1();
  console.log(`    Max latency: ${e1.maxLatencyMs} ms  |  Drops: ${e1.drops}`);
  assert(e1.captionLatencyOk, 'E1: Captions < 200 ms');
  assert(e1.noDrops,          'E1: No frame drops');
  assert(e1.avatarCorrect,    'E1: Avatar signs correctly');

  // ── E2: ISL ↔ Hindi, 4G ──────────────────────────────────
  console.log('\n─── E2: ISL ↔ Hindi, 4G ───');
  const e2 = await runE2();
  console.log(`    Max latency: ${e2.maxLatencyMs} ms  |  Avg: ${e2.avgLatencyMs} ms`);
  assert(e2.latencyOk,          'E2: Latency < 400 ms');
  assert(e2.pipelineFunctional, 'E2: Pipeline functional');
  assert(e2.adaptiveBitrateEngaged, 'E2: Adaptive bitrate engages');

  // ── E3: Poor network ─────────────────────────────────────
  console.log('\n─── E3: Poor network (500 kbps, 200ms RTT) ───');
  const e3 = await runE3();
  console.log(`    Delivered: ${e3.delivered}/${e3.totalSigns}  |  Dropped: ${e3.dropped}`);
  assert(e3.gracefulDegradation, 'E3: Graceful degradation (≥80% delivered)');
  assert(e3.captionsNotLost,     'E3: Captions delayed but not lost');

  // ── E4: Network dropout ───────────────────────────────────
  console.log('\n─── E4: 5s network dropout ───');
  const e4 = await runE4();
  console.log(`    ICE restarts: ${e4.iceRestarts}  |  Reconnect: ${e4.reconnectTimeMs} ms`);
  assert(e4.duringDropped,            'E4: Signs dropped during disconnect');
  assert(e4.resumedWithin10s,         'E4: Call resumes within 10 s');
  assert(e4.noDataLossAfterResume,    'E4: No data loss after resume');

  // ── E5: ASL-to-ASL (full duplex) ─────────────────────────
  console.log('\n─── E5: ASL-to-ASL (full duplex) ───');
  const e5 = await runE5();
  console.log(`    User A: ${e5.userADelivered} delivered  |  User B: ${e5.userBDelivered} delivered`);
  assert(e5.fullDuplex,      'E5: Full duplex (both sides translate)');
  assert(e5.noInterference,  'E5: No inter-stream interference');

  // ── E6: Background noise (70 dB) ─────────────────────────
  console.log('\n─── E6: Background noise (70 dB) ───');
  const e6 = await runE6();
  console.log(`    Avg ASR accuracy: ${(e6.avgAsrAccuracy * 100).toFixed(1)}%  |  Avg word accuracy: ${(e6.avgWordAccuracy * 100).toFixed(1)}%`);
  assert(e6.accuracyAbove80, 'E6: ASR > 80% word accuracy at 70 dB');

  // ── E7: 60-minute long session ────────────────────────────
  console.log('\n─── E7: 60-minute long session ───');
  const e7 = await runE7();
  console.log(`    Heap growth: ${e7.heapGrowthMB} MB  |  Battery drain: ${e7.batteryDrainPercent}%`);
  assert(e7.noCrash,           'E7: No crash');
  assert(e7.noLeak,            'E7: No memory leak');
  assert(e7.noThermalThrottle, 'E7: No thermal throttle');
  assert(e7.batteryOk,         'E7: Battery drain < 30%');

  // ── Regression: Inference benchmark ───────────────────────
  console.log('\n─── Regression: Inference benchmark ───');
  const inf = runInferenceBenchmark(500);
  console.log(`    Mean: ${inf.meanMs} ms  |  P99: ${inf.p99Ms} ms  (n=${inf.iterations})`);
  assert(inf.meanMs < 65, 'R1: Inference mean < 65 ms');
  assert(inf.p99Ms < 90,  'R1: Inference P99 < 90 ms');

  // ── Regression: Accuracy check ────────────────────────────
  console.log('\n─── Regression: Accuracy check ───');
  const acc = runAccuracyCheck(1000);
  console.log(`    Top-1: ${(acc.top1Accuracy * 100).toFixed(1)}%  |  F1: ${acc.f1Score}`);
  assert(acc.top1Accuracy > 0.90, 'R2: Top-1 accuracy > 90%');
  assert(acc.f1Score > 0.85,      'R2: F1 score > 0.85');

  // ── Regression: Signaling load test ───────────────────────
  console.log('\n─── Regression: Signaling load test ───');
  const load = runSignalingLoadTest(100, 60);
  console.log(`    Rooms: ${load.rooms}  |  Avg latency: ${load.avgLatencyMs} ms  |  Failures: ${load.failures}/${load.totalMessages}`);
  assert(load.stable, 'R3: Signaling stable under load (< 0.1% failure)');

  // ── Deployment checklist ──────────────────────────────────
  console.log('\n─── Deployment Checklist ───');
  const deploy = runDeploymentChecklist();
  deploy.items.forEach(item => {
    console.log(`    [${item.status}] ${item.check}`);
  });
  assert(deploy.allClear, 'D1: All deployment checks pass');

  // ── Summary ───────────────────────────────────────────────
  console.log('\n═══════════════════════════════════════════════════');
  console.log(`  CHECK 8 RESULT: ${passed}/${total} passed, ${failed} failed`);
  if (failed === 0) {
    console.log('  *** GATE PASSED ***');
  } else {
    console.log('  *** GATE FAILED ***');
  }
  console.log('═══════════════════════════════════════════════════');

  process.exit(failed === 0 ? 0 : 1);
})();
