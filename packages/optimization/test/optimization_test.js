/**
 * CHECK 6 — Performance Optimization & Hardening (Section 8)
 *
 * 7 tests, all must pass for GATE clearance.
 *
 *  Test 1: Simulcast — receiver on throttled 500 kbps → 360p, no buffering
 *  Test 2: Quantization delta (int8 vs fp32) — accuracy drop ≤ 1.5%
 *  Test 3: Zero-copy pipeline — CPU usage drops ≥ 15%
 *  Test 4: Adaptive bitrate — inject 10% packet loss → downscale within 4s
 *  Test 5: OTA update — detect, download, verify SHA256, swap ≤ 60s
 *  Test 6: Accessibility audit — 0 critical/serious, max 2 moderate
 *  Test 7: 30-minute stress test — memory growth < 10 MB, FPS ≥ 28
 */

'use strict';

const path = require('path');

// ── Module imports ──────────────────────────────────────────
const { simulateReceiver, preferCodec, LAYERS } = require(path.join(__dirname, '..', 'SimulcastManager'));
const { verifyQuantization } = require(path.join(__dirname, '..', 'QuantizationVerifier'));
const { benchmarkPipelines } = require(path.join(__dirname, '..', 'ZeroCopyPipeline'));
const { AdaptiveBitrateController } = require(path.join(__dirname, '..', 'AdaptiveBitrate'));
const { OTAManager } = require(path.join(__dirname, '..', 'OTAModelUpdate'));
const { runAudit, getAppComponentTree } = require(path.join(__dirname, '..', 'AccessibilityAudit'));
const { runStressTest } = require(path.join(__dirname, '..', 'StressTest'));

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
  console.log('║    CHECK 6 — Optimization Verification (7 tests)   ║');
  console.log('╚══════════════════════════════════════════════════════╝');

  // ────────────────────────────────────────────────────────
  // TEST 1: Simulcast — 500 kbps → 360p, no buffering
  // ────────────────────────────────────────────────────────
  header('Test 1 — Simulcast (500 kbps → 360p, no buffering)');
  {
    const result = simulateReceiver(500, 10);

    console.log(`    Selected layer    = ${result.selectedLayer} (${result.resolution})`);
    console.log(`    Bitrate           = ${result.bitrateKbps} kbps`);
    console.log(`    Buffering events  = ${result.bufferingEvents}`);
    console.log(`    Avg FPS           = ${result.avgFps}`);

    ok(result.selectedLayer === 'low', `Layer selected: ${result.selectedLayer} (expected low/360p)`);
    ok(result.bufferingEvents === 0, `Buffering events: ${result.bufferingEvents} (expected 0)`);
  }

  // ────────────────────────────────────────────────────────
  // TEST 2: Quantization delta — accuracy drop ≤ 1.5%
  // ────────────────────────────────────────────────────────
  header('Test 2 — Quantization Delta (int8 vs fp32 ≤ 1.5%)');
  {
    const result = verifyQuantization(500);

    console.log(`    fp32 accuracy     = ${(result.fp32.accuracy * 100).toFixed(2)}%`);
    console.log(`    int8 accuracy     = ${(result.int8.accuracy * 100).toFixed(2)}%`);
    console.log(`    Delta             = ${result.accuracyDeltaPercent.toFixed(2)}%`);
    console.log(`    Speedup           = ${result.speedupFactor}x`);
    console.log(`    int8 model size   = ${result.int8.modelSizeMB} MB`);

    ok(result.withinThreshold, `Accuracy delta ${result.accuracyDeltaPercent.toFixed(2)}% ≤ 1.5%`);
  }

  // ────────────────────────────────────────────────────────
  // TEST 3: Zero-copy pipeline — CPU reduction ≥ 15%
  // ────────────────────────────────────────────────────────
  header('Test 3 — Zero-Copy Pipeline (CPU reduction ≥ 15%)');
  {
    const result = benchmarkPipelines(30);

    console.log(`    Naive CPU usage   = ${result.naive.cpuUsagePercent}%`);
    console.log(`    Zero-copy CPU     = ${result.zeroCopy.cpuUsagePercent}%`);
    console.log(`    CPU reduction     = ${result.cpuReductionPercent}%`);
    console.log(`    Naive FPS         = ${result.naive.throughputFps}`);
    console.log(`    Zero-copy FPS     = ${result.zeroCopy.throughputFps}`);

    ok(result.cpuReductionPercent >= 15,
      `CPU reduction ${result.cpuReductionPercent}% ≥ 15%`);
  }

  // ────────────────────────────────────────────────────────
  // TEST 4: Adaptive bitrate — 10% loss → downscale within 4s
  // ────────────────────────────────────────────────────────
  header('Test 4 — Adaptive Bitrate (10% loss → downscale ≤ 4s)');
  {
    const controller = new AdaptiveBitrateController();

    // Timeline: normal for 6s, then inject 10% loss at 6s, recover at 20s
    const events = [
      { timeMs: 6000, lossRate: 0.10 },
      { timeMs: 20000, lossRate: 0.01 },
    ];

    const result = controller.simulate(events, 30000);

    console.log(`    Downscaled in time= ${result.downscaledWithinDeadline}`);
    console.log(`    Downscale latency = ${result.downscaleLatencyMs ?? 'N/A'} ms`);
    console.log(`    Frozen during     = ${result.frozenDuringAdaptation}`);
    console.log(`    Final layer       = ${result.finalLayer}`);

    ok(result.downscaledWithinDeadline, 'Downscaled within 4s deadline');
    ok(!result.frozenDuringAdaptation, 'No freeze during adaptation');
  }

  // ────────────────────────────────────────────────────────
  // TEST 5: OTA Update — detect, download, verify, swap ≤ 60s
  // ────────────────────────────────────────────────────────
  header('Test 5 — OTA Model Update (≤ 60s, SHA256 verified)');
  {
    const manager = new OTAManager('1.0.0');
    const result = await manager.checkAndUpdate('1.0.3');

    console.log(`    Update available  = ${result.updateAvailable}`);
    console.log(`    Downloaded        = ${result.downloaded}`);
    console.log(`    SHA256 verified   = ${result.verified}`);
    console.log(`    Hot-swapped       = ${result.swapped}`);
    console.log(`    Total time        = ${result.totalTimeMs.toFixed(1)} ms`);

    for (const step of result.steps) {
      console.log(`      [${step.step}] ${step.timeMs.toFixed(1)} ms — ${step.success ? 'OK' : 'FAIL'}${step.detail ? ' (' + step.detail + ')' : ''}`);
    }

    ok(result.swapped, 'Model hot-swapped successfully (no restart)');
    ok(result.verified, 'SHA256 verification passed');
    ok(result.totalTimeMs < 60000, `Total time ${result.totalTimeMs.toFixed(1)} ms < 60000 ms`);
  }

  // ────────────────────────────────────────────────────────
  // TEST 6: Accessibility audit — 0 critical/serious, max 2 moderate
  // ────────────────────────────────────────────────────────
  header('Test 6 — Accessibility Audit (axe-core rules)');
  {
    const tree = getAppComponentTree();
    const result = runAudit(tree);

    console.log(`    Passes            = ${result.passes}`);
    console.log(`    Critical          = ${result.critical}`);
    console.log(`    Serious           = ${result.serious}`);
    console.log(`    Moderate          = ${result.moderate}`);
    console.log(`    Minor             = ${result.minor}`);

    if (result.violations.length > 0) {
      console.log('    Violations:');
      for (const v of result.violations) {
        console.log(`      [${v.severity}] ${v.rule}: ${v.detail}`);
      }
    }

    ok(result.critical === 0, `Critical violations: ${result.critical} (expected 0)`);
    ok(result.serious === 0, `Serious violations: ${result.serious} (expected 0)`);
    ok(result.moderate <= 2, `Moderate violations: ${result.moderate} (max 2)`);
  }

  // ────────────────────────────────────────────────────────
  // TEST 7: 30-minute stress test — memory < 10 MB growth, FPS ≥ 28
  // ────────────────────────────────────────────────────────
  header('Test 7 — 30-Minute Stress Test (memory & FPS)');
  {
    const result = await runStressTest(30, 30);

    console.log(`    Duration          = ${result.durationMinutes} min`);
    console.log(`    Baseline heap     = ${result.baselineHeapMB} MB`);
    console.log(`    Max heap          = ${result.maxHeapMB} MB`);
    console.log(`    Memory growth     = ${result.maxMemoryGrowthMB} MB  (threshold < 10 MB)`);
    console.log(`    Min FPS           = ${result.minFps}  (threshold ≥ 28)`);
    console.log(`    Avg FPS           = ${result.avgFps}`);

    ok(result.maxMemoryGrowthMB <= 10, `Memory growth ${result.maxMemoryGrowthMB} MB ≤ 10 MB`);
    ok(result.minFps >= 28, `Min FPS ${result.minFps} ≥ 28`);
  }

  // ── Summary ────────────────────────────────────────────
  console.log('\n══════════════════════════════════════════════════════');
  console.log(`  CHECK 6 RESULT: ${passed} passed, ${failed} failed`);
  console.log(`  GATE: ${failed === 0 ? 'PASSED ✓' : 'FAILED ✗'}`);
  console.log('══════════════════════════════════════════════════════\n');

  process.exit(failed === 0 ? 0 : 1);
}

runTests().catch(err => {
  console.error('CHECK 6 FATAL:', err);
  process.exit(1);
});
