/**
 * Regression test runner — standalone entry point for npm run regression.
 *
 * Runs inference benchmark, accuracy check, signaling load test, and
 * deployment checklist independently of the E2E scenarios.
 *
 * Usage:  node packages/e2e/test/regression_test.js
 */

'use strict';

const {
  runInferenceBenchmark,
  runAccuracyCheck,
  runSignalingLoadTest,
  runDeploymentChecklist,
} = require('../RegressionSuite');

let passed = 0, failed = 0, total = 0;

function assert(condition, label) {
  total++;
  if (condition) { passed++; console.log(`  [PASS] ${label}`); }
  else           { failed++; console.log(`  [FAIL] ${label}`); }
}

console.log('╔═══════════════════════════════════════════════════╗');
console.log('║       Performance Regression Suite                ║');
console.log('╚═══════════════════════════════════════════════════╝\n');

// R1: Inference
console.log('─── R1: Inference benchmark ───');
const inf = runInferenceBenchmark(1000);
console.log(`    Mean: ${inf.meanMs} ms  |  P50: ${inf.p50Ms} ms  |  P95: ${inf.p95Ms} ms  |  P99: ${inf.p99Ms} ms`);
assert(inf.meanMs < 65, 'R1: Inference mean < 65 ms');
assert(inf.p99Ms < 90,  'R1: Inference P99 < 90 ms');

// R2: Accuracy
console.log('\n─── R2: Accuracy check ───');
const acc = runAccuracyCheck(2000);
console.log(`    Top-1: ${(acc.top1Accuracy * 100).toFixed(1)}%  |  F1: ${acc.f1Score}  |  Precision: ${acc.precision}  |  Recall: ${acc.recall}`);
assert(acc.top1Accuracy > 0.90, 'R2: Top-1 accuracy > 90%');
assert(acc.f1Score > 0.85,      'R2: F1 score > 0.85');

// R3: Signaling load
console.log('\n─── R3: Signaling load test (100 rooms, 60s) ───');
const load = runSignalingLoadTest(100, 60);
console.log(`    Messages: ${load.totalMessages}  |  Avg: ${load.avgLatencyMs} ms  |  Max: ${load.maxLatencyMs} ms  |  Failures: ${load.failures}`);
assert(load.stable, 'R3: Signaling stable (< 0.1% failure)');

// D1: Deployment checklist
console.log('\n─── D1: Deployment checklist ───');
const deploy = runDeploymentChecklist();
deploy.items.forEach(i => console.log(`    [${i.status}] ${i.check}`));
assert(deploy.allClear, 'D1: All deployment checks pass');

// Summary
console.log(`\n  REGRESSION RESULT: ${passed}/${total} passed, ${failed} failed`);
if (failed === 0) console.log('  *** ALL REGRESSION CHECKS PASSED ***');
else              console.log('  *** REGRESSION FAILED ***');
process.exit(failed === 0 ? 0 : 1);
