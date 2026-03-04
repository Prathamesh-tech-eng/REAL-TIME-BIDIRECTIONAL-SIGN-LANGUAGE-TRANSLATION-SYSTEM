"""
evaluate.py
SignBridge ML — Model Evaluation & CHECK-2 Validation

Measures all CHECK-2 metrics against required thresholds:
  1. Top-1 accuracy   ≥ 90 %
  2. Top-5 accuracy   ≥ 97 %
  3. Mean inference    ≤ 65 ms
  4. Model size (int8) ≤ 8 MB
  5. Macro-F1 score    ≥ 0.85
  6. False Acceptance  ≤ 5 %
  7. Peak memory       ≤ 80 MB

Usage:
  python evaluate.py \
    --model models/tms_wlasl100_int8.tflite \
    --dataset data/processed \
    --report models/check2_report.json
"""

import argparse
import json
import os
import sys
import time
import tracemalloc
import numpy as np
from pathlib import Path


# ====================================================================
# CHECK-2 Thresholds (from master execution document Table 6)
# ====================================================================
THRESHOLDS = {
    'top1_accuracy':      0.90,
    'top5_accuracy':      0.97,
    'mean_inference_ms':  65.0,
    'model_size_mb':      8.0,
    'macro_f1':           0.85,
    'false_acceptance':   0.05,
    'peak_memory_mb':     80.0,
}


# ====================================================================
# Metric helpers
# ====================================================================

def top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute top-k accuracy from integer labels and probabilities."""
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return float(correct.mean())


def macro_f1_score(y_true: np.ndarray, y_pred_classes: np.ndarray,
                   n_classes: int) -> float:
    """Compute macro-averaged F1 without sklearn dependency."""
    f1_scores = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred_classes == c) & (y_true == c)))
        fp = int(np.sum((y_pred_classes == c) & (y_true != c)))
        fn = int(np.sum((y_pred_classes != c) & (y_true == c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def false_acceptance_rate(y_true: np.ndarray, y_pred_probs: np.ndarray,
                          threshold: float = 0.5) -> float:
    """
    False Acceptance Rate — fraction of *incorrect* predictions whose
    confidence exceeds `threshold`.  Measures overconfident errors.
    """
    pred_classes = np.argmax(y_pred_probs, axis=1)
    max_conf = np.max(y_pred_probs, axis=1)
    incorrect_mask = pred_classes != y_true
    n_incorrect = int(np.sum(incorrect_mask))
    if n_incorrect == 0:
        return 0.0
    false_accepts = int(np.sum(max_conf[incorrect_mask] >= threshold))
    return false_accepts / len(y_true)


# ====================================================================
# Main evaluation
# ====================================================================

def evaluate(model_path: str, dataset_dir: str,
             report_path: str | None = None):
    """
    Run full CHECK-2 evaluation on a TFLite model.

    Returns:
        dict with all metrics and pass/fail status.
    """
    import tensorflow as tf

    dataset_path = Path(dataset_dir)

    # ---- Load test data ----
    test_dir = dataset_path / 'test'
    X_test = np.load(test_dir / 'sequences.npy').astype(np.float32)
    y_test = np.load(test_dir / 'labels.npy').astype(np.int32)

    with open(dataset_path / 'label_map.json', 'r') as f:
        label_map = json.load(f)
    n_classes = len(label_map)

    print(f"Test set : {X_test.shape[0]} samples, {n_classes} classes")
    print(f"Model    : {model_path}")
    print()

    # ---- Model size ----
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    # ---- Load interpreter ----
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ---- Inference + timing + memory ----
    tracemalloc.start()
    all_probs = []
    latencies = []

    for i in range(len(X_test)):
        sample = X_test[i:i+1]
        interpreter.set_tensor(input_details[0]['index'], sample)

        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # ms
        probs = interpreter.get_tensor(output_details[0]['index'])[0]
        all_probs.append(probs)

    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    all_probs = np.array(all_probs)
    pred_classes = np.argmax(all_probs, axis=1)

    # ---- Compute metrics ----
    metrics = {
        'top1_accuracy':     top_k_accuracy(y_test, all_probs, k=1),
        'top5_accuracy':     top_k_accuracy(y_test, all_probs, k=5),
        'mean_inference_ms': float(np.mean(latencies)),
        'p95_inference_ms':  float(np.percentile(latencies, 95)),
        'model_size_mb':     round(model_size_mb, 3),
        'macro_f1':          macro_f1_score(y_test, pred_classes, n_classes),
        'false_acceptance':  false_acceptance_rate(y_test, all_probs),
        'peak_memory_mb':    round(peak_memory_mb, 2),
        'n_samples':         int(len(y_test)),
        'n_classes':         n_classes,
    }

    # ---- Pass / Fail ----
    checks = {}
    checks['top1_accuracy']     = metrics['top1_accuracy']     >= THRESHOLDS['top1_accuracy']
    checks['top5_accuracy']     = metrics['top5_accuracy']     >= THRESHOLDS['top5_accuracy']
    checks['mean_inference_ms'] = metrics['mean_inference_ms'] <= THRESHOLDS['mean_inference_ms']
    checks['model_size_mb']     = metrics['model_size_mb']     <= THRESHOLDS['model_size_mb']
    checks['macro_f1']          = metrics['macro_f1']          >= THRESHOLDS['macro_f1']
    checks['false_acceptance']  = metrics['false_acceptance']  <= THRESHOLDS['false_acceptance']
    checks['peak_memory_mb']    = metrics['peak_memory_mb']    <= THRESHOLDS['peak_memory_mb']

    all_pass = all(checks.values())

    # ---- Pretty print ----
    print("=" * 62)
    print("  CHECK 2 — Model Validation Results")
    print("=" * 62)
    fmt = "  {:<22} {:>10}  threshold {:<10}  {}"
    print(fmt.format("METRIC", "VALUE", "", "STATUS"))
    print("-" * 62)

    def fmtval(key):
        v = metrics[key]
        t = THRESHOLDS[key]
        ok = checks[key]
        if 'accuracy' in key or 'f1' in key or 'acceptance' in key:
            vstr = f"{v*100:.2f}%"
            tstr = f"{'≥' if 'acceptance' not in key else '≤'}{t*100:.0f}%"
        elif 'ms' in key:
            vstr = f"{v:.1f} ms"
            tstr = f"≤{t:.0f} ms"
        elif 'mb' in key.lower():
            vstr = f"{v:.2f} MB"
            tstr = f"≤{t:.0f} MB"
        else:
            vstr = str(v)
            tstr = str(t)
        status = "PASS" if ok else "** FAIL **"
        return vstr, tstr, status

    for key in THRESHOLDS:
        vstr, tstr, status = fmtval(key)
        print(fmt.format(key, vstr, tstr, status))

    print("-" * 62)
    overall = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"  Overall: {overall}")
    print("=" * 62)

    # ---- Per-class breakdown (top 10 worst) ----
    print("\n  Per-class accuracy (10 worst):")
    per_class_acc = []
    for c in range(n_classes):
        mask = y_test == c
        if mask.sum() == 0:
            continue
        acc = float((pred_classes[mask] == c).mean())
        per_class_acc.append((label_map.get(str(c), str(c)), acc, int(mask.sum())))

    per_class_acc.sort(key=lambda x: x[1])
    for name, acc, count in per_class_acc[:10]:
        print(f"    {name:<20} {acc*100:6.2f}%  (n={count})")

    # ---- Save report ----
    report = {
        'model': model_path,
        'dataset': dataset_dir,
        'metrics': metrics,
        'thresholds': {k: v for k, v in THRESHOLDS.items()},
        'checks': checks,
        'all_pass': all_pass,
    }

    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to: {report_path}")

    return report


# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SignBridge CHECK-2 Evaluation')
    parser.add_argument('--model', required=True,
                        help='Path to .tflite model')
    parser.add_argument('--dataset', required=True,
                        help='Path to preprocessed dataset directory')
    parser.add_argument('--report', default=None,
                        help='Path for JSON report output')
    args = parser.parse_args()

    report = evaluate(args.model, args.dataset, args.report)

    # Exit code: 0 if all pass, 1 if any fail
    sys.exit(0 if report['all_pass'] else 1)
