//! # Basic Drift Detection
//!
//! This example demonstrates how iii-observability detects performance drift
//! before it crosses traditional threshold boundaries.
//!
//! ## Scenario
//!
//! An API service's latency gradually increases from 45ms to 95ms over a window
//! of observations. Threshold alerts at 100ms won't fire — but distribution-based
//! monitoring detects the drift immediately.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example basic_drift_detection
//! ```

use iii_observability::{anomaly_score, PhaseDetector, ServiceMetrics, MetricSnapshot};

fn main() {
    println!("=== Basic Drift Detection ===\n");

    // ── Phase 1: Establish baseline ──────────────────────────────────

    println!("Phase 1: Establishing baseline (stable operation)...\n");

    let mut obs = ServiceMetrics::new("api-gateway");

    // Record 50 snapshots during normal operation
    for i in 0..50 {
        let jitter: f64 = (i as f64).sin() * 2.0; // natural ±2ms variation
        obs.record(MetricSnapshot {
            latency_ms: 45.0 + jitter,
            error_rate: 0.01,
            throughput: 1000.0,
            queue_depth: 3.0,
        })
        .expect("recording");
    }

    // Collect baseline latencies
    let baseline_latencies: Vec<f64> = (0..50)
        .map(|i| 45.0 + (i as f64).sin() * 2.0)
        .collect();

    println!("  Baseline latency:  ~45ms (±2ms)");
    println!("  Baseline samples:  {}\n", baseline_latencies.len());

    // ── Phase 2: Gradual drift ──────────────────────────────────────

    println!("Phase 2: Introducing gradual latency drift...\n");

    // Simulate a memory leak causing latency to slowly climb
    let mut current_latencies: Vec<f64> = Vec::new();

    for i in 0..50 {
        let drift = (i as f64) * 1.0; // +1ms per sample
        let jitter: f64 = (i as f64).cos() * 3.0;
        let latency = 45.0 + drift + jitter;
        current_latencies.push(latency);
    }

    // ── Phase 3: Compute anomaly scores ─────────────────────────────

    println!("Phase 3: Computing anomaly scores...\n");

    let score = anomaly_score(&current_latencies, &baseline_latencies, 20);

    println!("  Results:");
    println!("    KL divergence:   {:.6}", score.kl);
    println!("    JSD:             {:.6}", score.jsd);
    println!("    Normalized:      {:.6}", score.normalized);
    println!();
    println!("  Interpretation:");

    if score.normalized > 0.3 {
        println!("    ⚠ SIGNIFICANT DRIFT DETECTED");
        println!("    The latency distribution has shifted meaningfully.");
        println!("    Recommended action: investigate memory/CPU usage.");
    } else if score.normalized > 0.1 {
        println!("    ⚡ EARLY DRIFT DETECTED");
        println!("    The distribution is starting to shift — before any");
        println!("    single request approaches a threshold limit.");
    } else {
        println!("    ✅ System is nominal. No significant drift.");
    }

    println!();

    // ── Phase 4: Traditional threshold comparison ──────────────────

    println!("Phase 4: Threshold comparison...\n");

    let max_current = current_latencies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let threshold_ms = 100.0;

    println!("  Current max latency: {:.1}ms", max_current);
    println!("  Alert threshold:     {:.0}ms", threshold_ms);

    if max_current < threshold_ms {
        println!();
        println!("  ❌ Traditional threshold: NO ALERT (latency below {:.0}ms)", threshold_ms);
        println!("     BUT the distribution has already drifted by {:.1}%!",
            score.normalized * 100.0);
        println!();
        println!("  ✅ Info-theoretic: DETECTED DRIFT at normalized = {:.4}", score.normalized);
        println!();
        println!("  This is the key insight: threshold alerts fire AFTER the damage.");
        println!("  Distribution-based detection catches the drift as it happens.");
    }

    // ── Phase 5: Full service lifecycle ─────────────────────────────

    println!("\nPhase 5: Full service lifecycle with state detection...\n");

    // Cool the kiln and check system state
    let patterns = obs.cool();
    let state = PhaseDetector::aggregate_state(&patterns);

    println!("  Patterns detected: {}", patterns.len());
    println!("  System state:      {:?}", state);
    println!();
    println!("  Summary:");
    println!("{}", PhaseDetector::summarize(&patterns));

    if state.requires_attention() {
        println!();
        println!("  ⚠ Action required: Investigate the root cause of drift.");
    } else {
        println!("  ✅ System is healthy.");
    }
}
