# iii-observability

> *Threshold alerts lie. Your service was already broken when the alert fired.
> Information-theoretic observability detects the drift BEFORE it crosses a line.*

**Information-theoretic observability for SRE**, built on [`crackle-runtime`](https://github.com/SuperInstance/crackle-runtime).

## Why?

Static thresholds like `latency > 500ms → PagerDuty` are **reactive**. By the time the alert fires, your users have already felt the pain.

**Information-theoretic measures detect the drift BEFORE it crosses a line:**

| Measure | What It Detects |
|---------|----------------|
| **KL Divergence** | How much info is lost using yesterday's baseline for today's metrics |
| **Jensen-Shannon Divergence** | Smooth, bounded [0, 1] anomaly score — 0 = identical, 1 = completely different |
| **Transfer Entropy** | *"Service A's latency is CAUSING service B's errors"* — directional causality |
| **Mutual Information** | Which services are moving together (Pearson misses non-linear relationships) |

## Quick Start

```rust
use iii_observability::{
    ServiceMetrics, MetricSnapshot, anomaly_score,
    PhaseDetector, transfer_causality,
};
use crackle_runtime::PatternKind;

// 1. Collect service metrics
let mut api = ServiceMetrics::new("api-gateway");

api.record(MetricSnapshot {
    latency_ms: 42.0, error_rate: 0.01,
    throughput: 1000.0, queue_depth: 5.0,
});
api.record(MetricSnapshot {
    latency_ms: 45.0, error_rate: 0.012,
    throughput: 980.0, queue_depth: 7.0,
});
// ... more observations ...

// 2. Cool and detect patterns
let patterns = api.cool();

// 3. Check the operational state
let state = PhaseDetector::aggregate_state(&patterns);
println!("State: {}", state);
// → "nominal", "degrading", or "recovered"

// 4. Compare current metrics against baseline
let baseline = vec![42.0, 45.0, 39.0, 48.0, 43.0]; // last week
let current  = vec![58.0, 62.0, 55.0, 70.0, 61.0]; // right now

let score = anomaly_score(&current, &baseline, 10);
println!("KL: {:.4}, JSD: {:.4}", score.kl, score.jsd);

if score.jsd > 0.1 {
    println!("🚨 Anomaly detected! Distributions have shifted.");
}

// 5. Detect causal relationships
let causality = transfer_causality(
    &a_latency,  // source: service A's latency
    &b_errors,   // target: service B's error rate
    1,           // lag
    10,          // bins
);
if causality > 0.0 {
    println!("Service A's latency is causally influencing Service B's errors");
}
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  iii-observability                    │
│                                                      │
│  ┌────────────────┐   ┌───────────────────────────┐  │
│  │ ServiceMetrics  │   │   Information Theory       │  │
│  │  ┌──────────┐   │   │  ├─ KL Divergence          │  │
│  │  │ Kiln     │   │   │  ├─ Jensen-Shannon         │  │
│  │  │ [tasks]  │───┼──>│  ├─ Transfer Entropy       │  │
│  │  └──────────┘   │   │  └─ Mutual Information     │  │
│  └────────────────┘   └───────────────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ PhaseDetector                                  │    │
│  │  ├─ PhaseTransition → Degrading               │    │
│  │  ├─ Clustering → Degrading                    │    │
│  │  ├─ Correlation → Degrading                   │    │
│  │  └─ Conservation → Recovered                  │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## SRE State Mapping

| `PatternKind` | SRE State | Meaning |
|---|---|---|
| `PhaseTransition` | `Degrading` | Service distribution has shifted — this IS degradation |
| `Clustering` | `Degrading` | Metrics splitting into groups = instability |
| `Correlation` | `Degrading` | Unexpected metric coupling = cascading effects |
| `Conservation` | `Recovered` | Metrics stable and consistent = service healed |

The `PhaseDetector::aggregate_state()` is conservative:
- **Any** degrading pattern → overall state is `Degrading`
- All recovery patterns → `Recovered`
- No patterns → `Nominal`

## Anomaly Score Interpretation

| JSD | Level | Action |
|-----|-------|--------|
| < 0.01 | Normal | Distributions identical. No action needed. |
| 0.01–0.1 | Drift | Distributions shifted slightly. Worth investigating. |
| 0.1–0.5 | Anomalous | Significant change. Check dashboards. |
| > 0.5 | Critical | Completely different distribution. Incident. |

## API Reference

### `ServiceMetrics`

Collects metric snapshots and uses crackle-runtime's kiln for pattern detection.

```rust
ServiceMetrics::new("service-name")
ServiceMetrics::with_profile("service-name", ThermalProfile::slow_cooling())
.record(MetricSnapshot { ... })  // Add an observation
.cool()                          // Detect patterns → Vec<CracklePattern>
.reset()                         // Start a new observation window
```

### `anomaly_score()`

```rust
fn anomaly_score(current: &[f64], baseline: &[f64], bins: usize) -> AnomalyScore
```

Returns KL divergence, JSD, and normalized [0, 1] score.

### `transfer_causality()`

```rust
fn transfer_causality(source: &[f64], target: &[f64], lag: usize, bins: usize) -> f64
```

Directional causal influence: `source → target`. Higher = stronger evidence that source is driving target.

### `causality_matrix()`

```rust
fn causality_matrix(metrics: &[(String, Vec<f64>)], lag: usize, bins: usize) -> Vec<Vec<f64>>
```

N×N matrix — entry `[i][j]` = transfer entropy from metric i to metric j.

### `correlation_matrix()`

```rust
fn correlation_matrix(metrics: &[(String, Vec<f64>)], bins: usize) -> Vec<Vec<f64>>
```

Symmetric N×N matrix using mutual information. Captures non-linear dependencies.

### `PhaseDetector`

Maps crackle-runtime patterns to SRE states:

```rust
PhaseDetector::pattern_to_state(&PatternKind::PhaseTransition)  // → Degrading
PhaseDetector::aggregate_state(&patterns)                        // → SreState
PhaseDetector::summarize(&patterns)                              // → String summary
```

## License

MIT
