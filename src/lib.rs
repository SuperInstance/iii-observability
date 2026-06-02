//! # iii-observability
//!
//! *Threshold alerts lie. Your service was already broken when the alert fired.
//! Information-theoretic observability detects the drift BEFORE it crosses a line.*
//!
//! An SRE observability toolkit built on top of [`crackle-runtime`] that uses
//! information-theoretic measures — KL divergence, Jensen-Shannon divergence,
//! transfer entropy, and mutual information — to detect service anomalies before
//! they become incidents.
//!
//! Instead of static thresholds (latency > 500ms → alert), this crate measures
//! *distribution shifts* in your metrics. A service whose latency distribution
//! drifts from its baseline is already telling you something — before any single
//! request crosses a limit.
//!
//! ## Architecture
//!
//! The crate builds on crackle-runtime's firing/cooling metaphor:
//!
//! - **Firing**: Each service metric observation is a task that produces named
//!   metrics (latency, error rate, throughput, queue depth, etc.)
//! - **Cooling**: After collecting a window of observations, the runtime detects
//!   patterns using information theory
//!
//! ## Quick Start
//!
//! ```rust
//! use iii_observability::{ServiceMetrics, anomaly_score, PhaseDetector, MetricSnapshot};
//!
//! // Build a service metrics collector using crackle-runtime's kiln
//! let mut obs = ServiceMetrics::new("api-gateway");
//!
//! // Record some normal metrics
//! obs.record(MetricSnapshot {
//!     latency_ms: 45.0,
//!     error_rate: 0.01,
//!     throughput: 1000.0,
//!     queue_depth: 5.0,
//! });
//! obs.record(MetricSnapshot {
//!     latency_ms: 52.0,
//!     error_rate: 0.015,
//!     throughput: 950.0,
//!     queue_depth: 8.0,
//! });
//!
//! // Cool the kiln — detect patterns
//! let patterns = obs.cool();
//!
//! // Check for anomalies using information theory
//! let baseline = vec![45.0, 52.0, 48.0, 51.0, 47.0];
//! let current = vec![45.0, 52.0, 48.0, 51.0, 47.0];
//! let score = anomaly_score(&baseline, &current, 10);
//! assert!(score.jsd < 0.1); // identical distributions → near 0
//! ```

mod info;
mod metrics;
mod anomaly;
mod phase;

pub use metrics::{ServiceMetrics, MetricSnapshot};
pub use anomaly::{anomaly_score, AnomalyScore, transfer_causality, causality_matrix, correlation_matrix};
pub use phase::PhaseDetector;
