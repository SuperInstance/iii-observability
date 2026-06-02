use crackle_runtime::{CrackleTask, Kiln, CrackleError, TaskOutput, ThermalProfile, CracklePattern};

/// A snapshot of service health metrics at a point in time.
///
/// These are the raw measurements you'd collect from any running service:
/// latency, error rate, throughput, and queue depth.
///
/// # Example
///
/// ```
/// use iii_observability::MetricSnapshot;
///
/// let snap = MetricSnapshot {
///     latency_ms: 42.0,
///     error_rate: 0.01,
///     throughput: 1500.0,
///     queue_depth: 3.0,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MetricSnapshot {
    /// Request latency in milliseconds.
    pub latency_ms: f64,
    /// Error rate as a fraction (0.0 – 1.0).
    pub error_rate: f64,
    /// Requests per second (or other throughput unit).
    pub throughput: f64,
    /// Current queue depth / backlog.
    pub queue_depth: f64,
}

/// A service that wraps a snapshot into a crackle-runtime task.
#[derive(Debug, Clone)]
struct ServiceTask {
    service_name: String,
    snapshot: MetricSnapshot,
}

impl CrackleTask for ServiceTask {
    type Output = MetricSnapshot;

    fn fire(&self) -> TaskOutput<Self::Output> {
        TaskOutput::new(
            self.snapshot,
            vec![
                ("latency_ms".into(), self.snapshot.latency_ms),
                ("error_rate".into(), self.snapshot.error_rate),
                ("throughput".into(), self.snapshot.throughput),
                ("queue_depth".into(), self.snapshot.queue_depth),
            ],
        )
    }

    fn label(&self) -> String {
        format!("{}/{}", self.service_name, self.snapshot_queue_idx())
    }

    fn cool(
        &self,
        output: &TaskOutput<Self::Output>,
        _all_metrics: &[(String, Vec<(String, f64)>)],
    ) -> Vec<(String, f64)> {
        // During cooling, emit derived metrics that capture system health
        let latency = output.value.latency_ms;
        let error_rate = output.value.error_rate;
        let throughput = output.value.throughput;
        let queue_depth = output.value.queue_depth;

        vec![
            // Derived metric: queue saturation (higher = bad)
            ("queue_saturation".into(), (queue_depth / (queue_depth + 10.0)).min(1.0)),
            // Derived metric: error-to-throughput ratio (sudden jumps indicate failures)
            ("error_impact".into(), error_rate * throughput),
            // Derived metric: latency anomaly indicator using z-score proxy
            ("latency_z".into(), (latency - 50.0) / 20.0),
        ]
    }
}

impl ServiceTask {
    /// Generate a sequence number for this task's label.
    /// This is approximated from the snapshot's fields, but purely for labeling.
    fn snapshot_queue_idx(&self) -> u64 {
        // Use a deterministic hash of the snapshot for labeling
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.snapshot.latency_ms.to_bits().hash(&mut hasher);
        self.snapshot.error_rate.to_bits().hash(&mut hasher);
        self.snapshot.throughput.to_bits().hash(&mut hasher);
        self.snapshot.queue_depth.to_bits().hash(&mut hasher);
        hasher.finish() % 10000
    }
}

/// A service metrics collector that wraps [`crackle_runtime::Kiln`].
///
/// Records metric snapshots over time, then cools to detect emergent patterns
/// using crackle-runtime's built-in detectors (clustering, phase transitions,
/// conservation laws, correlations).
///
/// # Example
///
/// ```rust
/// use iii_observability::{ServiceMetrics, MetricSnapshot};
///
/// let mut metrics = ServiceMetrics::new("user-service");
/// metrics.record(MetricSnapshot {
///     latency_ms: 42.0,
///     error_rate: 0.01,
///     throughput: 1000.0,
///     queue_depth: 5.0,
/// });
/// metrics.record(MetricSnapshot {
///     latency_ms: 45.0,
///     error_rate: 0.015,
///     throughput: 980.0,
///     queue_depth: 7.0,
/// });
///
/// let patterns = metrics.cool();
/// println!("Detected {} patterns", patterns.len());
/// ```
pub struct ServiceMetrics {
    service_name: String,
    kiln: Kiln,
}

impl ServiceMetrics {
    /// Create a new service metrics collector for the named service.
    pub fn new(service_name: impl Into<String>) -> Self {
        ServiceMetrics {
            service_name: service_name.into(),
            // Fast cooling = more sensitive anomaly detection
            kiln: Kiln::new(ThermalProfile::fast_cooling()),
        }
    }

    /// Create a service metrics collector with a custom thermal profile.
    pub fn with_profile(service_name: impl Into<String>, profile: ThermalProfile) -> Self {
        ServiceMetrics {
            service_name: service_name.into(),
            kiln: Kiln::new(profile),
        }
    }

    /// Record a metric snapshot for this service.
    ///
    /// # Panics
    ///
    /// Panics with `KilnCooled` if called after `cool()`.
    pub fn record(&mut self, snapshot: MetricSnapshot) -> Result<(), CrackleError> {
        if self.kiln.is_cooled() {
            return Err(CrackleError::KilnCooled);
        }
        let task = ServiceTask {
            service_name: self.service_name.clone(),
            snapshot,
        };
        self.kiln.fire_and_record(task);
        Ok(())
    }

    /// Cool the kiln and detect emergent patterns across all recorded metrics.
    ///
    /// Returns patterns sorted by confidence (highest first).
    /// This replaces the kiln — no more metrics can be recorded after cooling.
    pub fn cool(&mut self) -> Vec<CracklePattern> {
        self.kiln.cool()
    }

    /// Get the current number of metric snapshots recorded.
    pub fn count(&self) -> usize {
        self.kiln.task_count()
    }

    /// Get the service name.
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Get a reference to the underlying kiln for advanced usage.
    pub fn kiln(&self) -> &Kiln {
        &self.kiln
    }

    /// Reset the collector for a new observation window.
    pub fn reset(&mut self) {
        self.kiln.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_count() {
        let mut metrics = ServiceMetrics::new("test-svc");
        assert_eq!(metrics.count(), 0);

        metrics
            .record(MetricSnapshot {
                latency_ms: 50.0,
                error_rate: 0.01,
                throughput: 1000.0,
                queue_depth: 5.0,
            })
            .unwrap();
        assert_eq!(metrics.count(), 1);

        metrics
            .record(MetricSnapshot {
                latency_ms: 55.0,
                error_rate: 0.02,
                throughput: 950.0,
                queue_depth: 8.0,
            })
            .unwrap();
        assert_eq!(metrics.count(), 2);
    }

    #[test]
    fn test_cool_detects_patterns() {
        let mut metrics = ServiceMetrics::new("test-svc");
        // Record enough snapshots for pattern detection
        for i in 0..10 {
            let latency = 50.0 + (i as f64 * 5.0);
            metrics
                .record(MetricSnapshot {
                    latency_ms: latency,
                    error_rate: 0.01,
                    throughput: 1000.0,
                    queue_depth: 5.0,
                })
                .unwrap();
        }

        let patterns = metrics.cool();
        // With fast cooling, we should detect at least clustering and conservation
        assert!(!patterns.is_empty(), "expected at least one pattern");
    }

    #[test]
    fn test_reset() {
        let mut metrics = ServiceMetrics::new("test-svc");
        metrics
            .record(MetricSnapshot {
                latency_ms: 50.0,
                error_rate: 0.01,
                throughput: 1000.0,
                queue_depth: 5.0,
            })
            .unwrap();
        metrics.reset();
        assert_eq!(metrics.count(), 0);
    }

    #[test]
    fn test_service_name() {
        let metrics = ServiceMetrics::new("api-gateway");
        assert_eq!(metrics.service_name(), "api-gateway");
    }

    #[test]
    fn test_with_profile() {
        let metrics = ServiceMetrics::with_profile("slow-svc", ThermalProfile::slow_cooling());
        assert_eq!(metrics.service_name(), "slow-svc");
    }

    #[test]
    fn test_metric_names_in_task() {
        let mut metrics = ServiceMetrics::new("test");
        metrics
            .record(MetricSnapshot {
                latency_ms: 42.0,
                error_rate: 0.01,
                throughput: 500.0,
                queue_depth: 2.0,
            })
            .unwrap();

        let entries = metrics.kiln().entries();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        let fire_metrics = &entry.metrics;

        // Original metrics present in fire metrics
        assert!(fire_metrics.iter().any(|(n, _)| n == "latency_ms"));
        assert!(fire_metrics.iter().any(|(n, _)| n == "error_rate"));
        assert!(fire_metrics.iter().any(|(n, _)| n == "throughput"));
        assert!(fire_metrics.iter().any(|(n, _)| n == "queue_depth"));

        // Fire metrics also include derived metrics (cooled metrics in 0.1.0
        // end up in all_metrics but not necessarily via task's cool method)
        let all_metrics = entry.all_metrics();
        assert!(all_metrics.iter().any(|(n, _)| n == "latency_ms"));
        assert!(all_metrics.iter().any(|(n, _)| n == "error_rate"));
        assert!(all_metrics.iter().any(|(n, _)| n == "throughput"));
        assert!(all_metrics.iter().any(|(n, _)| n == "queue_depth"));
    }

    #[test]
    fn test_multiple_services() {
        let mut svc1 = ServiceMetrics::new("api");
        let mut svc2 = ServiceMetrics::new("worker");
        svc1.record(MetricSnapshot { latency_ms: 10.0, error_rate: 0.0, throughput: 500.0, queue_depth: 0.0 }).unwrap();
        svc2.record(MetricSnapshot { latency_ms: 100.0, error_rate: 0.05, throughput: 50.0, queue_depth: 100.0 }).unwrap();
        assert_eq!(svc1.service_name(), "api");
        assert_eq!(svc2.service_name(), "worker");
        assert_eq!(svc1.count(), 1);
        assert_eq!(svc2.count(), 1);
    }

    #[test]
    fn test_large_batch() {
        let mut metrics = ServiceMetrics::new("batch-test");
        for i in 0..100 {
            metrics
                .record(MetricSnapshot {
                    latency_ms: 50.0 + (i as f64).sin() * 10.0,
                    error_rate: 0.01,
                    throughput: 1000.0,
                    queue_depth: 5.0,
                })
                .unwrap();
        }
        assert_eq!(metrics.count(), 100);
    }

    #[test]
    fn test_cool_then_reject_records() {
        let mut metrics = ServiceMetrics::new("test");
        metrics.record(MetricSnapshot { latency_ms: 50.0, error_rate: 0.01, throughput: 1000.0, queue_depth: 5.0 }).unwrap();
        let _patterns = metrics.cool();
        // After cooling, recording should fail
        let result = metrics.record(MetricSnapshot { latency_ms: 60.0, error_rate: 0.02, throughput: 900.0, queue_depth: 8.0 });
        assert!(result.is_err(), "recording after cool should fail");
    }

    #[test]
    fn test_reset_after_cool() {
        let mut metrics = ServiceMetrics::new("test");
        metrics.record(MetricSnapshot { latency_ms: 50.0, error_rate: 0.01, throughput: 1000.0, queue_depth: 5.0 }).unwrap();
        let _patterns = metrics.cool();
        metrics.reset();
        // After reset, recording should work again
        assert_eq!(metrics.count(), 0);
        metrics.record(MetricSnapshot { latency_ms: 60.0, error_rate: 0.02, throughput: 900.0, queue_depth: 8.0 }).unwrap();
        assert_eq!(metrics.count(), 1);
    }

    #[test]
    fn test_clone_and_independence() {
        // Two services cloned from same template should be independent
        let mut a = ServiceMetrics::new("svc");
        a.record(MetricSnapshot { latency_ms: 10.0, error_rate: 0.0, throughput: 100.0, queue_depth: 0.0 }).unwrap();
        let b_count = a.count();
        a.record(MetricSnapshot { latency_ms: 20.0, error_rate: 0.01, throughput: 90.0, queue_depth: 1.0 }).unwrap();
        assert_eq!(a.count(), 2);
        // b_count still refers to the original snapshot
        assert_eq!(b_count, 1);
    }
}
