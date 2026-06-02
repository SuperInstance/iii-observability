use crate::info::{jsd, kl_divergence, mutual_information, transfer_entropy};

/// An anomaly score comparing a current metric distribution against a baseline.
///
/// Uses two complementary information-theoretic measures:
///
/// - **KL divergence**: Measures how much information is lost when the baseline
///   distribution is used to approximate the current distribution. Asymmetric,
///   good for detecting when the current distribution has *new* modes (values
///   that weren't present in the baseline).
///
/// - **JSD (Jensen-Shannon divergence)**: Symmetric, always finite, and its
///   square root is a proper metric. Provides a smooth anomaly score in [0, 1].
///
/// # Interpretation
///
/// | Score | Meaning |
/// |-------|---------|
/// | <0.01 | Normal — distributions are essentially identical |
/// | 0.01–0.1 | Drift — distribution shifted slightly, worth investigating |
/// | 0.1–0.5 | Anomalous — significant distribution change |
/// | >0.5 | Critical — completely different distribution |
///
/// # Example
///
/// ```rust
/// use iii_observability::anomaly_score;
///
/// let baseline = vec![45.0, 48.0, 50.0, 52.0, 55.0];
/// // Slight latency increase
/// let current = vec![60.0, 62.0, 58.0, 65.0, 55.0];
/// let score = anomaly_score(&current, &baseline, 10);
/// println!("KL: {:.4}, JSD: {:.4}", score.kl, score.jsd);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AnomalyScore {
    /// Kullback-Leibler divergence (asymmetric, can be infinite).
    /// High values mean the current distribution has values the baseline never saw.
    pub kl: f64,
    /// Jensen-Shannon divergence (symmetric, bounded, always finite).
    /// Smooth measure of distribution shift.
    pub jsd: f64,
    /// Normalized anomaly score in [0, 1] based on JSD.
    /// 0 = identical distributions, 1 = completely different.
    pub normalized: f64,
}

impl AnomalyScore {
    /// Create a new anomaly score with KL divergence and JSD.
    pub fn new(kl: f64, jsd: f64) -> Self {
        let max_jsd = 1.0_f64.max(jsd);
        let normalized = if max_jsd > 0.0 {
            (jsd / max_jsd).min(1.0)
        } else {
            0.0
        };
        AnomalyScore { kl, jsd, normalized }
    }
}

/// Compute information-theoretic anomaly scores between current and baseline metrics.
///
/// Returns both KL divergence and Jensen-Shannon divergence.
///
/// # Arguments
///
/// * `current` - Current metric values (e.g., latency readings from the last 5 minutes)
/// * `baseline` - Baseline metric values (e.g., latency readings from the same time yesterday)
/// * `bins` - Number of bins for histogram discretization (default 10 is a good choice)
///
/// # Returns
///
/// An [`AnomalyScore`] containing KL divergence, JSD, and a normalized score in [0, 1].
///
/// # Example
///
/// ```rust
/// use iii_observability::anomaly_score;
///
/// let baseline = vec![50.0; 100]; // all at 50ms
/// let current = vec![55.0; 100];  // all at 55ms — shifted
/// let score = anomaly_score(&current, &baseline, 10);
/// assert!(score.jsd > 0.0);
/// ```
pub fn anomaly_score(current: &[f64], baseline: &[f64], bins: usize) -> AnomalyScore {
    let kl = kl_divergence(current, baseline, bins);
    let js = jsd(current, baseline, bins);
    AnomalyScore::new(kl, js)
}

/// Compute transfer entropy to detect causal influence between two service metrics.
///
/// Transfer entropy measures whether one time series (the potential cause) helps
/// predict another (the potential effect) beyond the target's own history.
///
/// # Arguments
///
/// * `source` - The potential cause metric (e.g., latency of service A)
/// * `target` - The potential effect metric (e.g., error rate of service B)
/// * `lag` - Time lag for causality (typically 1)
/// * `bins` - Number of bins for discretization
///
/// # Returns
///
/// Transfer entropy in bits. Non-negative. Higher values = stronger directional influence.
///
/// # Example
///
/// ```rust
/// use iii_observability::transfer_causality;
///
/// // Service A's latency increases → Service B's errors follow
/// let a_latency = vec![50.0, 100.0, 150.0, 200.0, 250.0];
/// let b_errors = vec![0.01, 0.02, 0.05, 0.08, 0.12];
///
/// let te = transfer_causality(&a_latency, &b_errors, 1, 5);
/// // transfer_entropy should be non-zero — A's latency helps predict B's errors
/// assert!(te >= 0.0);
/// ```
pub fn transfer_causality(
    source: &[f64],
    target: &[f64],
    lag: usize,
    bins: usize,
) -> f64 {
    transfer_entropy(source, target, lag, bins)
}

/// Build a causality matrix from a set of named metric time series.
///
/// Entry `(i, j)` is the transfer entropy from metric `i` to metric `j`,
/// indicating how much metric `i` causally influences metric `j`.
///
/// # Arguments
///
/// * `metrics` - Named metric time series (e.g., `[("latency", [...])]`)
/// * `lag` - Time lag for transfer entropy
/// * `bins` - Number of bins for discretization
///
/// # Returns
///
/// A matrix of size n×n where n is the number of metrics.
///
/// # Example
///
/// ```rust
/// use iii_observability::causality_matrix;
///
/// let metrics = vec![
///     ("latency".to_string(), vec![50.0, 100.0, 150.0, 200.0, 250.0]),
///     ("errors".to_string(), vec![0.01, 0.02, 0.05, 0.08, 0.12]),
///     ("throughput".to_string(), vec![1000.0, 950.0, 900.0, 850.0, 800.0]),
/// ];
///
/// let matrix = causality_matrix(&metrics, 1, 5);
/// assert_eq!(matrix.len(), 3); // 3×3 matrix
/// // matrix[0][1] = transfer entropy from latency → errors
/// assert!(matrix[0][1] >= 0.0);
/// ```
pub fn causality_matrix(
    metrics: &[(String, Vec<f64>)],
    lag: usize,
    bins: usize,
) -> Vec<Vec<f64>> {
    let n = metrics.len();
    if n == 0 {
        return vec![];
    }

    let mut matrix = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[i][j] = 0.0; // No self-influence measured
            } else {
                let te = transfer_entropy(&metrics[i].1, &metrics[j].1, lag, bins);
                matrix[i][j] = te;
            }
        }
    }

    matrix
}

/// Build a correlation matrix from a set of named metric time series using mutual information.
///
/// Captures non-linear dependencies that Pearson correlation misses.
/// Entry `(i, j)` is the mutual information between metric `i` and metric `j`.
/// The matrix is symmetric with self-MI on the diagonal (entropy of the metric itself).
///
/// # Arguments
///
/// * `metrics` - Named metric time series
/// * `bins` - Number of bins for discretization
///
/// # Returns
///
/// A symmetric matrix of size n×n.
///
/// # Example
///
/// ```rust
/// use iii_observability::correlation_matrix;
///
/// let metrics = vec![
///     ("latency".to_string(), vec![50.0, 52.0, 48.0, 53.0, 51.0]),
///     ("errors".to_string(), vec![0.01, 0.015, 0.008, 0.02, 0.012]),
/// ];
///
/// let matrix = correlation_matrix(&metrics, 10);
/// assert_eq!(matrix.len(), 2);
/// // matrix[0][1] = mutual information between latency and errors
/// assert!(matrix[0][1] >= 0.0);
/// assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-10); // symmetric
/// ```
pub fn correlation_matrix(metrics: &[(String, Vec<f64>)], bins: usize) -> Vec<Vec<f64>> {
    let n = metrics.len();
    if n == 0 {
        return vec![];
    }

    let mut matrix = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        matrix[i][i] = crate::info::entropy(&metrics[i].1, bins);
        for j in (i + 1)..n {
            let mi = mutual_information(&metrics[i].1, &metrics[j].1, bins);
            matrix[i][j] = mi;
            matrix[j][i] = mi;
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_score_identical() {
        let data = vec![50.0, 52.0, 48.0, 51.0, 47.0];
        let score = anomaly_score(&data, &data, 10);
        assert!(
            score.jsd < 0.05,
            "identical distributions should have near-zero JSD, got {}",
            score.jsd
        );
        assert!(
            score.kl < 0.05,
            "identical distributions should have near-zero KL, got {}",
            score.kl
        );
    }

    #[test]
    fn test_anomaly_score_different() {
        let baseline = vec![50.0; 100];
        let current = vec![200.0; 100];
        let score = anomaly_score(&current, &baseline, 10);
        assert!(
            score.jsd > 0.1,
            "different distributions should have positive JSD, got {}",
            score.jsd
        );
    }

    #[test]
    fn test_anomaly_score_normalized_range() {
        let baseline = vec![50.0; 100];
        let current = vec![200.0; 100];
        let score = anomaly_score(&current, &baseline, 10);
        assert!((0.0..=1.0).contains(&score.normalized));
    }

    #[test]
    fn test_transfer_causality_self() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let te = transfer_causality(&data, &data, 1, 5);
        assert!(
            te >= 0.0,
            "transfer entropy should be non-negative, got {}",
            te
        );
    }

    #[test]
    fn test_transfer_causality_independent() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let te = transfer_causality(&a, &b, 1, 5);
        // Independent — should be low
        assert!(te >= 0.0);
    }

    #[test]
    fn test_causality_matrix_size() {
        let metrics = vec![
            ("a".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("b".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]),
            ("c".to_string(), vec![5.0, 4.0, 3.0, 2.0, 1.0]),
        ];
        let matrix = causality_matrix(&metrics, 1, 5);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
    }

    #[test]
    fn test_causality_matrix_empty() {
        let matrix = causality_matrix(&[], 1, 5);
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_causality_matrix_diagonal_zero() {
        let metrics = vec![
            ("a".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("b".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]),
        ];
        let matrix = causality_matrix(&metrics, 1, 5);
        assert_eq!(matrix[0][0], 0.0);
        assert_eq!(matrix[1][1], 0.0);
    }

    #[test]
    fn test_correlation_matrix_symmetric() {
        let metrics = vec![
            ("a".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("b".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]),
        ];
        let matrix = correlation_matrix(&metrics, 10);
        assert!(
            (matrix[0][1] - matrix[1][0]).abs() < 1e-10,
            "correlation matrix should be symmetric"
        );
    }

    #[test]
    fn test_correlation_matrix_empty() {
        let matrix = correlation_matrix(&[], 10);
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_correlation_matrix_non_negative() {
        let metrics = vec![
            ("a".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("b".to_string(), vec![5.0, 4.0, 3.0, 2.0, 1.0]),
        ];
        let matrix = correlation_matrix(&metrics, 10);
        for row in &matrix {
            for &val in row {
                assert!(val >= -1e-10, "MI should be non-negative, got {}", val);
            }
        }
    }

    #[test]
    fn test_anomaly_score_bins_zero() {
        let baseline = vec![50.0, 51.0, 49.0];
        let current = vec![55.0, 56.0, 54.0];
        let score = anomaly_score(&current, &baseline, 0);
        assert_eq!(score.kl, 0.0);
        assert_eq!(score.jsd, 0.0);
    }

    #[test]
    fn test_anomaly_score_empty() {
        let score = anomaly_score(&[], &[], 10);
        assert_eq!(score.kl, 0.0);
        assert_eq!(score.jsd, 0.0);
    }

    #[test]
    fn test_drift_detection() {
        // Simulate gradual drift: latency slowly increases
        let baseline = (0..100).map(|_| 50.0).collect::<Vec<_>>();
        let current = (0..100).map(|i| 50.0 + (i as f64)).collect::<Vec<_>>();
        let score = anomaly_score(&current, &baseline, 10);
        // JSD should detect the drift
        assert!(
            score.jsd > 0.01,
            "should detect drift, got JSD = {}",
            score.jsd
        );
    }
}
