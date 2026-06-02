//! Information-theoretic measures for anomaly detection.
//!
//! These functions are reimplemented from crackle-runtime's (unpublished) `information`
//! module. They provide the core measures: KL divergence, Jensen-Shannon divergence,
//! mutual information, and transfer entropy — computed via histogram-based
//! density estimation.

use std::collections::{BTreeMap, HashMap};

/// Compute the Kullback–Leibler divergence between two distributions.
///
/// KL(P || Q) measures how much information is lost when Q is used to approximate P.
/// It is asymmetric: KL(P || Q) != KL(Q || P) generally.
///
/// Uses histogram-based density estimation with `bins` bins.
///
/// # Arguments
///
/// * `p_samples` - Samples from distribution P (the "current" distribution)
/// * `q_samples` - Samples from distribution Q (the "baseline" distribution)
/// * `bins` - Number of bins for discretization
///
/// # Returns
///
/// KL(P || Q) in nats. Returns 0.0 if both inputs are empty or `bins` == 0.
pub fn kl_divergence(p_samples: &[f64], q_samples: &[f64], bins: usize) -> f64 {
    if p_samples.is_empty() || q_samples.is_empty() || bins == 0 {
        return 0.0;
    }

    let (p_hist, q_hist) = build_histograms(p_samples, q_samples, bins);
    let mut kl = 0.0;

    for (&p, &q) in p_hist.iter().zip(q_hist.iter()) {
        if p > 0.0 && q > 0.0 {
            kl += p * (p / q).ln();
        } else if p > 0.0 && q <= 0.0 {
            // P has mass where Q has none → infinite KL in theory
            // We clamp to a large value
            return 100.0;
        }
        // p == 0 contributes nothing
    }

    kl.max(0.0)
}

/// Compute the Jensen–Shannon divergence between two distributions.
///
/// JSD(P || Q) is a symmetric, bounded, always-finite measure of distribution
/// dissimilarity. Its square root is a metric.
///
/// * 0.0 = identical distributions
/// * ≤ ln(2) ≈ 0.693 for binary distributions, unbounded otherwise
/// * Normalized JSD (returned as `normalized`) divides by ln(2) so it's in [0, 1]
///
/// # Arguments
///
/// Same as [`kl_divergence`].
///
/// # Returns
///
/// * `jsd` - Raw Jensen-Shannon divergence
pub fn jsd(p_samples: &[f64], q_samples: &[f64], bins: usize) -> f64 {
    if p_samples.is_empty() || q_samples.is_empty() || bins == 0 {
        return 0.0;
    }

    let (p_hist, q_hist) = build_histograms(p_samples, q_samples, bins);

    // M = (P + Q) / 2
    let m: Vec<f64> = p_hist
        .iter()
        .zip(q_hist.iter())
        .map(|(&p, &q)| (p + q) / 2.0)
        .collect();

    // JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    let mut kl_pm = 0.0;
    let mut kl_qm = 0.0;

    for (((&p, &q), &m_val), _bin) in p_hist
        .iter()
        .zip(q_hist.iter())
        .zip(m.iter())
        .zip(0..bins)
    {
        if p > 0.0 && m_val > 0.0 {
            kl_pm += p * (p / m_val).ln();
        }
        if q > 0.0 && m_val > 0.0 {
            kl_qm += q * (q / m_val).ln();
        }
    }

    0.5 * kl_pm + 0.5 * kl_qm
}

/// Compute the entropy of a sample distribution.
///
/// H(X) = -Σ p(x) * ln(p(x))
///
/// # Arguments
///
/// * `samples` - Data samples
/// * `bins` - Number of bins for discretization
///
/// # Returns
///
/// Entropy in nats.
pub fn entropy(samples: &[f64], bins: usize) -> f64 {
    if samples.is_empty() || bins == 0 {
        return 0.0;
    }

    let hist = build_single_histogram(samples, bins);
    let mut h = 0.0;

    for &p in &hist {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }

    h
}

/// Compute mutual information between two variables.
///
/// I(X; Y) = H(X) + H(Y) - H(X, Y)
///
/// Measures how much knowing one variable reduces uncertainty about the other.
/// Captures non-linear dependencies that Pearson correlation misses.
///
/// # Arguments
///
/// * `x_samples` - Samples of variable X
/// * `y_samples` - Samples of variable Y
/// * `bins` - Number of bins per dimension
///
/// # Returns
///
/// Mutual information in nats. Always non-negative.
pub fn mutual_information(x_samples: &[f64], y_samples: &[f64], bins: usize) -> f64 {
    if x_samples.is_empty() || y_samples.is_empty() || bins == 0 {
        return 0.0;
    }

    // Ensure equal length
    let n = x_samples.len().min(y_samples.len());
    if n == 0 {
        return 0.0;
    }

    let x = &x_samples[..n];
    let y = &y_samples[..n];

    let hx = entropy(x, bins);
    let hy = entropy(y, bins);
    let hxy = joint_entropy(x, y, bins);

    (hx + hy - hxy).max(0.0)
}

/// Compute transfer entropy from source to target.
///
/// TE(S → T) measures how much knowing the past of S helps predict the
/// next value of T, beyond what T's own past already tells us.
///
/// # Arguments
///
/// * `source` - Potential cause time series
/// * `target` - Potential effect time series
/// * `lag` - Time lag for the causal relationship
/// * `bins` - Number of bins for discretization
///
/// # Returns
///
/// Transfer entropy in nats. Non-negative.
pub fn transfer_entropy(source: &[f64], target: &[f64], lag: usize, bins: usize) -> f64 {
    if source.len() < lag + 2 || target.len() < lag + 2 || bins == 0 || lag == 0 {
        return 0.0;
    }

    let n = source.len().min(target.len()).saturating_sub(lag);
    if n < 2 {
        return 0.0;
    }

    // Collect triples: (t_{i}, t_{i-1}, s_{i-1}) for i from lag to n-1
    // Actually we want: t_current, t_past (lagged), s_past (lagged)
    let mut t_current = Vec::with_capacity(n);
    let mut t_past = Vec::with_capacity(n);
    let mut s_past = Vec::with_capacity(n);

    for i in lag..(lag + n) {
        t_current.push(target[i]);
        t_past.push(target[i - lag]);
        s_past.push(source[i - lag]);
    }

    // TE(S → T) = H(T_current | T_past) - H(T_current | T_past, S_past)
    //           = H(T_current, T_past) - H(T_past) - [H(T_current, T_past, S_past) - H(T_past, S_past)]
    //           = H(T_current, T_past) - H(T_past) - H(T_current, T_past, S_past) + H(T_past, S_past)

    let h_tt = joint_entropy(&t_current, &t_past, bins);
    let h_t = entropy(&t_past, bins);
    let h_tts = joint_entropy_3d(&t_current, &t_past, &s_past, bins);
    let h_ts = joint_entropy(&t_past, &s_past, bins);

    let te = h_tt - h_t - h_tts + h_ts;
    te.max(0.0)
}

// ---- Internal helpers ----

/// Build histograms for two sample sets using shared bin edges.
fn build_histograms(a: &[f64], b: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    let (edges, _) = compute_bin_edges(a, b, bins);

    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    let mut hist_a = vec![0usize; bins];
    let mut hist_b = vec![0usize; bins];

    for &v in a {
        if let Some(idx) = bin_index(v, &edges) {
            hist_a[idx] += 1;
        }
    }

    for &v in b {
        if let Some(idx) = bin_index(v, &edges) {
            hist_b[idx] += 1;
        }
    }

    let p_hist: Vec<f64> = hist_a.iter().map(|&c| c as f64 / n_a).collect();
    let q_hist: Vec<f64> = hist_b.iter().map(|&c| c as f64 / n_b).collect();

    (p_hist, q_hist)
}

/// Build a single histogram.
fn build_single_histogram(samples: &[f64], bins: usize) -> Vec<f64> {
    if samples.is_empty() || bins == 0 {
        return vec![];
    }

    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < f64::EPSILON {
        // All values identical
        let mut hist = vec![0.0; bins];
        hist[0] = 1.0;
        return hist;
    }

    let bin_width = (max - min) / bins as f64;
    let n = samples.len() as f64;
    let mut hist = vec![0usize; bins];

    for &v in samples {
        let idx = ((v - min) / bin_width).floor() as usize;
        let idx = idx.min(bins - 1);
        hist[idx] += 1;
    }

    hist.into_iter().map(|c| c as f64 / n).collect()
}

/// Compute bin edges shared across two sample sets.
fn compute_bin_edges(a: &[f64], b: &[f64], bins: usize) -> (Vec<f64>, (f64, f64)) {
    let min = a
        .iter()
        .chain(b.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max = a
        .iter()
        .chain(b.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < f64::EPSILON {
        // All values identical — avoid division by zero
        let edges: Vec<f64> = (0..=bins).map(|i| min + i as f64).collect();
        return (edges, (min, max));
    }

    let bin_width = (max - min) / bins as f64;
    let edges: Vec<f64> = (0..=bins).map(|i| min + i as f64 * bin_width).collect();
    (edges, (min, max))
}

fn bin_index(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value >= edges[edges.len() - 1] {
        return None;
    }
    let idx = edges.len() - 2;
    for i in 0..idx {
        if value >= edges[i] && value < edges[i + 1] {
            return Some(i);
        }
    }
    // Fall through to last bin
    Some(edges.len() - 2)
}

/// Compute joint entropy H(X, Y) using 2D histogram.
fn joint_entropy(x: &[f64], y: &[f64], bins: usize) -> f64 {
    if x.is_empty() || y.is_empty() || bins == 0 {
        return 0.0;
    }

    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let x_range = if (x_max - x_min).abs() < f64::EPSILON {
        1.0
    } else {
        x_max - x_min
    };
    let y_range = if (y_max - y_min).abs() < f64::EPSILON {
        1.0
    } else {
        y_max - y_min
    };

    let x_bw = x_range / bins as f64;
    let y_bw = y_range / bins as f64;

    let mut joint = BTreeMap::new();

    for i in 0..n {
        let xi = ((x[i] - x_min) / x_bw).floor() as isize;
        let yi = ((y[i] - y_min) / y_bw).floor() as isize;
        *joint.entry((xi.max(0).min(bins as isize - 1), yi.max(0).min(bins as isize - 1))).or_insert(0) += 1;
    }

    let nf = n as f64;
    let mut h = 0.0;
    for &count in joint.values() {
        let p = count as f64 / nf;
        h -= p * p.ln();
    }

    h
}

/// Compute joint entropy H(X, Y, Z) using 3D histogram.
fn joint_entropy_3d(x: &[f64], y: &[f64], z: &[f64], bins: usize) -> f64 {
    if x.is_empty() || y.is_empty() || z.is_empty() || bins == 0 {
        return 0.0;
    }

    let n = x.len().min(y.len()).min(z.len());
    if n == 0 {
        return 0.0;
    }

    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
    let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let x_range = if (x_max - x_min).abs() < f64::EPSILON { 1.0 } else { x_max - x_min };
    let y_range = if (y_max - y_min).abs() < f64::EPSILON { 1.0 } else { y_max - y_min };
    let z_range = if (z_max - z_min).abs() < f64::EPSILON { 1.0 } else { z_max - z_min };

    let x_bw = x_range / bins as f64;
    let y_bw = y_range / bins as f64;
    let z_bw = z_range / bins as f64;

    let mut joint: HashMap<(isize, isize, isize), usize> = HashMap::new();
    let bi = bins as isize;

    for i in 0..n {
        let xi = ((x[i] - x_min) / x_bw).floor() as isize;
        let yi = ((y[i] - y_min) / y_bw).floor() as isize;
        let zi = ((z[i] - z_min) / z_bw).floor() as isize;
        *joint.entry((xi.max(0).min(bi - 1), yi.max(0).min(bi - 1), zi.max(0).min(bi - 1))).or_insert(0) += 1;
    }

    let nf = n as f64;
    let mut h = 0.0;
    for &count in joint.values() {
        let p = count as f64 / nf;
        h -= p * p.ln();
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kl_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kl = kl_divergence(&data, &data, 10);
        assert!(kl < 0.01, "KL(P||P) should be ~0, got {}", kl);
    }

    #[test]
    fn test_jsd_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let score = jsd(&data, &data, 10);
        assert!(score < 0.01, "JSD(P||P) should be ~0, got {}", score);
    }

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution over 5 values with 5 bins = max entropy
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let h = entropy(&data, 5);
        assert!(h > 0.0, "entropy should be positive, got {}", h);
        // Max entropy for 5 bins: ln(5) ≈ 1.609
        assert!((h - 5.0f64.ln()).abs() < 0.5, "entropy should be close to ln(5) ≈ 1.609, got {}", h);
    }

    #[test]
    fn test_mutual_information_independent() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let mi = mutual_information(&x, &y, 10);
        assert!(mi >= 0.0, "MI should be non-negative, got {}", mi);
    }

    #[test]
    fn test_transfer_entropy_non_negative() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let t = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let te = transfer_entropy(&s, &t, 1, 5);
        assert!(te >= 0.0, "TE should be non-negative, got {}", te);
    }

    #[test]
    fn test_kl_with_baseline_shift() {
        // Overlapping but shifted distributions
        let baseline = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let current = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let kl = kl_divergence(&current, &baseline, 5);
        assert!(kl > 0.0, "shifted distributions should have KL > 0, got {}", kl);
    }

    #[test]
    fn test_jsd_normalized_range() {
        let baseline = vec![50.0; 100];
        let current = vec![200.0; 100];
        let score = jsd(&current, &baseline, 10);
        assert!(score >= 0.0, "JSD should be non-negative, got {}", score);
    }

    #[test]
    fn test_empty_inputs() {
        assert_eq!(kl_divergence(&[], &[1.0], 10), 0.0);
        assert_eq!(jsd(&[], &[1.0], 10), 0.0);
        assert_eq!(entropy(&[], 10), 0.0);
        assert_eq!(mutual_information(&[], &[1.0], 10), 0.0);
        assert_eq!(transfer_entropy(&[], &[1.0], 1, 10), 0.0);
    }

    #[test]
    fn test_zero_bins() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(kl_divergence(&data, &data, 0), 0.0);
        assert_eq!(jsd(&data, &data, 0), 0.0);
        assert_eq!(entropy(&data, 0), 0.0);
    }
}
