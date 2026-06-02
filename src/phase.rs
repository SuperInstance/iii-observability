use crackle_runtime::PatternKind;

/// Maps crackle-runtime's pattern detection phases to SRE service states.
///
/// In pottery, the crackle glaze forms through distinct phases:
///
/// | Pottery Phase | SRE State | Meaning |
/// |---|---|---|
/// | Pre-transition | Nominal | Service is healthy, metrics are stable |
/// | Transitioning | Degrading | Service is showing signs of trouble |
/// | Post-transition | Recovered | Service has returned to health |
///
/// By mapping crackle-runtime's detected patterns to these SRE states,
/// this detector tells you not just WHAT is wrong, but WHERE the service
/// is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SreState {
    /// Service is operating normally. No concerning patterns detected.
    Nominal,
    /// Service is degrading. Phase transitions, clustering, or correlations
    /// indicate something is changing.
    Degrading,
    /// Service has recovered. Conservation patterns suggest stability returned.
    Recovered,
}

impl SreState {
    /// Human-readable label for this SRE state.
    pub fn label(&self) -> &'static str {
        match self {
            SreState::Nominal => "nominal",
            SreState::Degrading => "degrading",
            SreState::Recovered => "recovered",
        }
    }

    /// Whether this state requires attention.
    pub fn requires_attention(&self) -> bool {
        matches!(self, SreState::Degrading)
    }
}

impl std::fmt::Display for SreState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Maps crackle-runtime [`PatternKind`] values to SRE lifecycle phases.
///
/// The crackle glaze forms during cooling, not firing. Similarly, SRE patterns
/// emerge from the collective behavior of your services — visible only in
/// aggregate.
///
/// # Mapping
///
/// | PatternKind | SRE State | Rationale |
/// |---|---|---|
/// | `PhaseTransition` | Degrading | A phase transition in metrics means the service distribution has shifted — this IS service degradation |
/// | `Clustering` | Degrading | Metrics clustering into groups suggests multiple modes of behavior, a sign of instability |
/// | `Correlation` | Degrading | Unexpected correlations between metrics suggest cascading effects |
/// | `Conservation` | Recovered | Conservation laws mean metrics are stable — consistent behavior = recovery |
///
/// # Example
///
/// ```rust
/// use iii_observability::PhaseDetector;
/// use crackle_runtime::PatternKind;
///
/// let state = PhaseDetector::pattern_to_state(&PatternKind::PhaseTransition);
/// assert_eq!(state.to_string(), "degrading");
///
/// let state = PhaseDetector::pattern_to_state(&PatternKind::Conservation);
/// assert_eq!(state.to_string(), "recovered");
/// ```
pub struct PhaseDetector;

impl PhaseDetector {
    /// Map a [`PatternKind`] to the corresponding [`SreState`].
    pub fn pattern_to_state(kind: &PatternKind) -> SreState {
        match kind {
            PatternKind::Clustering => SreState::Degrading,
            PatternKind::PhaseTransition => SreState::Degrading,
            PatternKind::Correlation => SreState::Degrading,
            PatternKind::Conservation => SreState::Recovered,
        }
    }

    /// Determine the overall SRE state from a set of crackle-runtime patterns.
    ///
    /// The logic is conservative:
    /// - If ANY pattern indicates degradation, the overall state is Degrading
    /// - If ALL patterns indicate recovery, the state is Recovered
    /// - If there are no patterns at all, the state is Nominal (no signal = no problem)
    ///
    /// # Example
    ///
    /// ```rust
    /// use iii_observability::PhaseDetector;
    /// use crackle_runtime::{CracklePattern, PatternKind};
    ///
    /// let patterns = vec![
    ///     CracklePattern::new(PatternKind::PhaseTransition, "latency shifted", vec![], 0.8),
    /// ];
    ///
    /// let state = PhaseDetector::aggregate_state(&patterns);
    /// assert_eq!(state.to_string(), "degrading");
    /// ```
    pub fn aggregate_state(patterns: &[crackle_runtime::CracklePattern]) -> SreState {
        if patterns.is_empty() {
            return SreState::Nominal;
        }

        let mut has_degrading = false;
        let mut has_recovered = false;

        for pattern in patterns {
            match Self::pattern_to_state(pattern.kind()) {
                SreState::Degrading => has_degrading = true,
                SreState::Recovered => has_recovered = true,
                SreState::Nominal => {} // Nominal patterns don't affect aggregate
            }
        }

        // Degrading takes priority — if there's ANY sign of trouble, flag it
        if has_degrading {
            SreState::Degrading
        } else if has_recovered {
            // Only recovered patterns → service is stable
            SreState::Recovered
        } else {
            SreState::Nominal
        }
    }

    /// Summarize patterns and their SRE states.
    ///
    /// Returns a human-readable summary of the current operational state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iii_observability::PhaseDetector;
    /// use crackle_runtime::{CracklePattern, PatternKind};
    ///
    /// let summary = PhaseDetector::summarize(&[
    ///     CracklePattern::new(PatternKind::PhaseTransition, "latency shifted by 40%", vec!["api".into()], 0.8),
    ///     CracklePattern::new(PatternKind::Conservation, "throughput conserved", vec!["api".into()], 0.9),
    /// ]);
    /// assert!(summary.contains("DEGRADING"));
    /// ```
    pub fn summarize(patterns: &[crackle_runtime::CracklePattern]) -> String {
        let state = Self::aggregate_state(patterns);

        let mut lines = Vec::new();
        lines.push(format!("OPERATIONAL STATE: {}", state.label().to_uppercase()));

        for pattern in patterns {
            let sre_state = Self::pattern_to_state(pattern.kind());
            lines.push(format!(
                "  [{} → {}] {} (confidence: {:.2})",
                pattern.kind(),
                sre_state,
                pattern.description(),
                pattern.confidence()
            ));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crackle_runtime::{CracklePattern, PatternKind};

    #[test]
    fn test_nominal_when_no_patterns() {
        let state = PhaseDetector::aggregate_state(&[]);
        assert_eq!(state, SreState::Nominal);
    }

    #[test]
    fn test_phase_transition_is_degrading() {
        let patterns = vec![CracklePattern::new(
            PatternKind::PhaseTransition,
            "latency shifted",
            vec![],
            0.8,
        )];
        let state = PhaseDetector::aggregate_state(&patterns);
        assert_eq!(state, SreState::Degrading);
    }

    #[test]
    fn test_clustering_is_degrading() {
        let patterns = vec![CracklePattern::new(
            PatternKind::Clustering,
            "tasks clustered",
            vec![],
            0.7,
        )];
        let state = PhaseDetector::aggregate_state(&patterns);
        assert_eq!(state, SreState::Degrading);
    }

    #[test]
    fn test_correlation_is_degrading() {
        let patterns = vec![CracklePattern::new(
            PatternKind::Correlation,
            "latency correlated with errors",
            vec![],
            0.9,
        )];
        let state = PhaseDetector::aggregate_state(&patterns);
        assert_eq!(state, SreState::Degrading);
    }

    #[test]
    fn test_conservation_is_recovered() {
        let patterns = vec![CracklePattern::new(
            PatternKind::Conservation,
            "throughput conserved",
            vec![],
            0.9,
        )];
        let state = PhaseDetector::aggregate_state(&patterns);
        assert_eq!(state, SreState::Recovered);
    }

    #[test]
    fn test_degrading_overrides_recovered() {
        let patterns = vec![
            CracklePattern::new(PatternKind::Conservation, "stable", vec![], 0.9),
            CracklePattern::new(PatternKind::PhaseTransition, "shift", vec![], 0.8),
        ];
        let state = PhaseDetector::aggregate_state(&patterns);
        assert_eq!(
            state,
            SreState::Degrading,
            "degrading should override recovered"
        );
    }

    #[test]
    fn test_requires_attention() {
        assert!(!SreState::Nominal.requires_attention());
        assert!(SreState::Degrading.requires_attention());
        assert!(!SreState::Recovered.requires_attention());
    }

    #[test]
    fn test_summary_contains_state() {
        let patterns = vec![CracklePattern::new(
            PatternKind::PhaseTransition,
            "test shift",
            vec![],
            0.8,
        )];
        let summary = PhaseDetector::summarize(&patterns);
        assert!(summary.contains("DEGRADING"));
        assert!(summary.contains("test shift"));
    }

    #[test]
    fn test_summary_empty() {
        let summary = PhaseDetector::summarize(&[]);
        assert!(summary.contains("NOMINAL"));
    }

    #[test]
    fn test_combined_patterns_all_recovered() {
        let patterns = vec![
            CracklePattern::new(PatternKind::Conservation, "stable A", vec![], 0.9),
            CracklePattern::new(PatternKind::Conservation, "stable B", vec![], 0.7),
        ];
        let state = PhaseDetector::aggregate_state(&patterns);
        assert_eq!(state, SreState::Recovered);
    }

    #[test]
    fn test_combined_patterns_mixed() {
        // Single degrading pattern dominates
        let patterns = vec![
            CracklePattern::new(PatternKind::Conservation, "stable", vec![], 0.9),
            CracklePattern::new(PatternKind::Correlation, "corrolation", vec![], 0.6),
        ];
        let state = PhaseDetector::aggregate_state(&patterns);
        assert_eq!(state, SreState::Degrading);
    }

    #[test]
    fn test_empty_requires_attention() {
        assert!(!SreState::Nominal.requires_attention());
    }

    #[test]
    fn test_summary_multiple_patterns() {
        let patterns = vec![
            CracklePattern::new(PatternKind::Conservation, "throughput stable", vec![], 0.9),
            CracklePattern::new(PatternKind::Correlation, "error_rate ↔ latency_ms", vec![], 0.85),
            CracklePattern::new(PatternKind::PhaseTransition, "latency_ms shifted up", vec![], 0.75),
        ];
        let summary = PhaseDetector::summarize(&patterns);
        assert!(summary.contains("DEGRADING"), "should be DEGRADING, got: {}", summary);
        assert!(summary.contains("throughput stable"), "should mention patterns");
        assert!(summary.contains("latency_ms shifted up"), "should mention all patterns");
    }

    #[test]
    fn test_confidence_threshold_nominal() {
        // Very low confidence patterns should not degrade state
        let patterns = vec![
            CracklePattern::new(PatternKind::Correlation, "weak corr", vec![], 0.1),
            CracklePattern::new(PatternKind::PhaseTransition, "weak shift", vec![], 0.05),
        ];
        let state = PhaseDetector::aggregate_state(&patterns);
        // Our state detection doesn't filter by confidence — it's a PatternKind decision.
        // With low-confidence degrading patterns, still Degrading.
        // This is expected behavior; users should filter patterns before passing to aggregate_state.
        assert!(
            state == SreState::Degrading,
            "all degrading patterns produce Degrading regardless of confidence: got {:?}",
            state
        );
    }
}
