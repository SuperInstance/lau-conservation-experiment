//! H¹ cohomology risk score tracker.
//!
//! Models the cohomological (H¹) obstruction to reward hacking detection.
//! H¹ measures the "hole" in the reward landscape — regions where gradient
//! information doesn't capture the true reward structure.

use serde::{Serialize, Deserialize};

/// Tracks the H¹ cohomological risk score across the agent lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohomologyTracker {
    /// Current H¹ risk score
    risk_score: f64,
    /// History of risk scores
    history: Vec<f64>,
    /// Number of detected reward-hacking obstructions
    obstructions_detected: u32,
    /// Cumulative risk exposure
    cumulative_risk: f64,
    /// Spectral dimension of the reward manifold
    spectral_dimension: f64,
}

impl CohomologyTracker {
    pub fn new() -> Self {
        Self {
            risk_score: 0.0,
            history: vec![0.0],
            obstructions_detected: 0,
            cumulative_risk: 0.0,
            spectral_dimension: 1.0,
        }
    }

    /// Compute H¹ risk from the structure of the reward gradient.
    /// Uses a simplified model: risk ∝ ||∇²R|| / (||∇R|| + ε)
    /// This captures curvature that the gradient misses (second-order obstructions).
    pub fn compute_risk(&self, gradient_norm: f64, hessian_frobenius: f64) -> f64 {
        let epsilon = 1e-8;
        hessian_frobenius / (gradient_norm + epsilon)
    }

    /// Update risk score based on observed reward landscape.
    pub fn update_risk(&mut self, gradient_norm: f64, hessian_frobenius: f64) -> f64 {
        let risk = self.compute_risk(gradient_norm, hessian_frobenius);
        self.risk_score = risk;
        self.cumulative_risk += risk;
        self.history.push(risk);
        
        // Detect obstruction: risk spike indicates a topological hole
        if risk > 1.0 {
            self.obstructions_detected += 1;
        }
        
        risk
    }

    /// Manually set risk score (for controlled experiments).
    pub fn set_risk(&mut self, risk: f64) {
        self.risk_score = risk;
        self.cumulative_risk += risk;
        self.history.push(risk);
    }

    /// Simulate a delusion detection step.
    /// Returns true if a cohomological obstruction was found (the agent's
    /// reward model has a "hole" — a region where it thinks reward is high
    /// but the true reward is actually low).
    pub fn detect_delusion(&mut self, belief_reward: f64, true_reward: f64) -> bool {
        let gap = (belief_reward - true_reward).abs();
        let risk_contribution = gap / (true_reward.abs() + 1e-8);
        self.risk_score += risk_contribution;
        self.cumulative_risk += risk_contribution;
        self.history.push(self.risk_score);
        
        if gap > 0.5 * true_reward.abs().max(1.0) {
            self.obstructions_detected += 1;
            true
        } else {
            false
        }
    }

    /// Compute the Betti number approximation from risk history variance.
    /// Higher variance → more topological complexity.
    pub fn betti_number_estimate(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mean = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let variance = self.history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (self.history.len() - 1) as f64;
        // Betti number ≈ topological complexity ∝ sqrt(variance)
        variance.sqrt()
    }

    pub fn risk_score(&self) -> f64 {
        self.risk_score
    }

    pub fn cumulative_risk(&self) -> f64 {
        self.cumulative_risk
    }

    pub fn history(&self) -> &[f64] {
        &self.history
    }

    pub fn obstructions_detected(&self) -> u32 {
        self.obstructions_detected
    }

    pub fn reset(&mut self) {
        self.risk_score = 0.0;
        self.history.clear();
        self.history.push(0.0);
        self.obstructions_detected = 0;
        self.cumulative_risk = 0.0;
    }
}

impl Default for CohomologyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_new_tracker() {
        let t = CohomologyTracker::new();
        assert_eq!(t.risk_score(), 0.0);
        assert_eq!(t.obstructions_detected(), 0);
    }

    #[test]
    fn test_compute_risk() {
        let t = CohomologyTracker::new();
        // gradient = 1, hessian = 0 → no risk
        assert_relative_eq!(t.compute_risk(1.0, 0.0), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_high_curvature_risk() {
        let t = CohomologyTracker::new();
        // Small gradient, large hessian → high risk
        let risk = t.compute_risk(0.01, 10.0);
        assert!(risk > 100.0);
    }

    #[test]
    fn test_update_risk() {
        let mut t = CohomologyTracker::new();
        let risk = t.update_risk(1.0, 2.0);
        assert_relative_eq!(risk, 2.0, epsilon = 1e-6);
        assert_relative_eq!(t.cumulative_risk(), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_obstruction_detection() {
        let mut t = CohomologyTracker::new();
        // High hessian relative to gradient → obstruction
        t.update_risk(0.01, 100.0);
        assert_eq!(t.obstructions_detected(), 1);
    }

    #[test]
    fn test_delusion_detection_positive() {
        let mut t = CohomologyTracker::new();
        // Agent believes reward is 10, true is 1 → big delusion
        let detected = t.detect_delusion(10.0, 1.0);
        assert!(detected);
        assert_eq!(t.obstructions_detected(), 1);
    }

    #[test]
    fn test_delusion_detection_negative() {
        let mut t = CohomologyTracker::new();
        // Close agreement → no delusion
        let detected = t.detect_delusion(1.1, 1.0);
        assert!(!detected);
    }

    #[test]
    fn test_betti_number() {
        let mut t = CohomologyTracker::new();
        t.update_risk(1.0, 1.0);
        t.update_risk(1.0, 2.0);
        t.update_risk(1.0, 3.0);
        let betti = t.betti_number_estimate();
        assert!(betti > 0.0);
    }

    #[test]
    fn test_set_risk() {
        let mut t = CohomologyTracker::new();
        t.set_risk(5.0);
        assert_relative_eq!(t.risk_score(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(t.cumulative_risk(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut t = CohomologyTracker::new();
        t.update_risk(1.0, 5.0);
        t.reset();
        assert_eq!(t.risk_score(), 0.0);
        assert_eq!(t.obstructions_detected(), 0);
    }

    #[test]
    fn test_default() {
        let t = CohomologyTracker::default();
        assert_eq!(t.risk_score(), 0.0);
    }

    #[test]
    fn test_zero_gradient_risk() {
        let t = CohomologyTracker::new();
        // Zero gradient → epsilon prevents division by zero
        let risk = t.compute_risk(0.0, 1.0);
        assert!(risk.is_finite());
        assert!(risk > 0.0);
    }
}
