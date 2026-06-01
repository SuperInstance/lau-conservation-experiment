//! Landauer cost tracker — computes the thermodynamic cost of erasure per step.
//!
//! Landauer's principle: erasing one bit of information costs kT·ln(2) of energy.
//! We model each agent step as partial erasure of prior belief states.

use serde::{Serialize, Deserialize};

/// Tracks cumulative Landauer cost across an agent lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandauerTracker {
    /// Boltzmann constant (normalized, we use k=1 in natural units)
    k_boltzmann: f64,
    /// Temperature parameter
    temperature: f64,
    /// ln(2) constant
    ln2: f64,
    /// Cumulative Landauer cost
    cumulative_cost: f64,
    /// Cost per step history
    step_costs: Vec<f64>,
    /// Total bits erased estimate
    bits_erased: f64,
}

impl LandauerTracker {
    pub fn new(temperature: f64) -> Self {
        Self {
            k_boltzmann: 1.0,
            temperature,
            ln2: std::f64::consts::LN_2,
            cumulative_cost: 0.0,
            step_costs: Vec::new(),
            bits_erased: 0.0,
        }
    }

    /// Compute Landauer cost for erasing `bits` bits at current temperature.
    /// Cost = kT * ln(2) * bits
    pub fn compute_erasure_cost(&self, bits: f64) -> f64 {
        self.k_boltzmann * self.temperature * self.ln2 * bits.abs()
    }

    /// Record an erasure step and return the cost.
    pub fn step(&mut self, bits_erased: f64) -> f64 {
        let cost = self.compute_erasure_cost(bits_erased);
        self.cumulative_cost += cost;
        self.bits_erased += bits_erased.abs();
        self.step_costs.push(cost);
        cost
    }

    /// Compute cost for a belief state update (KL divergence → bits erased).
    pub fn step_kl(&mut self, kl_divergence: f64) -> f64 {
        // KL divergence in nats → bits via / ln(2)
        let bits = kl_divergence / self.ln2;
        self.step(bits)
    }

    pub fn cumulative_cost(&self) -> f64 {
        self.cumulative_cost
    }

    pub fn bits_erased(&self) -> f64 {
        self.bits_erased
    }

    pub fn step_costs(&self) -> &[f64] {
        &self.step_costs
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t;
    }

    /// Reset tracker (for new lifecycle).
    pub fn reset(&mut self) {
        self.cumulative_cost = 0.0;
        self.step_costs.clear();
        self.bits_erased = 0.0;
    }

    /// Number of steps taken.
    pub fn step_count(&self) -> usize {
        self.step_costs.len()
    }

    /// Mean cost per step.
    pub fn mean_cost(&self) -> f64 {
        if self.step_costs.is_empty() {
            0.0
        } else {
            self.cumulative_cost / self.step_costs.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_new_tracker() {
        let t = LandauerTracker::new(1.0);
        assert_eq!(t.cumulative_cost(), 0.0);
        assert_eq!(t.bits_erased(), 0.0);
        assert!(t.step_costs().is_empty());
    }

    #[test]
    fn test_single_erasure() {
        let mut t = LandauerTracker::new(1.0);
        let cost = t.step(1.0); // erase 1 bit at T=1
        assert_relative_eq!(cost, std::f64::consts::LN_2, epsilon = 1e-10);
        assert_relative_eq!(t.cumulative_cost(), std::f64::consts::LN_2, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_bits() {
        let mut t = LandauerTracker::new(1.0);
        let cost = t.step(0.0);
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_temperature_scaling() {
        let mut t1 = LandauerTracker::new(1.0);
        let mut t2 = LandauerTracker::new(2.0);
        let c1 = t1.step(1.0);
        let c2 = t2.step(1.0);
        assert_relative_eq!(c2 / c1, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cumulative_multiple_steps() {
        let mut t = LandauerTracker::new(1.0);
        t.step(1.0);
        t.step(2.0);
        t.step(0.5);
        let expected = std::f64::consts::LN_2 * (1.0 + 2.0 + 0.5);
        assert_relative_eq!(t.cumulative_cost(), expected, epsilon = 1e-10);
        assert_eq!(t.step_count(), 3);
    }

    #[test]
    fn test_kl_step() {
        let mut t = LandauerTracker::new(1.0);
        let kl = std::f64::consts::LN_2; // 1 nat = 1 bit
        let cost = t.step_kl(kl);
        // bits = kl / ln2 = 1, cost = ln2 * 1 = ln2
        assert_relative_eq!(cost, std::f64::consts::LN_2, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut t = LandauerTracker::new(1.0);
        t.step(5.0);
        t.step(3.0);
        t.reset();
        assert_eq!(t.cumulative_cost(), 0.0);
        assert_eq!(t.bits_erased(), 0.0);
        assert!(t.step_costs().is_empty());
    }

    #[test]
    fn test_mean_cost() {
        let mut t = LandauerTracker::new(1.0);
        t.step(1.0);
        t.step(3.0);
        let mean = t.mean_cost();
        let expected = std::f64::consts::LN_2 * 2.0; // (1+3)/2 * ln2
        assert_relative_eq!(mean, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_negative_bits_abs() {
        let mut t = LandauerTracker::new(1.0);
        let cost = t.step(-2.0);
        assert_relative_eq!(cost, 2.0 * std::f64::consts::LN_2, epsilon = 1e-10);
        assert_relative_eq!(t.bits_erased(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_set_temperature() {
        let mut t = LandauerTracker::new(1.0);
        t.set_temperature(3.0);
        let cost = t.step(1.0);
        assert_relative_eq!(cost, 3.0 * std::f64::consts::LN_2, epsilon = 1e-10);
    }
}
