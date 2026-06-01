//! Conservation invariant checker.
//!
//! THE CORE CLAIM: Landauer cost + Free energy + H¹ risk score ≈ constant
//! across the agent lifecycle.
//!
//! Variance from this constant should scale as 1/spectral_gap.

use crate::{LandauerTracker, FreeEnergyTracker, CohomologyTracker};
use serde::{Serialize, Deserialize};

/// Records the combined invariant at a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantSnapshot {
    pub step: usize,
    pub landauer_cost: f64,
    pub free_energy: f64,
    pub h1_risk: f64,
    pub invariant_value: f64,
}

/// Tracks and tests the emergent conservation law.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationInvariant {
    /// History of invariant snapshots
    snapshots: Vec<InvariantSnapshot>,
    /// Expected constant value (set from initial conditions)
    expected_constant: Option<f64>,
    /// Variance tolerance (controlled by spectral gap)
    tolerance: f64,
    /// Number of violations detected
    violations: u32,
    /// Details of violations
    violation_details: Vec<String>,
}

impl ConservationInvariant {
    pub fn new(tolerance: f64) -> Self {
        Self {
            snapshots: Vec::new(),
            expected_constant: None,
            tolerance,
            violations: 0,
            violation_details: Vec::new(),
        }
    }

    /// Compute the invariant: Landauer + F + H¹
    /// NOTE: We use cumulative Landauer, current F, and current H¹.
    /// The conservation law says: ΔLandauer + ΔF + ΔH¹ ≈ 0 per step.
    /// Equivalently: Landauer_cum + F_current + H¹_cum ≈ initial budget.
    pub fn compute(
        landauer: &LandauerTracker,
        free_energy: &FreeEnergyTracker,
        cohomology: &CohomologyTracker,
    ) -> f64 {
        // The invariant: cumulative landauer + current free energy + cumulative H¹ risk
        landauer.cumulative_cost() + free_energy.free_energy() + cohomology.cumulative_risk()
    }

    /// Record a snapshot and check conservation.
    pub fn check(
        &mut self,
        step: usize,
        landauer: &LandauerTracker,
        free_energy: &FreeEnergyTracker,
        cohomology: &CohomologyTracker,
    ) -> InvariantSnapshot {
        let invariant = Self::compute(landauer, free_energy, cohomology);
        
        let snapshot = InvariantSnapshot {
            step,
            landauer_cost: landauer.cumulative_cost(),
            free_energy: free_energy.free_energy(),
            h1_risk: cohomology.cumulative_risk(),
            invariant_value: invariant,
        };

        // Set expected constant from first snapshot
        if self.expected_constant.is_none() {
            self.expected_constant = Some(invariant);
        }

        // Check for violation
        if let Some(expected) = self.expected_constant {
            let deviation = (invariant - expected).abs();
            if deviation > self.tolerance {
                self.violations += 1;
                self.violation_details.push(format!(
                    "Step {}: invariant={:.6}, expected={:.6}, deviation={:.6} > tolerance={:.6}",
                    step, invariant, expected, deviation, self.tolerance
                ));
            }
        }

        self.snapshots.push(snapshot.clone());
        snapshot
    }

    /// Set the expected constant manually (for controlled experiments).
    pub fn set_expected_constant(&mut self, c: f64) {
        self.expected_constant = Some(c);
    }

    /// Compute variance of the invariant across all snapshots.
    pub fn variance(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }
        let values: Vec<f64> = self.snapshots.iter().map(|s| s.invariant_value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
    }

    /// Standard deviation of invariant.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Mean invariant value.
    pub fn mean(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        self.snapshots.iter().map(|s| s.invariant_value).sum::<f64>() / self.snapshots.len() as f64
    }

    /// Maximum absolute deviation from mean.
    pub fn max_deviation(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let mean = self.mean();
        self.snapshots.iter().map(|s| (s.invariant_value - mean).abs()).fold(0.0f64, f64::max)
    }

    /// Check if conservation law holds (violations == 0 within tolerance).
    pub fn is_conserved(&self) -> bool {
        self.violations == 0
    }

    pub fn violations(&self) -> u32 {
        self.violations
    }

    pub fn violation_details(&self) -> &[String] {
        &self.violation_details
    }

    pub fn snapshots(&self) -> &[InvariantSnapshot] {
        &self.snapshots
    }

    pub fn expected_constant(&self) -> Option<f64> {
        self.expected_constant
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Set tolerance (e.g., from spectral gap: tolerance ∝ 1/spectral_gap).
    pub fn set_tolerance(&mut self, t: f64) {
        self.tolerance = t;
    }

    pub fn reset(&mut self) {
        self.snapshots.clear();
        self.expected_constant = None;
        self.violations = 0;
        self.violation_details.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_new_invariant() {
        let inv = ConservationInvariant::new(0.1);
        assert!(inv.is_conserved());
        assert_eq!(inv.violations(), 0);
    }

    #[test]
    fn test_first_snapshot_sets_constant() {
        let mut inv = ConservationInvariant::new(1.0);
        let l = LandauerTracker::new(1.0);
        let fe = FreeEnergyTracker::new(1.0, 100.0);
        let co = CohomologyTracker::new();
        inv.check(0, &l, &fe, &co);
        assert!(inv.expected_constant().is_some());
    }

    #[test]
    fn test_conservation_holds_initial() {
        let mut inv = ConservationInvariant::new(1.0);
        let l = LandauerTracker::new(1.0);
        let fe = FreeEnergyTracker::new(1.0, 100.0);
        let co = CohomologyTracker::new();
        inv.check(0, &l, &fe, &co);
        inv.check(1, &l, &fe, &co);
        assert!(inv.is_conserved());
    }

    #[test]
    fn test_conservation_balanced_tradeoff() {
        // If Landauer goes up by X, free energy should go down by ~X (with H¹ constant)
        let mut inv = ConservationInvariant::new(5.0);
        let mut l = LandauerTracker::new(1.0);
        let fe = FreeEnergyTracker::new(1.0, 100.0);
        let co = CohomologyTracker::new();
        
        inv.check(0, &l, &fe, &co);
        
        // Erase some bits → Landauer increases
        l.step(5.0);
        
        // The invariant should shift by the change in Landauer
        let snapshot = inv.check(1, &l, &fe, &co);
        // Landauer increased by 5*ln2 ≈ 3.47
        assert!(snapshot.landauer_cost > 0.0);
    }

    #[test]
    fn test_variance_empty() {
        let inv = ConservationInvariant::new(0.1);
        assert_eq!(inv.variance(), 0.0);
    }

    #[test]
    fn test_variance_computation() {
        let mut inv = ConservationInvariant::new(100.0);
        let l = LandauerTracker::new(1.0);
        let fe = FreeEnergyTracker::new(1.0, 100.0);
        let co = CohomologyTracker::new();
        inv.check(0, &l, &fe, &co);
        inv.check(1, &l, &fe, &co);
        assert_relative_eq!(inv.variance(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_violation_detected() {
        let mut inv = ConservationInvariant::new(0.01); // tight tolerance
        let l = LandauerTracker::new(1.0);
        let mut fe = FreeEnergyTracker::new(1.0, 100.0);
        let mut co = CohomologyTracker::new();
        
        inv.check(0, &l, &fe, &co);
        
        // Sudden energy injection with no Landauer cost
        fe.add_energy(50.0);
        co.set_risk(0.0); // no risk change
        
        inv.check(1, &l, &fe, &co);
        // Should detect violation if invariant changed by more than tolerance
        // Actually: invariant = 0 + (150 - 0) + 0 = 150 ≠ 100
        assert!(inv.violations() > 0);
    }

    #[test]
    fn test_max_deviation() {
        let mut inv = ConservationInvariant::new(100.0);
        let l = LandauerTracker::new(1.0);
        let fe = FreeEnergyTracker::new(1.0, 100.0);
        let co = CohomologyTracker::new();
        inv.check(0, &l, &fe, &co);
        inv.check(1, &l, &fe, &co);
        assert_relative_eq!(inv.max_deviation(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut inv = ConservationInvariant::new(0.1);
        let l = LandauerTracker::new(1.0);
        let fe = FreeEnergyTracker::new(1.0, 100.0);
        let co = CohomologyTracker::new();
        inv.check(0, &l, &fe, &co);
        inv.reset();
        assert!(inv.snapshots().is_empty());
        assert!(inv.expected_constant().is_none());
    }

    #[test]
    fn test_snapshot_fields() {
        let mut inv = ConservationInvariant::new(1.0);
        let l = LandauerTracker::new(1.0);
        let fe = FreeEnergyTracker::new(1.0, 100.0);
        let co = CohomologyTracker::new();
        let snap = inv.check(42, &l, &fe, &co);
        assert_eq!(snap.step, 42);
        assert_relative_eq!(snap.free_energy, 100.0, epsilon = 1e-10);
    }
}
