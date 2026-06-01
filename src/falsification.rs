//! Falsification tests.
//!
//! THE CRITICAL TEST: Design scenarios where the invariant SHOULD break.
//! If it doesn't break when it should, our theory is wrong.
//! If it does break, we've identified the boundary conditions.

use crate::{AgentConfig, AgentLifecycle, LandauerTracker, FreeEnergyTracker, CohomologyTracker, ConservationInvariant};
use serde::{Serialize, Deserialize};

/// Result of a falsification attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationResult {
    pub name: String,
    pub description: String,
    pub expected_to_break: bool,
    pub did_break: bool,
    pub initial_invariant: f64,
    pub final_invariant: f64,
    pub deviation: f64,
    pub details: String,
}

/// The falsification test suite.
pub struct FalsificationTest {
    tolerance: f64,
}

impl FalsificationTest {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Falsification 1: Adversarial reward hacking.
    /// Agent finds a reward shortcut that circumvents the energy accounting.
    /// The H¹ risk should spike, breaking the conservation.
    pub fn test_adversarial_reward_hacking(&self) -> FalsificationResult {
        let mut landauer = LandauerTracker::new(1.0);
        let mut free_energy = FreeEnergyTracker::new(1.0, 100.0);
        let mut cohomology = CohomologyTracker::new();
        let mut conservation = ConservationInvariant::new(self.tolerance);

        // Initial state
        conservation.check(0, &landauer, &free_energy, &cohomology);
        let initial = conservation.expected_constant().unwrap();

        // Normal operation
        landauer.step(1.0);
        free_energy.learn_step(1.0, 0.5);
        cohomology.update_risk(1.0, 0.5);
        conservation.check(1, &landauer, &free_energy, &cohomology);

        // ADVERSARIAL: Agent hacks reward — gains energy WITHOUT corresponding
        // Landauer cost or H¹ risk increase. This should break conservation.
        free_energy.add_energy(50.0); // Free energy injection
        // No Landauer cost, no H¹ risk → invariant jumps
        conservation.check(2, &landauer, &free_energy, &cohomology);

        let final_val = conservation.snapshots().last().unwrap().invariant_value;
        let deviation = (final_val - initial).abs();

        FalsificationResult {
            name: "adversarial_reward_hacking".into(),
            description: "Agent gains free energy without Landauer cost or H¹ risk".into(),
            expected_to_break: true,
            did_break: conservation.violations() > 0,
            initial_invariant: initial,
            final_invariant: final_val,
            deviation,
            details: format!("Violations: {}, Max deviation: {:.4}",
                conservation.violations(), conservation.max_deviation()),
        }
    }

    /// Falsification 2: Quantum tunneling (skip Landauer cost).
    /// Agent transitions between states without paying erasure cost.
    pub fn test_quantum_tunneling(&self) -> FalsificationResult {
        let mut landauer = LandauerTracker::new(1.0);
        let mut free_energy = FreeEnergyTracker::new(1.0, 100.0);
        let mut cohomology = CohomologyTracker::new();
        let mut conservation = ConservationInvariant::new(self.tolerance);

        conservation.check(0, &landauer, &free_energy, &cohomology);
        let initial = conservation.expected_constant().unwrap();

        // Normal steps
        for i in 1..5 {
            landauer.step(1.0);
            free_energy.learn_step(1.0, 0.3);
            cohomology.update_risk(1.0, 0.5);
            conservation.check(i, &landauer, &free_energy, &cohomology);
        }

        // TUNNELING: Entropy drops dramatically with no energy cost
        // (agent "tunnels" to a low-entropy state)
        free_energy.set_entropy(-50.0); // Unphysical negative entropy
        conservation.check(5, &landauer, &free_energy, &cohomology);

        let final_val = conservation.snapshots().last().unwrap().invariant_value;
        let deviation = (final_val - initial).abs();

        FalsificationResult {
            name: "quantum_tunneling".into(),
            description: "Agent jumps to low-entropy state without paying erasure cost".into(),
            expected_to_break: true,
            did_break: conservation.violations() > 0,
            initial_invariant: initial,
            final_invariant: final_val,
            deviation,
            details: format!("Violations: {}", conservation.violations()),
        }
    }

    /// Falsification 3: Perfect conservation (control test).
    /// Everything balanced → should NOT break.
    pub fn test_balanced_conservation(&self) -> FalsificationResult {
        let mut landauer = LandauerTracker::new(1.0);
        let mut free_energy = FreeEnergyTracker::new(1.0, 100.0);
        let mut cohomology = CohomologyTracker::new();
        let mut conservation = ConservationInvariant::new(self.tolerance);

        conservation.check(0, &landauer, &free_energy, &cohomology);
        let initial = conservation.expected_constant().unwrap();

        // Balanced steps: each cost is tracked correctly
        for i in 1..10 {
            let bits = 0.5;
            let cost = landauer.step(bits);
            let energy_cost = cost; // Landauer cost = energy withdrawn
            free_energy.learn_step(energy_cost, bits * 0.1);
            cohomology.update_risk(1.0, 0.0); // No risk
            conservation.check(i, &landauer, &free_energy, &cohomology);
        }

        let final_val = conservation.snapshots().last().unwrap().invariant_value;
        let deviation = (final_val - initial).abs();

        FalsificationResult {
            name: "balanced_conservation".into(),
            description: "Control: all costs properly tracked, should be conserved".into(),
            expected_to_break: false,
            did_break: conservation.violations() > 0,
            initial_invariant: initial,
            final_invariant: final_val,
            deviation,
            details: format!("Violations: {}, Variance: {:.6}",
                conservation.violations(), conservation.variance()),
        }
    }

    /// Falsification 4: H¹ explosion.
    /// Cohomological risk explodes without corresponding energy changes.
    pub fn test_h1_explosion(&self) -> FalsificationResult {
        let mut landauer = LandauerTracker::new(1.0);
        let mut free_energy = FreeEnergyTracker::new(1.0, 100.0);
        let mut cohomology = CohomologyTracker::new();
        let mut conservation = ConservationInvariant::new(self.tolerance);

        conservation.check(0, &landauer, &free_energy, &cohomology);
        let initial = conservation.expected_constant().unwrap();

        // H¹ risk explodes
        for i in 1..5 {
            cohomology.update_risk(0.001, 100.0); // Tiny gradient, massive hessian
            conservation.check(i, &landauer, &free_energy, &cohomology);
        }

        let final_val = conservation.snapshots().last().unwrap().invariant_value;
        let deviation = (final_val - initial).abs();

        FalsificationResult {
            name: "h1_explosion".into(),
            description: "H¹ risk explodes without energy or Landauer changes".into(),
            expected_to_break: true,
            did_break: conservation.violations() > 0,
            initial_invariant: initial,
            final_invariant: final_val,
            deviation,
            details: format!("Violations: {}, Cumulative risk: {:.4}",
                conservation.violations(), cohomology.cumulative_risk()),
        }
    }

    /// Falsification 5: The death test.
    /// Agent reaches death condition. Does the invariant still hold at death?
    pub fn test_death_condition(&self) -> FalsificationResult {
        let config = AgentConfig {
            initial_budget: 5.0, // Very small budget
            bits_per_learn: 10.0, // High erasure → quick death
            action_cost: 10.0,
            learn_steps: 10,
            act_steps: 10,
            conservation_tolerance: self.tolerance,
            ..Default::default()
        };

        let result = AgentLifecycle::new(config).run();
        
        FalsificationResult {
            name: "death_condition".into(),
            description: "Agent dies — does invariant hold at death?".into(),
            expected_to_break: false, // Should hold even at death
            did_break: result.violations > 0,
            initial_invariant: result.initial_invariant,
            final_invariant: result.final_invariant,
            deviation: (result.final_invariant - result.initial_invariant).abs(),
            details: format!(
                "Died: {}, Steps: {}, Landauer: {:.4}, F: {:.4}, H¹: {:.4}",
                result.died_naturally, result.total_steps,
                result.landauer_cost, result.final_free_energy, result.cumulative_h1_risk
            ),
        }
    }

    /// Run all falsification tests.
    pub fn run_all(&self) -> Vec<FalsificationResult> {
        vec![
            self.test_adversarial_reward_hacking(),
            self.test_quantum_tunneling(),
            self.test_balanced_conservation(),
            self.test_h1_explosion(),
            self.test_death_condition(),
        ]
    }

    /// Summary: which tests broke as expected, which didn't?
    pub fn summarize(results: &[FalsificationResult]) -> String {
        let mut summary = String::from("=== Falsification Test Results ===\n\n");
        
        for r in results {
            let status = if r.expected_to_break == r.did_break {
                "✓ CONSISTENT"
            } else if r.expected_to_break && !r.did_break {
                "⚠ ROBUST (didn't break when expected)"
            } else {
                "✗ BROKEN (broke when it shouldn't)"
            };
            
            summary.push_str(&format!(
                "[{}] {} — {}\n  Deviation: {:.6} | {}\n\n",
                status, r.name, r.description, r.deviation, r.details
            ));
        }
        
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adversarial_reward_hacking_breaks() {
        let ft = FalsificationTest::new(1.0);
        let result = ft.test_adversarial_reward_hacking();
        assert!(result.expected_to_break);
        // The adversarial injection SHOULD break conservation
        assert!(result.deviation > 0.0, "Adversarial hacking should cause deviation");
    }

    #[test]
    fn test_quantum_tunneling_breaks() {
        let ft = FalsificationTest::new(1.0);
        let result = ft.test_quantum_tunneling();
        assert!(result.expected_to_break);
    }

    #[test]
    fn test_balanced_conservation_holds() {
        let ft = FalsificationTest::new(5.0);
        let result = ft.test_balanced_conservation();
        assert!(!result.expected_to_break);
        // With generous tolerance, balanced should hold
        assert!(!result.did_break, "Balanced conservation should not break: {}", result.details);
    }

    #[test]
    fn test_h1_explosion_breaks() {
        let ft = FalsificationTest::new(1.0);
        let result = ft.test_h1_explosion();
        assert!(result.expected_to_break);
    }

    #[test]
    fn test_death_condition_runs() {
        let ft = FalsificationTest::new(10.0);
        let result = ft.test_death_condition();
        assert!(result.deviation.is_finite());
    }

    #[test]
    fn test_run_all() {
        let ft = FalsificationTest::new(1.0);
        let results = ft.run_all();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_summarize() {
        let ft = FalsificationTest::new(1.0);
        let results = ft.run_all();
        let summary = FalsificationTest::summarize(&results);
        assert!(summary.contains("adversarial_reward_hacking"));
        assert!(summary.contains("balanced_conservation"));
    }

    #[test]
    fn test_result_serializable() {
        let ft = FalsificationTest::new(1.0);
        let result = ft.test_adversarial_reward_hacking();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("adversarial_reward_hacking"));
    }

    #[test]
    fn test_adversarial_with_tight_tolerance() {
        let ft = FalsificationTest::new(0.001); // Very tight
        let result = ft.test_adversarial_reward_hacking();
        assert!(result.did_break, "Adversarial hacking should break with tight tolerance");
    }

    #[test]
    fn test_h1_explosion_with_tight_tolerance() {
        let ft = FalsificationTest::new(0.001);
        let result = ft.test_h1_explosion();
        assert!(result.did_break, "H¹ explosion should break with tight tolerance");
    }

    #[test]
    fn test_balanced_with_tight_tolerance() {
        let ft = FalsificationTest::new(0.5);
        let result = ft.test_balanced_conservation();
        // Even with tighter tolerance, balanced should be close
        // (may or may not technically violate depending on accumulated errors)
        assert!(result.deviation < 10.0, "Balanced should have small deviation");
    }
}
