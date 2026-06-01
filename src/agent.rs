//! Agent lifecycle simulation.
//!
//! Models a complete agent lifecycle: birth → learn → act → conserve → detect delusions → die.
//! Tracks all three quantities (Landauer, Free Energy, H¹ risk) and checks conservation.

use crate::{LandauerTracker, FreeEnergyTracker, CohomologyTracker, ConservationInvariant, SpectralGap};
use serde::{Serialize, Deserialize};

/// Phase of the agent lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentPhase {
    Birth,
    Learn,
    Act,
    Conserve,
    DetectDelusions,
    Dead,
}

impl std::fmt::Display for AgentPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentPhase::Birth => write!(f, "Birth"),
            AgentPhase::Learn => write!(f, "Learn"),
            AgentPhase::Act => write!(f, "Act"),
            AgentPhase::Conserve => write!(f, "Conserve"),
            AgentPhase::DetectDelusions => write!(f, "DetectDelusions"),
            AgentPhase::Dead => write!(f, "Dead"),
        }
    }
}

/// Configuration for agent lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Temperature (ℏ analog)
    pub temperature: f64,
    /// Initial free energy budget
    pub initial_budget: f64,
    /// Bits to erase per learning step
    pub bits_per_learn: f64,
    /// Energy cost per action
    pub action_cost: f64,
    /// Energy recovered per conserve step
    pub conserve_recovery: f64,
    /// Number of learning steps
    pub learn_steps: usize,
    /// Number of action steps
    pub act_steps: usize,
    /// Number of conserve steps
    pub conserve_steps: usize,
    /// Reward gradient norm (for H¹ computation)
    pub gradient_norm: f64,
    /// Reward hessian frobenius (for H¹ computation)
    pub hessian_frobenius: f64,
    /// Conservation tolerance
    pub conservation_tolerance: f64,
    /// Delusion detection threshold
    pub delusion_threshold: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            initial_budget: 100.0,
            bits_per_learn: 1.0,
            action_cost: 5.0,
            conserve_recovery: 2.0,
            learn_steps: 10,
            act_steps: 10,
            conserve_steps: 5,
            gradient_norm: 1.0,
            hessian_frobenius: 0.5,
            conservation_tolerance: 5.0,
            delusion_threshold: 0.3,
        }
    }
}

/// Result of a complete agent lifecycle run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleResult {
    pub config: AgentConfig,
    pub final_phase: AgentPhase,
    pub total_steps: usize,
    pub landauer_cost: f64,
    pub final_free_energy: f64,
    pub cumulative_h1_risk: f64,
    pub final_invariant: f64,
    pub initial_invariant: f64,
    pub invariant_variance: f64,
    pub invariant_std_dev: f64,
    pub max_deviation: f64,
    pub violations: u32,
    pub is_conserved: bool,
    pub spectral_gap: f64,
    pub died_naturally: bool,
    /// Invariant history for plotting
    pub invariant_history: Vec<f64>,
}

/// The agent lifecycle simulator.
pub struct AgentLifecycle {
    pub config: AgentConfig,
    pub landauer: LandauerTracker,
    pub free_energy: FreeEnergyTracker,
    pub cohomology: CohomologyTracker,
    pub conservation: ConservationInvariant,
    pub spectral: SpectralGap,
    pub phase: AgentPhase,
    pub step: usize,
}

impl AgentLifecycle {
    pub fn new(config: AgentConfig) -> Self {
        let landauer = LandauerTracker::new(config.temperature);
        let free_energy = FreeEnergyTracker::new(config.temperature, config.initial_budget);
        let cohomology = CohomologyTracker::new();
        let conservation = ConservationInvariant::new(config.conservation_tolerance);
        let spectral = SpectralGap::new(4);

        Self {
            config,
            landauer,
            free_energy,
            cohomology,
            conservation,
            spectral,
            phase: AgentPhase::Birth,
            step: 0,
        }
    }

    /// Run the complete lifecycle.
    pub fn run(mut self) -> LifecycleResult {
        let initial_invariant = ConservationInvariant::compute(
            &self.landauer, &self.free_energy, &self.cohomology
        );

        // Record initial state
        self.conservation.check(
            self.step, &self.landauer, &self.free_energy, &self.cohomology
        );
        self.step += 1;

        // Phase 1: Learn
        self.phase = AgentPhase::Learn;
        for _ in 0..self.config.learn_steps {
            let bits = self.config.bits_per_learn;
            let entropy_gain = bits * 0.5; // Learning adds entropy
            self.landauer.step(bits);
            self.free_energy.learn_step(bits * self.config.temperature * std::f64::consts::LN_2, entropy_gain);
            self.cohomology.update_risk(self.config.gradient_norm, self.config.hessian_frobenius);
            
            self.conservation.check(
                self.step, &self.landauer, &self.free_energy, &self.cohomology
            );
            self.step += 1;

            if self.check_death() {
                self.phase = AgentPhase::Dead;
                return self.build_result(initial_invariant, true);
            }
        }

        // Phase 2: Act
        self.phase = AgentPhase::Act;
        for _ in 0..self.config.act_steps {
            let energy_cost = self.config.action_cost;
            let entropy_delta = -0.3; // Acting reduces entropy (committing to decisions)
            
            // Compute bits erased from action
            let action_bits = energy_cost / (self.config.temperature * std::f64::consts::LN_2 + 1e-10);
            self.landauer.step(action_bits * 0.1);
            self.free_energy.act_step(energy_cost, entropy_delta);
            self.cohomology.update_risk(
                self.config.gradient_norm * 0.8,
                self.config.hessian_frobenius * 1.2,
            );
            
            self.conservation.check(
                self.step, &self.landauer, &self.free_energy, &self.cohomology
            );
            self.step += 1;

            if self.check_death() {
                self.phase = AgentPhase::Dead;
                return self.build_result(initial_invariant, true);
            }
        }

        // Phase 3: Conserve
        self.phase = AgentPhase::Conserve;
        for _ in 0..self.config.conserve_steps {
            let recovery = self.config.conserve_recovery;
            let entropy_delta = -0.1;
            
            self.landauer.step(0.01); // Minimal erasure during conservation
            self.free_energy.conserve_step(recovery, entropy_delta);
            self.cohomology.update_risk(
                self.config.gradient_norm * 1.5,
                self.config.hessian_frobenius * 0.5,
            );
            
            self.conservation.check(
                self.step, &self.landauer, &self.free_energy, &self.cohomology
            );
            self.step += 1;

            if self.check_death() {
                self.phase = AgentPhase::Dead;
                return self.build_result(initial_invariant, true);
            }
        }

        // Phase 4: Detect delusions
        self.phase = AgentPhase::DetectDelusions;
        for i in 0..3 {
            let belief_reward = 10.0 - i as f64;
            let true_reward = 5.0 + i as f64 * 0.5;
            let is_delusion = self.cohomology.detect_delusion(belief_reward, true_reward);
            
            if is_delusion {
                // Delusion correction costs energy and bits
                self.landauer.step(0.5);
                self.free_energy.act_step(2.0, -0.5);
            }
            
            self.conservation.check(
                self.step, &self.landauer, &self.free_energy, &self.cohomology
            );
            self.step += 1;
        }

        // Final death check
        if self.check_death() {
            self.phase = AgentPhase::Dead;
            return self.build_result(initial_invariant, true);
        }

        // Agent survives
        self.phase = AgentPhase::Dead;
        let depleted = self.free_energy.internal_energy() <= 0.0;
        self.build_result(initial_invariant, depleted)
    }

    /// Check if the agent should die: cumulative Landauer ≥ initial budget.
    fn check_death(&self) -> bool {
        self.landauer.cumulative_cost() >= self.config.initial_budget
            || self.free_energy.internal_energy() <= 0.0
    }

    fn build_result(self, initial_invariant: f64, died_naturally: bool) -> LifecycleResult {
        let invariant_history: Vec<f64> = self.conservation.snapshots()
            .iter().map(|s| s.invariant_value).collect();
        
        let variance = self.conservation.variance();
        
        // Compute spectral gap from invariant history
        let mut spectral = self.spectral;
        let gap = spectral.compute_from_history(&invariant_history);

        LifecycleResult {
            config: self.config,
            final_phase: self.phase,
            total_steps: self.step,
            landauer_cost: self.landauer.cumulative_cost(),
            final_free_energy: self.free_energy.free_energy(),
            cumulative_h1_risk: self.cohomology.cumulative_risk(),
            final_invariant: self.conservation.snapshots().last()
                .map(|s| s.invariant_value)
                .unwrap_or(0.0),
            initial_invariant,
            invariant_variance: variance,
            invariant_std_dev: variance.sqrt(),
            max_deviation: self.conservation.max_deviation(),
            violations: self.conservation.violations(),
            is_conserved: self.conservation.is_conserved(),
            spectral_gap: gap,
            died_naturally,
            invariant_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_lifecycle() {
        let config = AgentConfig::default();
        let lifecycle = AgentLifecycle::new(config);
        let result = lifecycle.run();
        
        assert!(result.total_steps > 0);
        assert_eq!(result.final_phase, AgentPhase::Dead);
    }

    #[test]
    fn test_death_by_energy_depletion() {
        let config = AgentConfig {
            initial_budget: 10.0,
            action_cost: 20.0, // High cost → quick death
            learn_steps: 0,
            act_steps: 5,
            ..Default::default()
        };
        let result = AgentLifecycle::new(config).run();
        assert!(result.died_naturally);
    }

    #[test]
    fn test_survival_with_high_budget() {
        let config = AgentConfig {
            initial_budget: 10000.0,
            ..Default::default()
        };
        let result = AgentLifecycle::new(config).run();
        assert!(!result.died_naturally);
    }

    #[test]
    fn test_lifecycle_phases_progress() {
        let config = AgentConfig {
            learn_steps: 2,
            act_steps: 2,
            conserve_steps: 2,
            initial_budget: 10000.0,
            ..Default::default()
        };
        let result = AgentLifecycle::new(config).run();
        // Should complete all phases
        assert!(result.total_steps >= 7); // 1 birth + 2 learn + 2 act + 2 conserve + 3 delusion
    }

    #[test]
    fn test_invariant_history_populated() {
        let result = AgentLifecycle::new(AgentConfig::default()).run();
        assert!(!result.invariant_history.is_empty());
        assert_eq!(result.invariant_history.len(), result.total_steps);
    }

    #[test]
    fn test_spectral_gap_computed() {
        let result = AgentLifecycle::new(AgentConfig::default()).run();
        assert!(result.spectral_gap > 0.0);
    }

    #[test]
    fn test_zero_temperature() {
        let config = AgentConfig {
            temperature: 0.01, // Near zero
            initial_budget: 100.0,
            ..Default::default()
        };
        let result = AgentLifecycle::new(config).run();
        // Should still run (low Landauer cost at low T)
        assert!(result.total_steps > 0);
    }

    #[test]
    fn test_high_temperature() {
        let config = AgentConfig {
            temperature: 100.0,
            initial_budget: 10000.0,
            ..Default::default()
        };
        let result = AgentLifecycle::new(config).run();
        assert!(result.total_steps > 0);
        // High T → high Landauer cost per bit → faster death possible
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", AgentPhase::Birth), "Birth");
        assert_eq!(format!("{}", AgentPhase::Learn), "Learn");
        assert_eq!(format!("{}", AgentPhase::Dead), "Dead");
    }

    #[test]
    fn test_result_serializable() {
        let result = AgentLifecycle::new(AgentConfig::default()).run();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("total_steps"));
    }

    #[test]
    fn test_config_serializable() {
        let config = AgentConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();
        assert!((deserialized.temperature - config.temperature).abs() < 1e-10);
    }
}
