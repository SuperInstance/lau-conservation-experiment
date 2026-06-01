//! Free energy tracker — Helmholtz free energy F = U - TS.
//!
//! Models the agent's thermodynamic free energy as a KL-regularized objective.
//! F = <energy> - T * S, where S is the entropy of the agent's belief distribution.

use serde::{Serialize, Deserialize};

/// Tracks the free energy state of an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyTracker {
    /// Temperature
    temperature: f64,
    /// Initial free energy budget
    initial_budget: f64,
    /// Current internal energy U
    internal_energy: f64,
    /// Current entropy S
    entropy: f64,
    /// History of free energy values per step
    history: Vec<f64>,
    /// Energy contributions per step
    energy_deltas: Vec<f64>,
}

impl FreeEnergyTracker {
    pub fn new(temperature: f64, initial_budget: f64) -> Self {
        Self {
            temperature,
            initial_budget,
            internal_energy: initial_budget,
            entropy: 0.0,
            history: vec![initial_budget],
            energy_deltas: Vec::new(),
        }
    }

    /// Compute current Helmholtz free energy: F = U - T*S
    pub fn free_energy(&self) -> f64 {
        self.internal_energy - self.temperature * self.entropy
    }

    /// Update internal energy (reward signal, resource consumption).
    pub fn add_energy(&mut self, delta: f64) {
        self.internal_energy += delta;
        self.energy_deltas.push(delta);
        self.history.push(self.free_energy());
    }

    /// Update entropy (information gain/loss).
    pub fn set_entropy(&mut self, s: f64) {
        self.entropy = s;
        self.history.push(self.free_energy());
    }

    /// Record a learning step: gain information (entropy increases) but pay energy.
    pub fn learn_step(&mut self, energy_cost: f64, entropy_gain: f64) -> f64 {
        self.internal_energy -= energy_cost;
        self.entropy += entropy_gain;
        let fe = self.free_energy();
        self.history.push(fe);
        self.energy_deltas.push(-energy_cost);
        fe
    }

    /// Record an action step: spend energy, may reduce entropy (commit to decision).
    pub fn act_step(&mut self, energy_cost: f64, entropy_delta: f64) -> f64 {
        self.internal_energy -= energy_cost;
        self.entropy += entropy_delta;
        let fe = self.free_energy();
        self.history.push(fe);
        self.energy_deltas.push(-energy_cost);
        fe
    }

    /// Conserve step: recover some energy through homeostatic regulation.
    pub fn conserve_step(&mut self, energy_recovered: f64, entropy_delta: f64) -> f64 {
        self.internal_energy += energy_recovered;
        self.entropy += entropy_delta;
        let fe = self.free_energy();
        self.history.push(fe);
        self.energy_deltas.push(energy_recovered);
        fe
    }

    pub fn initial_budget(&self) -> f64 {
        self.initial_budget
    }

    pub fn internal_energy(&self) -> f64 {
        self.internal_energy
    }

    pub fn entropy(&self) -> f64 {
        self.entropy
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t;
    }

    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Is the agent depleted? (energy ≤ 0)
    pub fn is_depleted(&self) -> bool {
        self.internal_energy <= 0.0
    }

    /// Free energy deficit from initial budget.
    pub fn deficit(&self) -> f64 {
        self.initial_budget - self.free_energy()
    }

    pub fn reset(&mut self) {
        self.internal_energy = self.initial_budget;
        self.entropy = 0.0;
        self.history.clear();
        self.history.push(self.initial_budget);
        self.energy_deltas.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_new_tracker() {
        let fe = FreeEnergyTracker::new(1.0, 100.0);
        assert_relative_eq!(fe.free_energy(), 100.0, epsilon = 1e-10);
        assert_relative_eq!(fe.initial_budget(), 100.0);
    }

    #[test]
    fn test_free_energy_formula() {
        let fe = FreeEnergyTracker::new(2.0, 100.0);
        // Add entropy
        let mut fe = fe;
        fe.set_entropy(5.0);
        // F = U - TS = 100 - 2*5 = 90
        assert_relative_eq!(fe.free_energy(), 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_learn_step() {
        let mut fe = FreeEnergyTracker::new(1.0, 100.0);
        let result = fe.learn_step(10.0, 2.0);
        // U = 90, S = 2, F = 90 - 1*2 = 88
        assert_relative_eq!(result, 88.0, epsilon = 1e-10);
    }

    #[test]
    fn test_act_step() {
        let mut fe = FreeEnergyTracker::new(1.0, 100.0);
        fe.set_entropy(3.0);
        let result = fe.act_step(5.0, -2.0); // commit, reduce entropy
        // U = 95, S = 1, F = 95 - 1 = 94
        assert_relative_eq!(result, 94.0, epsilon = 1e-10);
    }

    #[test]
    fn test_conserve_step() {
        let mut fe = FreeEnergyTracker::new(1.0, 100.0);
        fe.set_entropy(5.0);
        let result = fe.conserve_step(8.0, -1.0);
        // U = 108, S = 4, F = 108 - 4 = 104
        assert_relative_eq!(result, 104.0, epsilon = 1e-10);
    }

    #[test]
    fn test_depleted() {
        let mut fe = FreeEnergyTracker::new(1.0, 10.0);
        assert!(!fe.is_depleted());
        fe.add_energy(-10.0);
        assert!(fe.is_depleted());
    }

    #[test]
    fn test_deficit() {
        let mut fe = FreeEnergyTracker::new(1.0, 100.0);
        fe.set_entropy(10.0);
        // F = 100 - 10 = 90, deficit = 10
        assert_relative_eq!(fe.deficit(), 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut fe = FreeEnergyTracker::new(1.0, 100.0);
        fe.learn_step(50.0, 5.0);
        fe.reset();
        assert_relative_eq!(fe.free_energy(), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_history_tracking() {
        let mut fe = FreeEnergyTracker::new(1.0, 100.0);
        fe.learn_step(10.0, 2.0);
        fe.act_step(5.0, 0.0);
        // Initial + learn + act = 3 entries
        assert!(fe.history().len() >= 3);
    }

    #[test]
    fn test_temperature_effect() {
        let fe1 = FreeEnergyTracker::new(1.0, 100.0);
        let mut fe2 = FreeEnergyTracker::new(10.0, 100.0);
        fe2.set_entropy(5.0);
        let mut fe1 = fe1;
        fe1.set_entropy(5.0);
        // Higher T → lower free energy for same entropy
        assert!(fe2.free_energy() < fe1.free_energy());
    }
}
