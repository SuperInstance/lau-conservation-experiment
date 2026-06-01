# lau-conservation-experiment

> The first real experiment: testing the emergent conservation law that no single crate encodes

## What This Does

The first real experiment: testing the emergent conservation law that no single crate encodes. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-conservation-experiment
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_conservation_experiment::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct SweepResult 
pub struct TemperatureSweep 
    pub fn new(base_config: AgentConfig, temperatures: Vec<f64>) -> Self 
    pub fn default_sweep() -> Self 
    pub fn run(&self) -> Vec<SweepResult> 
    pub fn conservation_holds_across_temps(&self) -> bool 
    pub fn relative_deviation_range(&self) -> Vec<(f64, f64)> 
pub struct FalsificationResult 
pub struct FalsificationTest 
    pub fn new(tolerance: f64) -> Self 
    pub fn test_adversarial_reward_hacking(&self) -> FalsificationResult 
    pub fn test_quantum_tunneling(&self) -> FalsificationResult 
    pub fn test_balanced_conservation(&self) -> FalsificationResult 
    pub fn test_h1_explosion(&self) -> FalsificationResult 
    pub fn test_death_condition(&self) -> FalsificationResult 
    pub fn run_all(&self) -> Vec<FalsificationResult> 
    pub fn summarize(results: &[FalsificationResult]) -> String 
pub struct CohomologyTracker 
    pub fn new() -> Self 
    pub fn compute_risk(&self, gradient_norm: f64, hessian_frobenius: f64) -> f64 
    pub fn update_risk(&mut self, gradient_norm: f64, hessian_frobenius: f64) -> f64 
    pub fn set_risk(&mut self, risk: f64) 
    pub fn detect_delusion(&mut self, belief_reward: f64, true_reward: f64) -> bool 
    pub fn betti_number_estimate(&self) -> f64 
    pub fn risk_score(&self) -> f64 
    pub fn cumulative_risk(&self) -> f64 
    pub fn history(&self) -> &[f64] 
    pub fn obstructions_detected(&self) -> u32 
    pub fn reset(&mut self) 
pub struct FreeEnergyTracker 
    pub fn new(temperature: f64, initial_budget: f64) -> Self 
    pub fn free_energy(&self) -> f64 
    pub fn add_energy(&mut self, delta: f64) 
    pub fn set_entropy(&mut self, s: f64) 
    pub fn learn_step(&mut self, energy_cost: f64, entropy_gain: f64) -> f64 
    pub fn act_step(&mut self, energy_cost: f64, entropy_delta: f64) -> f64 
    pub fn conserve_step(&mut self, energy_recovered: f64, entropy_delta: f64) -> f64 
    pub fn initial_budget(&self) -> f64 
    pub fn internal_energy(&self) -> f64 
    pub fn entropy(&self) -> f64 
    pub fn temperature(&self) -> f64 
    pub fn set_temperature(&mut self, t: f64) 
    pub fn history(&self) -> &[f64] 
    pub fn is_depleted(&self) -> bool 
    pub fn deficit(&self) -> f64 
    pub fn reset(&mut self) 
pub struct InvariantSnapshot 
pub struct ConservationInvariant 
    pub fn new(tolerance: f64) -> Self 
    pub fn compute(
    pub fn check(
    pub fn set_expected_constant(&mut self, c: f64) 
    pub fn variance(&self) -> f64 
    pub fn std_dev(&self) -> f64 
    pub fn mean(&self) -> f64 
    pub fn max_deviation(&self) -> f64 
    pub fn is_conserved(&self) -> bool 
    pub fn violations(&self) -> u32 
    pub fn violation_details(&self) -> &[String] 
    pub fn snapshots(&self) -> &[InvariantSnapshot] 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**85 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
