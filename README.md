# lau-conservation-experiment

> The first real experiment: testing the emergent conservation law that no single crate encodes.

## What This Does

This crate implements and tests a bold prediction: **composing thermodynamic, information-theoretic, and topological quantities produces an emergent conservation law**. Specifically:

> **Landauer cost + Free energy + H¹ risk score ≈ constant**

The crate simulates complete agent lifecycles (birth → learn → act → conserve → die), tracks three quantities at every step, and checks whether their sum is conserved. It includes a full **falsification suite** — adversarial tests designed to break the law — and **temperature sweeps** to check conservation across different thermodynamic regimes.

The death condition is baked in: an agent terminates when its cumulative Landauer cost equals its initial free energy budget.

## Key Idea

No single quantity in the system is conserved on its own. Landauer cost monotonically increases. Free energy fluctuates. H¹ risk spikes and settles. But the claim is that their *sum* remains approximately constant — and the variance from constancy scales as `1/spectral_gap`, a relationship predicted by the Markov chain mixing theory underlying the agent's state transitions.

This is an experiment in **emergence**: the conservation law isn't encoded anywhere. It arises from the interaction of information thermodynamics (Landauer), statistical physics (Helmholtz free energy), and algebraic topology (first cohomology risk).

## Install

```toml
[dependencies]
lau-conservation-experiment = "0.1.0"
```

Or:

```sh
cargo add lau-conservation-experiment
```

Dependencies: `nalgebra` 0.33, `serde` + `serde_json`, `rand` + `rand_distr`.

## Quick Start

```rust
use lau_conservation_experiment::*;

// Run a complete agent lifecycle with default config
let config = AgentConfig::default();
let result = AgentLifecycle::new(config).run();

println!("Conserved: {} (violations: {})", result.is_conserved, result.violations);
println!("Invariant: initial={:.4}, final={:.4}, std_dev={:.6}",
    result.initial_invariant, result.final_invariant, result.invariant_std_dev);
println!("Spectral gap: {:.4}", result.spectral_gap);
println!("Died naturally: {}", result.died_naturally);

// Run falsification tests
let falsification = FalsificationTest::new(1.0);
let results = falsification.run_all();
println!("{}", FalsificationTest::summarize(&results));

// Temperature sweep
let sweep = TemperatureSweep::default_sweep();
let sweep_results = sweep.run();
for r in &sweep_results {
    println!("T={:.1}: conserved={}, max_dev={:.4}, steps={}",
        r.temperature, r.is_conserved, r.max_deviation, r.total_steps);
}
```

## API Reference

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `landauer` | `LandauerTracker` | Thermodynamic cost of erasure (kT·ln(2) per bit) |
| `free_energy` | `FreeEnergyTracker` | Helmholtz free energy F = U − TS |
| `cohomology` | `CohomologyTracker` | H¹ cohomological risk (reward-hacking obstruction) |
| `conservation` | `ConservationInvariant` | Checks L + F + H¹ ≈ const |
| `agent` | `AgentLifecycle`, `AgentConfig`, `LifecycleResult` | Full lifecycle simulation |
| `spectral` | `SpectralGap` | Spectral gap measurement (variance ∝ 1/gap) |
| `falsification` | `FalsificationTest`, `FalsificationResult` | Adversarial break-the-law tests |
| `temperature` | `TemperatureSweep`, `SweepResult` | Multi-temperature regime tests |

### `LandauerTracker`

Tracks the cumulative thermodynamic cost of belief erasure. Each step records bits erased and computes cost = kT·ln(2)·bits. Supports KL-divergence input (converts nats to bits).

```rust
let mut l = LandauerTracker::new(temperature);
l.step(1.0);         // erase 1 bit → cost = T·ln(2)
l.step_kl(0.693);    // KL = ln(2) nats → 1 bit → cost = T·ln(2)
```

### `FreeEnergyTracker`

Tracks Helmholtz free energy F = U − TS through learning, acting, and conservation phases.

```rust
let mut fe = FreeEnergyTracker::new(temperature, initial_budget);
fe.learn_step(energy_cost, entropy_gain);    // learn: pay energy, gain info
fe.act_step(energy_cost, entropy_delta);      // act: pay energy, commit
fe.conserve_step(energy_recovered, entropy);  // conserve: recover, stabilize
```

### `CohomologyTracker`

Computes H¹ risk as `||∇²R|| / (||∇R|| + ε)` — the ratio of reward Hessian to gradient, capturing second-order obstructions that gradient-based optimization misses. Also detects "delusions" where the agent's believed reward diverges from true reward.

```rust
let mut co = CohomologyTracker::new();
co.update_risk(gradient_norm, hessian_frobenius);
co.detect_delusion(believed_reward, true_reward);
```

### `ConservationInvariant`

The core invariant checker. Computes L_cum + F_current + H¹_cum and checks constancy within tolerance.

```rust
let mut inv = ConservationInvariant::new(tolerance);
inv.check(step, &landauer, &free_energy, &cohomology);
println!("Conserved: {}, variance: {:.6}", inv.is_conserved(), inv.variance());
```

### `AgentLifecycle`

Simulates a complete agent lifecycle through phases: Birth → Learn → Act → Conserve → DetectDelusions → Dead.

```rust
let config = AgentConfig {
    temperature: 1.0,
    initial_budget: 100.0,
    learn_steps: 10,
    act_steps: 10,
    conserve_steps: 5,
    conservation_tolerance: 5.0,
    ..Default::default()
};
let result = AgentLifecycle::new(config).run();
```

Death condition: `cumulative_landauer ≥ initial_budget` OR `internal_energy ≤ 0`.

### `FalsificationTest`

Five adversarial tests:
1. **Adversarial reward hacking** — inject free energy without Landauer cost
2. **Quantum tunneling** — jump to low-entropy state without paying erasure
3. **Balanced conservation** — control: should NOT break
4. **H¹ explosion** — cohomological risk detonates
5. **Death condition** — does the invariant hold at agent termination?

### `TemperatureSweep`

Runs the full lifecycle across temperatures [0.1, 0.5, 1.0, 2.0, 5.0, 10.0], scaling the budget proportionally.

## How It Works

### Agent Lifecycle Simulation

1. **Birth**: Initialize trackers (Landauer=0, F=budget, H¹=0), record initial invariant.
2. **Learn**: Erase bits (Landauer increases), gain entropy (F decreases), update H¹ risk.
3. **Act**: Spend energy, commit to decisions (entropy decreases), H¹ risk may increase.
4. **Conserve**: Recover energy through homeostatic regulation, stabilize entropy.
5. **Detect Delusions**: Compare believed vs. true reward; pay correction cost for delusions.
6. **Death**: When cumulative Landauer cost ≥ initial budget, or energy depleted.

At each step, the conservation invariant is checked: `Landauer_cum + F_current + H¹_cum ≈ initial_constant`.

### Spectral Gap Computation

The spectral gap is computed from the invariant history using autocorrelation decay:
- `ρ₁` = autocorrelation at lag 1
- `gap ≈ 1 - |ρ₁|` (for reversible Markov chains)
- Constant history → infinite gap (perfect conservation)
- Expected tolerance: `∝ 1/spectral_gap`

Alternatively, from a transition matrix via power iteration deflation for top-k eigenvalues.

### Falsification Methodology

The falsification approach follows Karl Popper: a theory is only scientific if you can specify what would disprove it. Each test creates a scenario where the invariant *should* break:
- Unphysical operations (free energy injection, negative entropy)
- One-sided perturbations (H¹ explosion without energy change)
- The control test (balanced) should NOT break

If the law survives tests that should break it, the theory may be too robust (vacuously true). If it breaks on the control test, the implementation is wrong. The sweet spot: it breaks on adversarial tests and holds on legitimate dynamics.

## The Math

### The Conservation Invariant

```
I(t) = L(t) + F(t) + H¹(t)
```

where:
- `L(t) = Σ kT·ln(2)·Δbits` — cumulative Landauer cost
- `F(t) = U(t) - T·S(t)` — Helmholtz free energy
- `H¹(t) = Σ ||∇²R||/(||∇R||+ε)` — cumulative cohomological risk

**Claim**: `I(t) ≈ I(0) = U₀` (the initial energy budget)

### Landauer's Principle

Erasing n bits of information costs:

```
E ≥ nkT·ln(2)
```

Each agent step that discards prior beliefs (updates, commits, corrections) incurs this cost. KL-divergence between old and new beliefs converts naturally to bits erased: `bits = D_KL(P_old || P_new) / ln(2)`.

### Helmholtz Free Energy

```
F = U - TS
```

- U = internal energy (budget minus expenditures plus recoveries)
- T = temperature (ℏ analog — noise/disorder scale)
- S = entropy of the agent's belief distribution

Learning increases S (more uncertainty about the world). Acting decreases S (committing to decisions). Conservation attempts to restore S toward a homeostatic set point.

### H¹ Cohomological Risk

Inspired by de Rham cohomology, the first cohomology group H¹ of the reward landscape measures obstructions to finding a globally consistent reward function. The risk score:

```
H¹_risk = ||∇²R||_F / (||∇R|| + ε)
```

High H¹ risk means the gradient is small but curvature is large — the agent is in a region where local gradient information is unreliable (a "hole" in the reward topology).

### Spectral Gap and Variance

For an ergodic Markov chain with spectral gap γ = λ₁ - λ₂:

```
Var(I(t)) ∝ 1/γ
```

Larger spectral gap → faster mixing → tighter conservation. The tolerance for declaring a "violation" scales as `1/γ`.

## Tests

85 tests across all modules. Run with:

```sh
cargo test
```

Key test categories:
- **Unit tests** for each tracker (Landauer, free energy, cohomology, conservation, spectral gap)
- **Integration tests** for full agent lifecycles
- **Falsification tests** (adversarial + control)
- **Temperature sweep tests** across regimes

## License

MIT
