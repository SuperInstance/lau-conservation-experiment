# Lau Conservation Experiment

> The first real experiment: testing the emergent conservation law that no single crate encodes.

## The Prediction

Kimi predicted that composing the 14 theorem crates produces an **emergent conservation law**:

**Landauer cost + Free energy + H¹ risk score ≈ constant across the agent lifecycle.**

This conservation law is NOT programmed into any individual crate; it emerges from composition.

The agent dies exactly when cumulative Landauer cost = initial free energy budget.

## Architecture

| Module | Description |
|--------|-------------|
| `landauer` | Landauer cost tracker (kT·ln(2) per bit erased) |
| `free_energy` | Helmholtz free energy tracker (F = U - TS) |
| `cohomology` | H¹ risk score tracker (cohomological reward hacking risk) |
| `conservation` | Combined invariant checker |
| `agent` | Full lifecycle simulation (birth → learn → act → conserve → detect delusions → die) |
| `spectral` | Spectral gap measurement (variance ∝ 1/spectral_gap) |
| `temperature` | Temperature sweep experiments |
| `falsification` | Critical falsification tests |

## The Conservation Law

```
Invariant = cumulative_landauer_cost + current_free_energy + cumulative_H¹_risk ≈ constant
```

## The Death Condition

The agent terminates when cumulative Landauer cost ≥ initial free energy budget.

## The Falsifiable Prediction

If the conservation law is real, it should:
1. Hold across different temperatures (ℏ values)
2. Break under adversarial conditions (reward hacking, quantum tunneling, H¹ explosion)
3. Have variance inversely proportional to the spectral gap

## Tests

85 tests including:
- Unit tests for each tracker
- Integration tests for the full lifecycle
- Conservation law verification
- Falsification tests (adversarial reward hacking, quantum tunneling, H¹ explosion)
- Temperature sweep across multiple configurations

## License

MIT
