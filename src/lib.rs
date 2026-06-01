//! # Lau Conservation Experiment
//!
//! The first real experiment testing Kimi's prediction that composing the 14 theorem crates
//! produces an EMERGENT conservation law that no single crate encodes.
//!
//! **Conservation law:** Landauer cost + Free energy + H¹ risk score ≈ constant
//!
//! **Death condition:** Agent terminates when cumulative Landauer cost = initial free energy budget.

pub mod landauer;
pub mod free_energy;
pub mod cohomology;
pub mod conservation;
pub mod agent;
pub mod spectral;
pub mod falsification;
pub mod temperature;

pub use landauer::LandauerTracker;
pub use free_energy::FreeEnergyTracker;
pub use cohomology::CohomologyTracker;
pub use conservation::ConservationInvariant;
pub use agent::{AgentLifecycle, AgentPhase, AgentConfig};
pub use spectral::SpectralGap;
pub use falsification::FalsificationTest;
pub use temperature::TemperatureSweep;
