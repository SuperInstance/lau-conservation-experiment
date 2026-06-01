//! Spectral gap measurement.
//!
//! The spectral gap determines how tightly the conservation law holds.
//! Larger gap → tighter conservation (less variance in the invariant).
//! Variance ∝ 1/spectral_gap.

use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

/// Measures the spectral gap of the agent's state transition operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralGap {
    /// The spectral gap (λ₁ - λ₂ of the transition operator)
    gap: f64,
    /// Eigenvalues of the last computed operator
    eigenvalues: Vec<f64>,
    /// Dimension of the operator
    dimension: usize,
}

impl SpectralGap {
    pub fn new(dimension: usize) -> Self {
        Self {
            gap: 1.0, // default: unit gap
            eigenvalues: Vec::new(),
            dimension,
        }
    }

    /// Compute spectral gap from a transition matrix.
    /// The gap is λ₁ - λ₂ where λ₁ ≥ λ₂ are the top two eigenvalues.
    pub fn compute_from_matrix(&mut self, matrix: &DMatrix<f64>) -> f64 {
        let n = matrix.nrows();
        self.dimension = n;
        
        // Symmetrize for real eigenvalues
        let sym = (matrix + matrix.transpose()) * 0.5;
        
        // Power iteration for top eigenvalues
        let eigenvalues = self.power_iteration_eigenvalues(&sym, 3, 100);
        self.eigenvalues = eigenvalues.clone();
        
        if eigenvalues.len() >= 2 {
            self.gap = (eigenvalues[0] - eigenvalues[1]).abs();
        } else if eigenvalues.len() == 1 {
            self.gap = eigenvalues[0].abs();
        }
        
        self.gap
    }

    /// Simple power iteration to find top k eigenvalues.
    fn power_iteration_eigenvalues(&self, matrix: &DMatrix<f64>, k: usize, iterations: usize) -> Vec<f64> {
        let n = matrix.nrows();
        let mut eigenvalues = Vec::with_capacity(k);
        let mut residual = matrix.clone();

        for _ in 0..k.min(n) {
            let mut v = DMatrix::from_fn(n, 1, |_, _| {
                rand::random::<f64>() * 2.0 - 1.0
            });

            for _ in 0..iterations {
                let mut w = &residual * &v;
                let norm = w.norm();
                if norm < 1e-15 { break; }
                w /= norm;
                v = w;
            }

            let eigenvalue = (&v.transpose() * &residual * &v)[(0, 0)]
                / (&v.transpose() * &v)[(0, 0)];
            eigenvalues.push(eigenvalue);

            // Deflate
            residual = residual - eigenvalue * &v * &v.transpose();
        }

        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues
    }

    /// Compute spectral gap from a history of invariant values.
    /// Uses the autocorrelation decay rate as a proxy.
    pub fn compute_from_history(&mut self, history: &[f64]) -> f64 {
        if history.len() < 4 {
            self.gap = 1.0;
            return self.gap;
        }

        // Compute autocorrelation at lag 1
        let mean = history.iter().sum::<f64>() / history.len() as f64;
        let variance: f64 = history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / history.len() as f64;
        
        if variance < 1e-15 {
            self.gap = 1e10; // perfectly constant → infinite gap
            return self.gap;
        }

        let autocovariance: f64 = history.windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum::<f64>() / (history.len() - 1) as f64;

        let rho1 = autocovariance / variance;
        
        // Spectral gap ≈ 1 - |ρ₁| for reversible chains
        self.gap = (1.0 - rho1.abs()).max(1e-10);
        self.gap
    }

    /// Get the tolerance for conservation law checks from the spectral gap.
    /// Tolerance ∝ 1/spectral_gap.
    pub fn tolerance_from_gap(&self) -> f64 {
        1.0 / self.gap.max(1e-10)
    }

    /// Expected variance of the conservation invariant.
    pub fn expected_variance(&self, base_variance: f64) -> f64 {
        base_variance / self.gap.max(1e-10)
    }

    pub fn gap(&self) -> f64 {
        self.gap
    }

    pub fn set_gap(&mut self, g: f64) {
        self.gap = g;
    }

    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl Default for SpectralGap {
    fn default() -> Self {
        Self::new(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_new() {
        let sg = SpectralGap::new(4);
        assert_relative_eq!(sg.gap(), 1.0);
    }

    #[test]
    fn test_identity_matrix() {
        let mut sg = SpectralGap::new(2);
        let identity = DMatrix::identity(2, 2);
        let gap = sg.compute_from_matrix(&identity);
        // Identity has all eigenvalues = 1, gap ≈ 0
        assert_relative_eq!(gap, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_scalar_matrix() {
        let mut sg = SpectralGap::new(2);
        let m = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 1.0]);
        let gap = sg.compute_from_matrix(&m);
        // Eigenvalues 3, 1 → gap = 2
        assert_relative_eq!(gap, 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_from_constant_history() {
        let mut sg = SpectralGap::new(2);
        let history = vec![5.0; 100];
        let gap = sg.compute_from_history(&history);
        // Constant → very large gap
        assert!(gap > 100.0);
    }

    #[test]
    fn test_from_varying_history() {
        let mut sg = SpectralGap::new(2);
        let history: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let gap = sg.compute_from_history(&history);
        assert!(gap > 0.0);
        assert!(gap <= 2.0);
    }

    #[test]
    fn test_tolerance_from_gap() {
        let sg = SpectralGap::new(4);
        // gap = 1.0 → tolerance = 1.0
        assert_relative_eq!(sg.tolerance_from_gap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_small_gap_large_tolerance() {
        let mut sg = SpectralGap::new(4);
        sg.set_gap(0.01);
        assert_relative_eq!(sg.tolerance_from_gap(), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_large_gap_small_tolerance() {
        let mut sg = SpectralGap::new(4);
        sg.set_gap(100.0);
        assert_relative_eq!(sg.tolerance_from_gap(), 0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_expected_variance() {
        let sg = SpectralGap::new(4);
        let v = sg.expected_variance(1.0);
        assert_relative_eq!(v, 1.0, epsilon = 1e-10); // gap=1 → var = 1
    }

    #[test]
    fn test_default() {
        let sg = SpectralGap::default();
        assert_eq!(sg.dimension(), 4);
    }

    #[test]
    fn test_from_short_history() {
        let mut sg = SpectralGap::new(2);
        let gap = sg.compute_from_history(&[1.0, 2.0]);
        assert_relative_eq!(gap, 1.0); // default for short history
    }
}
