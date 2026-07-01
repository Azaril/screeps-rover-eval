//! Distributional statistics for the benchmark (ADR 0033 §D5 — the objective function). A rover
//! change is judged not by a single mean but by a **distribution** (where is the tail?) and a
//! **confidence interval** (is the change real, given corpus variance?). This module is the
//! measurement substrate: weighted means, empirical percentiles, and a seeded bootstrap CI. It is
//! pure math over `(value, weight)` samples — no pathing — reused by the haul benchmark.
//!
//! Determinism: the bootstrap resamples with the shared `sim-core::rng` (seeded), so a report is
//! byte-reproducible — a prerequisite of the determinism fence.

use screeps_sim_core::rng::Rng;

/// Default bootstrap resample count — enough for a stable 95% CI on a few-hundred-task corpus.
pub const DEFAULT_RESAMPLES: usize = 2000;

/// The `q`-quantile (`0..=1`) of a PRE-SORTED slice, by linear interpolation between order statistics.
/// Empty → 0.0.
pub fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// The weight-weighted mean of `(value, weight)` samples (Σ wᵢvᵢ / Σ wᵢ). Zero total weight → 0.0.
pub fn weighted_mean(samples: &[(f64, f64)]) -> f64 {
    let (mut sw, mut swv) = (0.0, 0.0);
    for &(v, w) in samples {
        sw += w;
        swv += w * v;
    }
    if sw > 0.0 {
        swv / sw
    } else {
        0.0
    }
}

/// A bootstrap `(1-alpha)` confidence interval for the weighted mean: resample the tasks WITH
/// replacement `resamples` times, recompute the weighted mean of each resample, and take the
/// `alpha/2` and `1-alpha/2` percentiles of those means. Seeded ⇒ reproducible.
pub fn bootstrap_weighted_mean_ci(
    samples: &[(f64, f64)],
    resamples: usize,
    alpha: f64,
    seed: u32,
) -> (f64, f64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut rng = Rng::seeded(seed);
    let mut means: Vec<f64> = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let (mut sw, mut swv) = (0.0, 0.0);
        for _ in 0..n {
            let idx = (rng.next_u64() % n as u64) as usize;
            let (v, w) = samples[idx];
            sw += w;
            swv += w * v;
        }
        means.push(if sw > 0.0 { swv / sw } else { 0.0 });
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        percentile_sorted(&means, alpha / 2.0),
        percentile_sorted(&means, 1.0 - alpha / 2.0),
    )
}

/// A distributional summary of a set of `(value, weight)` samples: the work-weighted mean (the
/// aggregate objective) with a bootstrap 95% CI, plus the UNWEIGHTED empirical distribution of the
/// values (so the tail — the fraction of *tasks* that are bad — is visible).
#[derive(Clone, Debug)]
pub struct Summary {
    pub n: usize,
    /// The weight-weighted mean — the aggregate objective `H`.
    pub weighted_mean: f64,
    /// Bootstrap 95% CI of `weighted_mean`.
    pub ci95: (f64, f64),
    pub min: f64,
    pub p05: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
    pub max: f64,
}

impl Summary {
    /// Summarize `(value, weight)` samples; `seed` drives the bootstrap.
    pub fn of(samples: &[(f64, f64)], seed: u32) -> Summary {
        let mut vals: Vec<f64> = samples.iter().map(|&(v, _)| v).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Summary {
            n: samples.len(),
            weighted_mean: weighted_mean(samples),
            ci95: bootstrap_weighted_mean_ci(samples, DEFAULT_RESAMPLES, 0.05, seed),
            min: vals.first().copied().unwrap_or(0.0),
            p05: percentile_sorted(&vals, 0.05),
            p25: percentile_sorted(&vals, 0.25),
            p50: percentile_sorted(&vals, 0.50),
            p75: percentile_sorted(&vals, 0.75),
            p95: percentile_sorted(&vals, 0.95),
            max: vals.last().copied().unwrap_or(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_are_ordered_and_bracket_the_data() {
        let data: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        assert!((percentile_sorted(&data, 0.5) - 50.0).abs() < 1e-9, "median of 0..100 is 50");
        assert!((percentile_sorted(&data, 0.0) - 0.0).abs() < 1e-9);
        assert!((percentile_sorted(&data, 1.0) - 100.0).abs() < 1e-9);
        assert!(percentile_sorted(&data, 0.05) < percentile_sorted(&data, 0.95));
    }

    #[test]
    fn weighted_mean_respects_weights() {
        // Values 0 and 1, but the 1 carries 3× the weight → mean 0.75.
        let s = [(0.0, 1.0), (1.0, 3.0)];
        assert!((weighted_mean(&s) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn bootstrap_ci_brackets_the_mean_and_is_reproducible() {
        // A spread of efficiencies around ~0.9.
        let samples: Vec<(f64, f64)> = (0..200)
            .map(|i| (0.80 + (i % 20) as f64 * 0.01, 1.0))
            .collect();
        let mean = weighted_mean(&samples);
        let (lo, hi) = bootstrap_weighted_mean_ci(&samples, 2000, 0.05, 42);
        assert!(lo < mean && mean < hi, "CI [{lo},{hi}] must bracket the mean {mean}");
        assert!(hi - lo < 0.05, "a 200-sample CI should be tight, width={}", hi - lo);
        // Same seed → identical CI.
        let again = bootstrap_weighted_mean_ci(&samples, 2000, 0.05, 42);
        assert_eq!((lo, hi), again, "the seeded bootstrap is reproducible");
    }

    #[test]
    fn ci_narrows_as_the_sample_grows() {
        let mk = |n: usize| -> Vec<(f64, f64)> {
            (0..n).map(|i| (0.5 + (i % 50) as f64 * 0.01, 1.0)).collect()
        };
        let width = |n: usize| {
            let (lo, hi) = bootstrap_weighted_mean_ci(&mk(n), 1500, 0.05, 7);
            hi - lo
        };
        assert!(width(1000) < width(50), "more samples ⇒ a tighter CI");
    }

    #[test]
    fn summary_orders_its_percentiles() {
        let samples: Vec<(f64, f64)> = (0..100).map(|i| (i as f64 / 100.0, 1.0)).collect();
        let s = Summary::of(&samples, 1);
        assert!(s.min <= s.p05 && s.p05 <= s.p25 && s.p25 <= s.p50);
        assert!(s.p50 <= s.p75 && s.p75 <= s.p95 && s.p95 <= s.max);
        assert!(s.ci95.0 <= s.weighted_mean && s.weighted_mean <= s.ci95.1);
    }
}
