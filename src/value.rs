//! The objective-function value kernel (ADR 0033 §D5.4): `w(creep)` — the expected
//! energy-equivalent value of spending one movement intent, the single scalar that collapses roles
//! so the benchmark (and eventually rover's own contention triage) can weight movements by what
//! they are worth.
//!
//! **Currency:** energy-equivalent per tick (e/t) for rates; energy-equivalent stocks for value at
//! risk. This is the codebase's one existing cross-goal currency (ADR 0032 `value_e`, ADR 0038
//! `room_net_roi`) — no new unit.
//!
//! **Contract:** `w = G · r_bid`, where `r_bid` is the role's slack-gated productive rate (the
//! §D5.4 role table) and `G = Δ_crit · A · S · U` are the criticality / arrive-in-time / survival /
//! slack gates. `w` is computable from a `SimCreep` + a per-creep goal annotation the scenario
//! harness precomputes (fleet/squad context arrives pre-priced — the eval never forks the economic
//! kernels). Weights are quantized to integer milli-e/t (+ a stable-id tie-break) before any
//! ordering, per the determinism fence.
//!
//! **v1 scope:** the hauler arm — the operator's "maximum value transport by distance" anchor —
//! which needs no policy decisions. The military / worker / claimer / scout arms are DESIGNED
//! (§D5.4 role table) but gated on the §D5.4 open decisions; they add `Role` variants, not schema
//! changes.

use screeps_sim_core::world::CreepId;

/// A creep's role/goal, carrying what the role table needs to price its productive rate. v1: haul.
#[derive(Clone, Debug)]
pub enum Role {
    /// A hauler cycling `q` cargo units per round trip (Q = 50 × CARRY parts loaded, or the
    /// scenario-pinned value-units for non-energy cargo).
    Haul { q: u32 },
}

impl Role {
    /// The role's productive rate `r` in e/t, given the fatigue-exact optimal duration of one
    /// productive cycle (`t_star` ticks — for a hauler, the oracle round-trip time). This is the
    /// opportunity cost of one tick of movement delay while the role's window binds.
    pub fn rate_e_t(&self, t_star: u32) -> f64 {
        match self {
            // ρ = Q / T*_rtt — room_net_roi's haul term inverted. Saturation caps
            // (marginal_route_share) arrive via the scenario annotation when fleets share a route.
            Role::Haul { q } => {
                if t_star == 0 {
                    0.0
                } else {
                    *q as f64 / t_star as f64
                }
            }
        }
    }

    /// The benchmark sample weight for one movement episode of optimal duration `t_star`:
    /// `W = r · T*` — the energy-tick value in flight over one optimal episode. For a hauler this
    /// is exactly `Q` (the cargo), making the aggregate `H` the operator's cargo-weighted
    /// "value transport by distance" (§D5.4 hauler reduction).
    pub fn sample_weight(&self, t_star: u32) -> f64 {
        self.rate_e_t(t_star) * t_star as f64
    }
}

/// Quantize a weight to integer milli-e/t — every ordering over weights MUST compare these (never
/// raw f64) per the determinism fence; ties break on the stable creep id via [`ordering_key`].
pub fn quantize_w(w: f64) -> i64 {
    (w * 1000.0).round() as i64
}

/// The total order for contention triage: quantized weight first (descending by caller), stable
/// creep id as the deterministic tie-break — no float compare, no map-iteration order, anywhere.
pub fn ordering_key(w: f64, id: CreepId) -> (i64, CreepId) {
    (quantize_w(w), id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hauler_rate_is_cargo_over_optimal_round_trip() {
        let role = Role::Haul { q: 100 };
        assert!((role.rate_e_t(50) - 2.0).abs() < 1e-9, "100 cargo / 50-tick cycle = 2 e/t");
        assert_eq!(role.rate_e_t(0), 0.0, "a degenerate zero-length cycle prices at 0");
    }

    #[test]
    fn hauler_sample_weight_reduces_to_cargo() {
        // W = r·T* = (Q/T*)·T* = Q — the §D5.4 hauler reduction, exactly.
        let role = Role::Haul { q: 150 };
        for t_star in [10u32, 37, 400] {
            assert!(
                (role.sample_weight(t_star) - 150.0).abs() < 1e-9,
                "the hauler sample weight is its cargo, independent of route length"
            );
        }
    }

    #[test]
    fn quantized_ordering_is_deterministic() {
        // Two weights closer than 1 milli-e/t collapse to the same quantum; the id breaks the tie.
        let a = ordering_key(1.2340004, 7);
        let b = ordering_key(1.2340001, 3);
        assert_eq!(a.0, b.0, "sub-milli differences are noise, not order");
        assert_ne!(a, b, "the stable id still yields a total order");
        assert!(ordering_key(2.0, 1).0 > ordering_key(1.999, 9).0);
    }
}
