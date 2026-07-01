//! rover parameter TOURNAMENT (ADR 0033 §D5.4 objective → §M5 sweep) — with the objective `H`
//! (value-weighted efficiency + bootstrap CI) defined, rover's tunables become searchable:
//! escalation speed ([`StuckThresholds`]), path commitment (`reuse_path_length`), search budget
//! (`pathfinding_ops_budget`), resolver reach (`max_shove_depth`, `friendly_creep_distance`).
//!
//! This generalizes the combat-eval `param_sweep.rs` idiom (its tuning arm, NOT `tournament.rs`'s
//! self-play — configs are non-adversarial, so each point is scored PURE and deterministically
//! against a fixed corpus rather than head-to-head): a pure scorer ([`evaluate_config`]), a
//! gates-first deterministic ranking key ([`TuneScore::ranked_key`] — the `winning_efficient_key`
//! shape), an env-driven STAGED sweep as an `#[ignore]` test (coordinate descent over layers:
//! escalation → commitment/budget → resolver — ordered by observed leverage: escalation governs
//! wasted-intent burn and the flap livelock, commitment governs detour stickiness, resolver depth
//! is the fine knob), an upfront budget estimate, and a determinism pin.
//!
//! The corpus deliberately mixes the synthetic mechanism scenarios with REAL foreman-planned
//! layouts ([`crate::base_traffic`]) including the two KNOWN-STARVED rooms (the park-sealed
//! corridor + repath-flap livelock) — the tournament's standing question is whether a config heals
//! them without degrading the healthy rooms. If one does, the tuned optimum becomes rover's
//! shipped DEFAULT (operator directive), with this tournament as the recorded rationale.
//!
//! Two corpora: [`corpus`] (fast — 5 synthetics + 3 real rooms, what the checked-in tests run) and
//! [`corpus_full`] (`TUNE_FULL_CORPUS=1` — all 13 real rooms + 2 multi-room border routes, priced
//! by the solo-T* baseline, [`crate::haul::t_star_rtt_solo`]). Every score reports H **per scenario
//! family** (synthetic / real / border — §D5.4 decision (11) primary view, [`family_report`]) on
//! top of the pooled H that keys the ranking.
//!
//! END-STATE RE-RUN 2026-07-01 (registration ON + denial-as-stuck + shoveable idles + the banked
//! per-request `ladder(8)` + value triage — the shipped default stack; full corpus, escalation
//! {2,8} × reuse {5,20}): **reuse(20) HOLDS** — reuse(5) H 0.9634 vs reuse(20) 0.9626 both
//! quantize to 963 milli (sub-milli = noise by [`TuneScore::ranked_key`]'s design) and the tie
//! breaks on fewer simulated ticks (1519 < 1524); the once-decisive gap (reuse(5) STARVED rooms:
//! H 0.8967 / completion 0.979 / 4583 parked intents at the pre-v2 re-confirmation) collapsed
//! because denial-as-stuck + registration now heal the repath-flap livelock at the source —
//! commitment is no longer the load-bearing fix, but stays the cheaper default (fewer expiry
//! repaths live). ladder(2) loses the escalation stage (0.9405 < 0.9626). Final default-stack
//! baseline: **pooled H 0.9626 CI95 [0.9518, 0.9729], per-family border 0.9636 / real 0.9654 /
//! synthetic 0.9273, completion 1.000, parked 0, gates held.** (History: the tuned
//! `reuse_path_length: 20` was first RE-CONFIRMED on the widened corpus at rover `e80b1dd`,
//! H 0.9036 vs 0.8967; ladder(8) rode the global lane then — since banked per-request, see
//! [`FleetOpts`].)

use crate::base_traffic::{base_scenario, captured_layouts, energy_traffic_fleet};
pub use crate::haul::ladder;
use crate::haul::{run_haul_fleet_opts, FleetOpts, HaulAssignment};
use crate::stats::Summary;
use crate::value::quantize_w;
use screeps::{Position, RoomCoordinate, RoomName};
use screeps_sim_core::{MoverConfig, SimBody, SimTerrain};
use std::collections::BTreeMap;

/// The scenario families H is reported per (§D5.4 decision (11): per-family PRIMARY, pooled
/// secondary — a config that buys synthetic-pinch H by taxing the real rooms must be visible as
/// such, not laundered through the pool). Static strings + `BTreeMap` iteration keep every report
/// byte-deterministic.
pub const FAMILY_SYNTHETIC: &str = "synthetic";
/// Real foreman-planned room layouts ([`crate::base_traffic`], the `base:*` scenarios).
pub const FAMILY_REAL: &str = "real";
/// Multi-room border routes (solo-T*-baselined — η is pure contention loss, see [`corpus_full`]).
pub const FAMILY_BORDER: &str = "border";

/// One corpus scenario: a named world + fleet + tick cap, tagged with its reporting family.
pub struct TuneScenario {
    pub name: String,
    /// One of [`FAMILY_SYNTHETIC`]/[`FAMILY_REAL`]/[`FAMILY_BORDER`] — explicit (not name-derived)
    /// so a renamed scenario cannot silently switch reporting pools.
    pub family: &'static str,
    pub terrain: SimTerrain,
    pub fleet: Vec<HaulAssignment>,
    pub tick_cap: u32,
}

/// The score of ONE [`MoverConfig`] point over the corpus (the `ParamScore` shape).
#[derive(Clone, Debug)]
pub struct TuneScore {
    /// The objective: pooled value-weighted efficiency over every trip in the corpus. Stays the
    /// RANKED key input (one scalar to sort on); [`per_family`](TuneScore::per_family) is the
    /// primary REPORT (decision (11)).
    pub h: f64,
    pub ci95: (f64, f64),
    pub p05: f64,
    /// Per-family [`Summary`] over the same samples, partitioned by [`TuneScenario::family`]
    /// (each sample lands in exactly one family, so the pooled `h` is the weight-share convex
    /// combination of these). BTreeMap: deterministic report order.
    pub per_family: BTreeMap<&'static str, Summary>,
    /// Completed / expected trips over the corpus (1.0 = nothing starved).
    pub completion: f64,
    pub deadlocks: u32,
    /// Unexplained active-creep rejections — the resolver↔engine divergence gate.
    pub failed_coordination: u32,
    /// Parked-blocker optimism cost (regression signal, not a gate).
    pub failed_into_parked: u32,
    pub intents_issued: u32,
    /// Σ ticks simulated (cost + a makespan-flavored tie-break).
    pub total_ticks: u32,
    /// `failed_coordination == 0` AND no deadlock. Completion is NOT gated — healing the starved
    /// rooms is what the tournament searches for, and incompletion already crushes `h` via η=0.
    pub gates_held: bool,
}

impl TuneScore {
    /// The deterministic descending-preference key (sort ASC, take LAST — or compare with
    /// `Reverse`): gates first, then quantized `H` (milli — sub-milli differences are noise), then
    /// FEWER wasted intents, then fewer simulated ticks. No raw-float compare anywhere.
    pub fn ranked_key(&self) -> (u8, i64, i64, i64) {
        (
            u8::from(self.gates_held),
            quantize_w(self.h),
            -i64::from(self.failed_into_parked + self.failed_coordination),
            -i64::from(self.total_ticks),
        )
    }
}

fn pos_in(room: &str, x: u8, y: u8) -> Position {
    let room: RoomName = room.parse().unwrap();
    Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
}

fn balanced_hauler() -> SimBody {
    use screeps::Part;
    SimBody::unboosted(&[Part::Carry, Part::Carry, Part::Move, Part::Move])
}

/// The real rooms the FAST corpus keeps (shared with [`corpus_full`], which adds the rest): one
/// healthy room as the regression floor + the two formerly-starved rooms as the healing ratchets.
const FAST_ROOM_PICKS: &[&str] = &["E11N13", "E13S29", "E11N17"];

/// The default tuning corpus: the five synthetic mechanism scenarios (free-flow floor, one-gap
/// pinch, heterogeneous-value pinch, crossing routes, road-through-swamp) + three REAL layouts —
/// one healthy room as the regression floor and the two formerly-starved rooms as the healing
/// ratchets.
pub fn corpus() -> Vec<TuneScenario> {
    let mut scenarios = Vec::new();
    let hauler = |source: Position, sink: Position| HaulAssignment {
        body: balanced_hauler(),
        q: 100,
        source,
        sink,
        trips: 2,
    };

    // Free-flow floor: parallel lanes, must stay perfect under every config.
    scenarios.push(TuneScenario {
        name: "open_lanes".into(),
        family: FAMILY_SYNTHETIC,
        terrain: SimTerrain::default(),
        fleet: [20u8, 22, 24, 26]
            .iter()
            .map(|&y| hauler(pos_in("W1N1", 10, y), pos_in("W1N1", 20, y)))
            .collect(),
        tick_cap: 400,
    });

    // One-gap pinch: 4 haulers share a route through a single-tile wall gap (parked + escalation).
    let mut pinch = SimTerrain::default();
    for y in 0..=49 {
        if y != 25 {
            pinch.walls.insert((15, y));
        }
    }
    scenarios.push(TuneScenario {
        name: "pinch".into(),
        family: FAMILY_SYNTHETIC,
        terrain: pinch,
        fleet: (0..4).map(|_| hauler(pos_in("W1N1", 10, 25), pos_in("W1N1", 20, 25))).collect(),
        tick_cap: 2_000,
    });

    // Heterogeneous-value pinch (the §D5.4 decision-(9) triage scenario): ONE 800-cargo hauler +
    // three 100-cargo haulers cycle the SAME one-gap route, so every tile contest at the gap is a
    // value decision — big-cargo-first triage (value_priority) should clear the 800-q trips
    // (72.7% of the scenario's weight) ahead of the 100-q crowd, raising weighted H; under
    // enum-anchor ordering all four tie at `Normal` and serialize by resolver tie-break instead.
    let mut pinch_hetero = SimTerrain::default();
    for y in 0..=49 {
        if y != 25 {
            pinch_hetero.walls.insert((15, y));
        }
    }
    let big_hauler = || {
        use screeps::Part;
        let mut parts = vec![Part::Carry; 16];
        parts.extend(std::iter::repeat_n(Part::Move, 16));
        SimBody::unboosted(&parts)
    };
    scenarios.push(TuneScenario {
        name: "pinch_hetero".into(),
        family: FAMILY_SYNTHETIC,
        terrain: pinch_hetero,
        fleet: std::iter::once(HaulAssignment {
            body: big_hauler(),
            q: 800,
            source: pos_in("W1N1", 10, 25),
            sink: pos_in("W1N1", 20, 25),
            trips: 2,
        })
        .chain((0..3).map(|_| hauler(pos_in("W1N1", 10, 25), pos_in("W1N1", 20, 25))))
        .collect(),
        tick_cap: 2_000,
    });

    // Crossing routes: two perpendicular 2-hauler routes through a shared center (resolver stress).
    scenarios.push(TuneScenario {
        name: "cross_traffic".into(),
        family: FAMILY_SYNTHETIC,
        terrain: SimTerrain::default(),
        fleet: vec![
            hauler(pos_in("W1N1", 15, 25), pos_in("W1N1", 35, 25)),
            hauler(pos_in("W1N1", 16, 25), pos_in("W1N1", 35, 25)),
            hauler(pos_in("W1N1", 25, 15), pos_in("W1N1", 25, 35)),
            hauler(pos_in("W1N1", 25, 16), pos_in("W1N1", 25, 35)),
        ],
        tick_cap: 1_000,
    });

    // A road punched through a full-height swamp band (M3 physics + ops/repath pressure).
    let mut swamp_road = SimTerrain::default();
    for x in 13..=17 {
        for y in 0..=49 {
            swamp_road.swamps.insert((x, y));
        }
    }
    for x in 13..=17 {
        swamp_road.roads.insert((x, 25));
    }
    scenarios.push(TuneScenario {
        name: "swamp_road".into(),
        family: FAMILY_SYNTHETIC,
        terrain: swamp_road,
        fleet: (0..2).map(|_| hauler(pos_in("W1N1", 10, 25), pos_in("W1N1", 20, 25))).collect(),
        tick_cap: 1_000,
    });

    // Real foreman-planned layouts: the first healthy room (regression floor) + both starved rooms
    // (the park-sealed-corridor repath-flap livelock — the healing targets).
    for layout in captured_layouts() {
        if !FAST_ROOM_PICKS.contains(&layout.room.as_str()) {
            continue;
        }
        let terrain = base_scenario(&layout);
        let fleet = energy_traffic_fleet(&layout, &terrain);
        scenarios.push(TuneScenario {
            name: format!("base:{}", layout.room),
            family: FAMILY_REAL,
            terrain,
            fleet,
            tick_cap: 2_500,
        });
    }
    scenarios
}

/// The **FULL** tuning corpus (`TUNE_FULL_CORPUS=1` on the staged sweep; the checked-in fast tests
/// keep [`corpus`] so the suite stays sub-second): [`corpus`] + ALL remaining captured real
/// layouts (13 rooms total — every `captured_layouts()` entry) + two multi-room border-route
/// scenarios, the C-family axis the fast corpus lacks. Cross-room assignments are priced by the
/// SOLO baseline ([`crate::haul::t_star_rtt_solo`] — η = pure contention loss; the single-room
/// oracle cannot serve there), so their samples pool into the same `H`. Multi-room worlds here
/// share ONE `SimTerrain` for every room (mirrored rooms — `MovementState.rooms` stays empty);
/// the cost source is per-room-aware now ([`crate::cost::WorldCostSource`] snapshots
/// `MovementState::rooms` + room-keys creep occupancy), so heterogeneous border scenarios only
/// wait on the haul driver populating `rooms` (owned elsewhere).
pub fn corpus_full() -> Vec<TuneScenario> {
    let mut scenarios = corpus();

    // The 10 real rooms the fast corpus skips ([`FAST_ROOM_PICKS`] carries the other three).
    // Same tick cap as the fast corpus's rooms: every captured room completes in < 200 ticks under
    // the shipped default (the base_traffic baseline report), so 2500 is an order of magnitude of
    // headroom, not a tuned number.
    for layout in captured_layouts() {
        if FAST_ROOM_PICKS.contains(&layout.room.as_str()) {
            continue;
        }
        let terrain = base_scenario(&layout);
        let fleet = energy_traffic_fleet(&layout, &terrain);
        scenarios.push(TuneScenario {
            name: format!("base:{}", layout.room),
            family: FAMILY_REAL,
            terrain,
            fleet,
            tick_cap: 2_500,
        });
    }

    let hauler = |source: Position, sink: Position| HaulAssignment {
        body: balanced_hauler(),
        q: 100,
        source,
        sink,
        trips: 2,
    };

    // Multi-room plain border route: 2 haulers cycling W1N1(10,25) → W2N1(40,25) — out through
    // W1N1's WEST exit (x=0 relocates to W2N1 x=49, the kernel edge-exit rule) and back. Prices
    // the border round trip end-to-end (the `aaac0f7` border-thrash class) under mild same-route
    // contention.
    scenarios.push(TuneScenario {
        name: "border_plain".into(),
        family: FAMILY_BORDER,
        terrain: SimTerrain::default(),
        fleet: (0..2).map(|_| hauler(pos_in("W1N1", 10, 25), pos_in("W2N1", 40, 25))).collect(),
        tick_cap: 800,
    });

    // Swampy variant: all-swamp except a 1-wide road corridor at y=25 (the shared terrain makes
    // the corridor border-continuous by construction). Off-road passing costs real fatigue
    // (loaded balanced hauler: +20/step, regen 4), so out-leg/back-leg exchanges stress
    // commitment + escalation ACROSS the border — the cross-room analogue of the E13S29
    // corridor-mouth mechanism the tuned reuse default healed.
    let mut swamp_corridor = SimTerrain::default();
    for x in 0..=49u8 {
        for y in 0..=49u8 {
            if y != 25 {
                swamp_corridor.swamps.insert((x, y));
            }
        }
        swamp_corridor.roads.insert((x, 25));
    }
    scenarios.push(TuneScenario {
        name: "border_swamp_road".into(),
        family: FAMILY_BORDER,
        terrain: swamp_corridor,
        fleet: (0..2).map(|_| hauler(pos_in("W1N1", 10, 25), pos_in("W2N1", 40, 25))).collect(),
        tick_cap: 2_000,
    });

    scenarios
}

/// Score one config over the corpus: pool every trip's `(η, W)` sample into ONE [`Summary`]
/// (value-weighted, exactly the §D5.4 aggregate) AND into one Summary per scenario family
/// (decision (11) — per-family primary, pooled secondary; the pool stays the ranked key). Sum the
/// audits/gates. PURE + deterministic — same config + corpus + seed ⇒ byte-identical score
/// (families are static strings in a BTreeMap; each family's bootstrap reuses `seed`).
pub fn evaluate_config(config: &MoverConfig, corpus: &[TuneScenario], seed: u32) -> TuneScore {
    evaluate_config_opts(config, &FleetOpts::default(), corpus, seed)
}

/// [`evaluate_config`] under explicit [`FleetOpts`] — a tournament point is really the pair
/// `(MoverConfig, FleetOpts)`: system knobs + per-request haul-lane shaping (the split-defaults
/// end-state's two layers). The default-opts path is what the checked-in gates pin.
pub fn evaluate_config_opts(
    config: &MoverConfig,
    opts: &FleetOpts,
    corpus: &[TuneScenario],
    seed: u32,
) -> TuneScore {
    let mut samples: Vec<(f64, f64)> = Vec::new();
    let mut family_samples: BTreeMap<&'static str, Vec<(f64, f64)>> = BTreeMap::new();
    let mut completed = 0u32;
    let mut expected = 0u32;
    let mut deadlocks = 0u32;
    let mut coordination = 0u32;
    let mut parked = 0u32;
    let mut intents = 0u32;
    let mut ticks = 0u32;

    for s in corpus {
        let out = run_haul_fleet_opts(&s.terrain, &s.fleet, s.tick_cap, seed, config, opts)
            .unwrap_or_else(|| panic!("corpus scenario `{}` must be oracle-solvable", s.name));
        samples.extend_from_slice(&out.samples);
        family_samples.entry(s.family).or_default().extend_from_slice(&out.samples);
        completed += out.completed_trips;
        expected += out.expected_trips;
        deadlocks += u32::from(out.deadlocked);
        coordination += out.audit.failed_coordination;
        parked += out.audit.failed_into_parked;
        intents += out.audit.intents_issued;
        ticks += out.ticks;
    }

    let summary = Summary::of(&samples, seed);
    TuneScore {
        h: summary.weighted_mean,
        ci95: summary.ci95,
        p05: summary.p05,
        per_family: family_samples
            .into_iter()
            .map(|(family, s)| (family, Summary::of(&s, seed)))
            .collect(),
        completion: if expected > 0 { completed as f64 / expected as f64 } else { 1.0 },
        deadlocks,
        failed_coordination: coordination,
        failed_into_parked: parked,
        intents_issued: intents,
        total_ticks: ticks,
        gates_held: coordination == 0 && deadlocks == 0,
    }
}

/// The per-family H report line (decision (11)'s PRIMARY view; the pooled `h` stays the ranked
/// key). BTreeMap iteration ⇒ deterministic family order, byte-stable across runs.
pub fn family_report(score: &TuneScore) -> String {
    score
        .per_family
        .iter()
        .map(|(family, s)| {
            format!("{family}: H={:.4} CI[{:.4},{:.4}] n={}", s.weighted_mean, s.ci95.0, s.ci95.1, s.n)
        })
        .collect::<Vec<_>>()
        .join("  ")
}

/// Rank named `(MoverConfig, FleetOpts)` points over the corpus, best first
/// ([`TuneScore::ranked_key`] descending). Points carry opts because the end-state's escalation
/// knob is the PER-REQUEST haul ladder ([`FleetOpts::haul_stuck_thresholds`]), not the system
/// `MoverConfig::stuck_thresholds` (which the corpus's all-haul requests would override anyway).
pub fn run_tournament(
    points: Vec<(String, MoverConfig, FleetOpts)>,
    corpus: &[TuneScenario],
    seed: u32,
) -> Vec<(String, MoverConfig, FleetOpts, TuneScore)> {
    let mut ranked: Vec<(String, MoverConfig, FleetOpts, TuneScore)> = points
        .into_iter()
        .map(|(name, config, opts)| {
            let score = evaluate_config_opts(&config, &opts, corpus, seed);
            (name, config, opts, score)
        })
        .collect();
    ranked.sort_by_key(|(name, _, _, score)| (std::cmp::Reverse(score.ranked_key()), name.clone()));
    ranked
}

/// Upper-bound cost of a sweep before running it: worst-case simulated creep-ticks (fleet × cap,
/// summed over the corpus, × points). The `#[ignore]` sweep prints this FIRST, with a measured
/// seconds-per-point from its first evaluation, so a widened env grid states its price up front.
pub fn estimate_budget(points: usize, corpus: &[TuneScenario]) -> (usize, u64) {
    let creep_ticks: u64 = corpus.iter().map(|s| s.fleet.len() as u64 * s.tick_cap as u64).sum();
    (points, creep_ticks * points as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// The shipped default must hold the hard gates on the whole corpus (the tournament's floor).
    #[test]
    fn default_config_holds_gates_on_the_corpus() {
        let score = evaluate_config(&MoverConfig::default(), &corpus(), 1);
        assert!(score.gates_held, "default config: {score:?}");
        assert!(score.h > 0.5, "the pooled objective is meaningfully positive, got {}", score.h);
        // HEALED (2026-07-01): under the pre-tuning default (`reuse_path_length: 5`) the two
        // starved real rooms held completion at 0.844 (H 0.680, 832 wasted intents) — the repath
        // flap livelock. The tournament-tuned commitment default (20) completes the whole corpus
        // (H 0.769, 28 wasted). This assert is the ratchet: a regression back to starvation fails.
        assert!(
            (score.completion - 1.0).abs() < 1e-9,
            "the tuned default must complete the corpus, got {}",
            score.completion
        );
    }

    /// The full-corpus ratchet, added with the corpus widening (2026-07-01): the shipped default
    /// must hold the hard gates AND complete on ALL 13 real rooms + both multi-room border routes.
    /// H here is NOT comparable 1:1 with the fast corpus's H (border samples are solo-baselined —
    /// pure contention loss — so they pull the pool up, real rooms keep the stricter oracle).
    /// BASELINE HISTORY (each re-pin mechanism-named per the no-silent-gates rule):
    /// H=0.8840 at widening (registration OFF, rover `e80b1dd`) → 0.9196 under the
    /// coordination-v2 stack (`register_idle_creeps` ON + denial-as-stuck + shoveable idles:
    /// every parked/failed-intent class dropped to literal 0 and the formerly-burned ticks became
    /// throughput) → **0.9626 CI95=[0.9518,0.9729], completion 1.000, parked 0, 1519 ticks**
    /// under the final shipped default stack (+ banked per-request haul ladder(8) + value triage,
    /// the two [`FleetOpts`] defaults; corpus also gained `pinch_hetero`). The floor below is the
    /// current measurement minus CI-scale slack for corpus evolution — a regression through it
    /// means the corpus lost real value-weighted efficiency, not noise.
    #[test]
    fn default_config_holds_gates_on_the_full_corpus() {
        let score = evaluate_config(&MoverConfig::default(), &corpus_full(), 1);
        eprintln!(
            "[tuning FULL-CORPUS default] H={:.4} CI95=[{:.4},{:.4}] p05={:.3} done={:.3} parked={} ticks={}",
            score.h, score.ci95.0, score.ci95.1, score.p05, score.completion,
            score.failed_into_parked, score.total_ticks
        );
        eprintln!("[tuning FULL-CORPUS default] per-family {}", family_report(&score));
        assert!(score.gates_held, "default config on the full corpus: {score:?}");
        // The widened completion ratchet — the fast corpus's starvation ratchet, over every room
        // + the border routes: a config regression that re-starves ANY of them fails here.
        assert!(
            (score.completion - 1.0).abs() < 1e-9,
            "the tuned default must complete the FULL corpus, got {}",
            score.completion
        );
        assert!(
            score.h > 0.94,
            "full-corpus pooled objective under the shipped default stack (measured 0.9626), got {}",
            score.h
        );
    }

    /// Same config + corpus + seed ⇒ byte-identical score (the param_sweep determinism pin) —
    /// including every per-family Summary (the decision-(11) report path).
    #[test]
    fn evaluation_is_deterministic() {
        let a = evaluate_config(&MoverConfig::default(), &corpus(), 7);
        let b = evaluate_config(&MoverConfig::default(), &corpus(), 7);
        assert_eq!(a.ranked_key(), b.ranked_key());
        assert_eq!(a.h.to_bits(), b.h.to_bits(), "bit-identical H");
        assert_eq!(a.intents_issued, b.intents_issued);
        let fams: Vec<_> = a.per_family.keys().copied().collect();
        assert_eq!(fams, b.per_family.keys().copied().collect::<Vec<_>>());
        for f in fams {
            assert_eq!(
                a.per_family[f].weighted_mean.to_bits(),
                b.per_family[f].weighted_mean.to_bits(),
                "bit-identical per-family H ({f})"
            );
            assert_eq!(a.per_family[f].n, b.per_family[f].n);
        }
        assert_eq!(family_report(&a), family_report(&b), "byte-identical family report");
    }

    /// Decision (11)'s partition invariant: every sample lands in exactly ONE family, so family
    /// sample counts sum to the corpus's expected trips and the pooled H is a convex combination
    /// of the family Hs (it can never leave their [min, max] envelope).
    #[test]
    fn per_family_h_partitions_the_pooled_samples() {
        let corpus = corpus_full();
        let score = evaluate_config(&MoverConfig::default(), &corpus, 1);

        let fams: Vec<_> = score.per_family.keys().copied().collect();
        assert_eq!(
            fams,
            vec![FAMILY_BORDER, FAMILY_REAL, FAMILY_SYNTHETIC],
            "the full corpus reports all three families, in BTreeMap (byte-stable) order"
        );

        let expected_trips: usize =
            corpus.iter().flat_map(|s| &s.fleet).map(|a| a.trips as usize).sum();
        let family_n: usize = score.per_family.values().map(|s| s.n).sum();
        assert_eq!(family_n, expected_trips, "families partition the samples: one per expected trip");

        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for s in score.per_family.values() {
            lo = lo.min(s.weighted_mean);
            hi = hi.max(s.weighted_mean);
        }
        assert!(
            lo - 1e-12 <= score.h && score.h <= hi + 1e-12,
            "pooled H {} must sit inside the family envelope [{lo}, {hi}]",
            score.h
        );
    }

    /// The determinism pin over the NEW cross-room T* path: the border scenarios' η is priced by
    /// `t_star_rtt_solo` (a nested sim run, not the closed-form oracle), so pin that the whole
    /// solo-baseline + fleet pipeline is byte-reproducible too. Only the 2 border scenarios run —
    /// cheap enough for the checked-in fast pass.
    #[test]
    fn cross_room_evaluation_is_deterministic() {
        let border: Vec<TuneScenario> = corpus_full()
            .into_iter()
            .filter(|s| s.name.starts_with("border_"))
            .collect();
        assert_eq!(border.len(), 2, "both border scenarios present in the full corpus");
        let a = evaluate_config(&MoverConfig::default(), &border, 7);
        let b = evaluate_config(&MoverConfig::default(), &border, 7);
        assert_eq!(a.ranked_key(), b.ranked_key());
        assert_eq!(a.h.to_bits(), b.h.to_bits(), "bit-identical H over the solo-T* path");
    }

    /// The escalation knob must actually REACH rover end-to-end (config wiring, not just plumbing
    /// types). MECHANISM UPDATE ×2 (2026-07-01): (a) under `register_idle_creeps` ON +
    /// denial-as-stuck + shoveable idles, a crippled ladder no longer manifests as parked-intent
    /// burn — parked blockers are resolver-known up front, so `failed_into_parked` is 0 under
    /// EVERY ladder (the old `>10×` assertion compared 0 to 0); what the ladder still governs is
    /// how fast a DENIED mover escalates through friendly-avoid/ops/shove, so crippling it shows
    /// up as slow-but-completing — same trips, more ticks, lower H. (b) with the per-request haul
    /// ladder BANKED as the driver default ([`FleetOpts`]), the live knob is
    /// `FleetOpts::haul_stuck_thresholds`, NOT `MoverConfig::stuck_thresholds` (which every haul
    /// request now overrides — crippling it would compare identical runs); the probe sweeps the
    /// per-request lane. It compares ladder(2) (fast) vs ladder(200) (crippled) rather than the
    /// shipped ladder(8), because on THIS homogeneous pinch the denial+shove machinery resolves
    /// every contest before tier 8 ever fires — ladder(8) and ladder(200) run byte-identically
    /// here (H 0.7809 both; ladder(8) earns its default on the real rooms, the per-family A/B in
    /// `per_request_haul_ladder_banks_its_h_gain`), so only a faster-than-8 ladder can prove the
    /// wiring end-to-end. Measured: ladder(2) H 0.8528 / 45 ticks vs ladder(200) H 0.7809 / 51;
    /// deterministic runs ⇒ strict inequalities on the quantized objective + realized ticks are
    /// stable. Completion stays 1.0 by mechanism (denial-fed shoves clear the seal regardless of
    /// ladder speed), so it is deliberately NOT asserted starved anymore.
    #[test]
    fn escalation_speed_knob_is_live() {
        let corpus: Vec<TuneScenario> =
            corpus().into_iter().filter(|s| s.name == "pinch").collect();
        let fast_opts =
            FleetOpts { haul_stuck_thresholds: Some(ladder(2)), ..FleetOpts::default() };
        let fast = evaluate_config_opts(&MoverConfig::default(), &fast_opts, &corpus, 1);
        let crippled =
            FleetOpts { haul_stuck_thresholds: Some(ladder(200)), ..FleetOpts::default() };
        let slow = evaluate_config_opts(&MoverConfig::default(), &crippled, &corpus, 1);
        eprintln!(
            "[escalation probe] ladder(2): H={:.4} ticks={} | ladder(200): H={:.4} ticks={}",
            fast.h, fast.total_ticks, slow.h, slow.total_ticks
        );
        assert!(
            quantize_w(slow.h) < quantize_w(fast.h),
            "a crippled per-request ladder must lose real objective on the pinch (default H {:.4} vs crippled {:.4})",
            fast.h,
            slow.h
        );
        assert!(
            slow.total_ticks > fast.total_ticks,
            "a crippled per-request ladder must slow the same trips (default {} ticks vs crippled {})",
            fast.total_ticks,
            slow.total_ticks
        );
        assert!(
            (slow.completion - 1.0).abs() < 1e-9,
            "denial-fed shoves clear the pinch even with escalation crippled, got {}",
            slow.completion
        );
    }

    /// The BANKED per-request haul ladder (item: the combat adjudication's split-defaults
    /// end-state, [`FleetOpts::haul_stuck_thresholds`] = `Some(ladder(8))`): the H gain the
    /// global-ladder tournament found must SURVIVE as a per-request setting — measured on the
    /// full corpus 2026-07-01 (triage held OFF on both sides to isolate the ladder axis):
    /// no-ladder H 0.9164 / 1660 ticks vs per-request ladder(8) H 0.9455 / 1524 ticks, gates
    /// green. This pin keeps the bank honest: if a rover change makes the per-request ladder a
    /// wash (or a loss), the default must be re-litigated, not silently carried.
    #[test]
    fn per_request_haul_ladder_banks_its_h_gain() {
        let full = corpus_full();
        let no_triage = |thresholds| FleetOpts { haul_stuck_thresholds: thresholds, value_priority: false };
        let without = evaluate_config_opts(&MoverConfig::default(), &no_triage(None), &full, 1);
        let with =
            evaluate_config_opts(&MoverConfig::default(), &no_triage(Some(ladder(8))), &full, 1);
        eprintln!(
            "[haul-ladder A/B] none: H={:.4} ticks={} | ladder(8): H={:.4} ticks={}",
            without.h, without.total_ticks, with.h, with.total_ticks
        );
        assert!(with.gates_held, "the banked default must hold the gates: {with:?}");
        assert!(
            quantize_w(with.h) > quantize_w(without.h),
            "the per-request ladder(8) bank must beat the system-default ladder (H {:.4} vs {:.4})",
            with.h,
            without.h
        );
    }

    /// §D5.4 decision-(9) value triage, benchmark side ([`FleetOpts::value_priority`], the
    /// HARNESS-recommended default): bidding each hauler `quantize_w(Q/T*_rtt)` on the numeric
    /// priority lane must beat enum-anchor ordering — globally on the full corpus (measured
    /// 2026-07-01 on top of the banked ladder: H 0.9455 → 0.9626) and decisively on the scenario
    /// built to need it (`pinch_hetero`, one 800-cargo vs three 100-cargo haulers through one
    /// gap: the 800-q hauler carries 72.7% of the weight, so big-cargo-first clearing raises
    /// weighted H — measured under the shipped ladder 0.7484 → 0.9346; ladder-off it is
    /// 0.8215 → 0.9411, i.e. triage recovers MORE than the slow ladder costs this synthetic).
    /// Runtime adoption stays combat-gated (M5 follow-up (7)); this pin covers the harness
    /// default only.
    #[test]
    fn value_triage_improves_h() {
        let full = corpus_full();
        let off = FleetOpts { value_priority: false, ..FleetOpts::default() };
        let a = evaluate_config_opts(&MoverConfig::default(), &off, &full, 1);
        let b = evaluate_config_opts(&MoverConfig::default(), &FleetOpts::default(), &full, 1);
        eprintln!(
            "[triage A/B full] off: H={:.4} | on: H={:.4} (ticks {} vs {})",
            a.h, b.h, a.total_ticks, b.total_ticks
        );
        assert!(b.gates_held, "triage-on must hold the gates: {b:?}");
        assert!(
            quantize_w(b.h) > quantize_w(a.h),
            "value triage must add real objective on the full corpus (H {:.4} vs {:.4})",
            b.h,
            a.h
        );

        let hetero: Vec<TuneScenario> =
            corpus().into_iter().filter(|s| s.name == "pinch_hetero").collect();
        assert_eq!(hetero.len(), 1, "the triage scenario is in the corpus");
        let ha = evaluate_config_opts(&MoverConfig::default(), &off, &hetero, 1);
        let hb = evaluate_config_opts(&MoverConfig::default(), &FleetOpts::default(), &hetero, 1);
        eprintln!("[triage A/B hetero] off: H={:.4} | on: H={:.4}", ha.h, hb.h);
        assert!(
            quantize_w(hb.h) > quantize_w(ha.h),
            "triage must win where value heterogeneity is the whole scenario (H {:.4} vs {:.4})",
            hb.h,
            ha.h
        );
    }

    fn env_u16_list(key: &str, default: &[u16]) -> Vec<u16> {
        // The combat-eval param_sweep env-list idiom (its `env_u32_list`), local on purpose —
        // a 10-line parser is cheaper than a cross-crate coupling for it.
        std::env::var(key)
            .ok()
            .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
            .filter(|v: &Vec<u16>| !v.is_empty())
            .unwrap_or_else(|| default.to_vec())
    }
    fn env_u32_list(key: &str, default: &[u32]) -> Vec<u32> {
        std::env::var(key)
            .ok()
            .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
            .filter(|v: &Vec<u32>| !v.is_empty())
            .unwrap_or_else(|| default.to_vec())
    }
    fn env_f64_list(key: &str, default: &[f64]) -> Vec<f64> {
        std::env::var(key)
            .ok()
            .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
            .filter(|v: &Vec<f64>| !v.is_empty())
            .unwrap_or_else(|| default.to_vec())
    }

    /// The STAGED tournament (coordinate descent over the tuning layers, incumbent carried
    /// forward). `#[ignore]`: run on demand —
    /// `cargo test -p screeps-rover-eval tune_rover_parameters -- --ignored --nocapture`
    /// Env overrides: `TUNE_ESCALATION` (avoid_friendly ladder bases), `TUNE_REUSE`,
    /// `TUNE_OPS`, `TUNE_SHOVE`, `TUNE_FRIENDLY_DIST` (comma lists);
    /// `TUNE_FULL_CORPUS=1` swaps in [`corpus_full`] (all 13 real rooms + the border routes).
    #[test]
    #[ignore]
    fn tune_rover_parameters() {
        let corpus = if std::env::var("TUNE_FULL_CORPUS").ok().as_deref() == Some("1") {
            corpus_full()
        } else {
            corpus()
        };
        let escalations = env_u16_list("TUNE_ESCALATION", &[1, 2, 4, 8]);
        let reuses = env_u32_list("TUNE_REUSE", &[2, 5, 10, 20]);
        let opses = env_u32_list("TUNE_OPS", &[5_000, 20_000, 60_000]);
        let shoves = env_u32_list("TUNE_SHOVE", &[1, 3, 6]);
        let dists = env_u32_list("TUNE_FRIENDLY_DIST", &[0, 3, 5, 10]);

        let n_points =
            escalations.len() + reuses.len() + opses.len() + shoves.len() + dists.len();
        let (points, creep_ticks) = estimate_budget(n_points, &corpus);
        eprintln!(
            "[budget] {points} points × {} scenarios ≤ {creep_ticks} creep-ticks (worst case)",
            corpus.len()
        );

        // The incumbent is the PAIR (system config, per-request opts) — the escalation stage
        // sweeps the haul-lane ladder (`FleetOpts::haul_stuck_thresholds`, the end-state's live
        // knob; the corpus's all-haul requests override the system ladder, so sweeping
        // `MoverConfig::stuck_thresholds` here would compare identical runs and combat's system
        // default stays untouched by construction).
        let mut incumbent = (MoverConfig::default(), FleetOpts::default());
        let started = Instant::now();

        enum Axis {
            Ladder(u16),
            Reuse(u32),
            Ops(u32),
            Shove(u32),
            Dist(u32),
        }
        let stages: Vec<(&str, Vec<(String, Axis)>)> = vec![
            (
                "escalation",
                escalations.iter().map(|&af| (format!("ladder({af})"), Axis::Ladder(af))).collect(),
            ),
            (
                "commitment/budget",
                reuses
                    .iter()
                    .map(|&r| (format!("reuse({r})"), Axis::Reuse(r)))
                    .chain(opses.iter().map(|&o| (format!("ops({o})"), Axis::Ops(o))))
                    .collect(),
            ),
            (
                "resolver",
                shoves
                    .iter()
                    .map(|&d| (format!("shove({d})"), Axis::Shove(d)))
                    .chain(dists.iter().map(|&d| (format!("dist({d})"), Axis::Dist(d))))
                    .collect(),
            ),
        ];

        for (stage_name, axes) in stages {
            // Rebase this stage's points on the CURRENT incumbent (coordinate descent), then add
            // the incumbent itself so a stage can only improve or hold.
            let mut points: Vec<(String, MoverConfig, FleetOpts)> = axes
                .into_iter()
                .map(|(name, axis)| {
                    let (mut cfg, mut opts) = incumbent.clone();
                    match axis {
                        Axis::Ladder(af) => opts.haul_stuck_thresholds = Some(ladder(af)),
                        Axis::Reuse(r) => cfg.reuse_path_length = r,
                        Axis::Ops(o) => cfg.pathfinding_ops_budget = o,
                        Axis::Shove(d) => cfg.max_shove_depth = d,
                        Axis::Dist(d) => cfg.friendly_creep_distance = d,
                    }
                    (name, cfg, opts)
                })
                .collect();
            points.push(("incumbent".into(), incumbent.0.clone(), incumbent.1.clone()));

            let ranked = run_tournament(points, &corpus, 1);
            eprintln!("── stage: {stage_name} ──");
            for (name, _, _, s) in &ranked {
                eprintln!(
                    "  {name:<14} H={:.4} CI[{:.4},{:.4}] p05={:.3} done={:.3} parked={} coord={} dead={} ticks={} gates={}",
                    s.h, s.ci95.0, s.ci95.1, s.p05, s.completion, s.failed_into_parked,
                    s.failed_coordination, s.deadlocks, s.total_ticks, s.gates_held
                );
                // Per-family H (decision (11) primary): a point that buys pooled H from one family
                // by taxing another is visible here, not laundered through the pool.
                eprintln!("  {:<14} └ {}", "", family_report(s));
            }
            let (winner_name, winner_cfg, winner_opts, winner_score) = &ranked[0];
            assert!(winner_score.gates_held, "stage `{stage_name}` winner must hold the gates");
            eprintln!("  → incumbent := {winner_name}");
            incumbent = (winner_cfg.clone(), winner_opts.clone());
        }

        let final_score = evaluate_config_opts(&incumbent.0, &incumbent.1, &corpus, 1);
        eprintln!(
            "[final] {:?} + {:?}\n  H={:.4} CI[{:.4},{:.4}] done={:.3} parked={} ({:.1?} total)\n  per-family {}",
            incumbent.0, incumbent.1, final_score.h, final_score.ci95.0, final_score.ci95.1,
            final_score.completion, final_score.failed_into_parked, started.elapsed(),
            family_report(&final_score)
        );
        assert!(final_score.gates_held);
    }

    // ── §D5.4 decision-(3) policy-constant sensitivity (value.rs `PolicyParams`) ────────────────
    // The follow-up the decisions block promises: sweep T_RAMP / S_REF / ε_intel / V_SINK and
    // report which points CHANGE the quantized contention-bid ORDER of a fixture set spanning the
    // §D5.4 role table. Rank changes are what matter — w feeds an ordering (resolver triage /
    // H sample weights are scale-free per role), so a constant is only "sensitive" where it flips
    // who outbids whom.

    use crate::value::{
        movement_intent_weight_with, GoalAnnotation, PolicyParams, Role, SquadRef, ThreatView,
        WorkKind,
    };
    use screeps_sim_core::SimCreep;

    struct PolicyFixture {
        name: &'static str,
        creep: SimCreep,
        role: Role,
        ann: GoalAnnotation,
        ttr: u32,
        ttl: u32,
        threat: Option<ThreatView>,
    }

    /// The §D5.4 role-table fixture set (value.rs test constructions, one per rail): a loaded
    /// hauler, a binding and a non-binding squad member, a slack-rich binding member (the U-decay
    /// probe), a deadline-tight claimer (slack 0 — the S_REF probe), a scout (the ε probe), two
    /// workers bracketing the hauler/slack gap (the V_SINK probe), and a wounded escaper under
    /// lethal fire. All Δ_crit = true; ids are the deterministic tie-break.
    fn policy_fixtures() -> Vec<PolicyFixture> {
        use screeps::Part;
        let creep = |id: u32, parts: &[Part]| SimCreep {
            id,
            owner: 0,
            pos: pos_in("W1N1", 25, 25),
            body: SimBody::unboosted(parts),
            fatigue: 0,
            carry_used: 0,
        };
        // The value.rs squad fixture: V_O = 0.8·45000 = 36000, R_O = 36000/900 = 40 e/t, exact.
        let squad = |binding: bool| SquadRef {
            p_win: 0.8,
            value_e: 45000.0,
            est_ticks: 900,
            alpha_class: 0.4,
            required_class_power: 60.0,
            fielded_class_power: 60.0,
            binding,
        };
        // Wounded escaper: 9×ATTACK+MOVE (1000 max hits) at half health under net 334 — S vetoes
        // progress (334·3 ≥ 500), escape bids (3000·0.5)·334/500 = 1002 e/t. No policy constant
        // touches this rail: it must stay rank-0 at every sweep point.
        let mut escaper = creep(1, &{
            let mut b = vec![screeps::Part::Attack; 9];
            b.push(screeps::Part::Move);
            b
        });
        escaper.body.hits = 500;
        vec![
            PolicyFixture {
                name: "wounded_escaper",
                creep: escaper,
                role: Role::Melee,
                ann: GoalAnnotation {
                    t_min: 50,
                    value_stock_e: 3000.0,
                    squad: Some(squad(true)),
                    ..Default::default()
                },
                ttr: 100,
                ttl: 1000,
                threat: Some(ThreatView { net_incoming_per_tick: 334 }),
            },
            // Deadline-tight claimer: ttr 500 + margin 100 == the 600 claim lifetime, slack 0 ⇒
            // w = 5000/S_REF (50 at default) — "deadline-tight claimers explode in priority".
            PolicyFixture {
                name: "claim_tight",
                creep: creep(2, &[screeps::Part::Claim, screeps::Part::Move]),
                role: Role::Claim,
                ann: GoalAnnotation { value_stock_e: 5000.0, ..Default::default() },
                ttr: 500,
                ttl: 1500,
                threat: None,
            },
            // Binding member, window binding (900 ≤ 950): bids the full R_O = 40.
            PolicyFixture {
                name: "squad_binding",
                creep: creep(3, &[screeps::Part::Attack, screeps::Part::Move]),
                role: Role::Melee,
                ann: GoalAnnotation { t_min: 50, squad: Some(squad(true)), ..Default::default() },
                ttr: 100,
                ttl: 1000,
                threat: None,
            },
            // Loaded hauler: ρ = 100/50 = 2 e/t, all gates open.
            PolicyFixture {
                name: "hauler_loaded",
                creep: creep(4, &[screeps::Part::Carry, screeps::Part::Carry, screeps::Part::Move]),
                role: Role::Haul { q: 100 },
                ann: GoalAnnotation { rate_e_t: 2.0, t_min: 1, ..Default::default() },
                ttr: 200,
                ttl: 1500,
                threat: None,
            },
            // Builder supply-capped at 1.6 e/t — sits just ABOVE the slack-rich member at default
            // V_SINK, so v_sink < 1 flips them (the sink-value probe).
            PolicyFixture {
                name: "worker_build",
                creep: creep(5, &[screeps::Part::Work, screeps::Part::Carry, screeps::Part::Move]),
                role: Role::Work { kind: WorkKind::Build, supply_rate_e_t: 1.6 },
                ann: GoalAnnotation::default(),
                ttr: 10,
                ttl: 1500,
                threat: None,
            },
            // Slack-rich binding member (window 1500 vs horizon 950, excess 550): U decays
            // 40 → 40/(1 + 550/T_RAMP) = 1.404 at default — the ADR's "slack-rich squad bids
            // ~upkeep and loses tiles to a loaded hauler". THE T_RAMP probe.
            PolicyFixture {
                name: "squad_slack",
                creep: creep(6, &[screeps::Part::Attack, screeps::Part::Move]),
                role: Role::Melee,
                ann: GoalAnnotation { t_min: 50, squad: Some(squad(true)), ..Default::default() },
                ttr: 0,
                ttl: 1500,
                threat: None,
            },
            // Upgrader: WORK·1 = 1.0 e/t — brackets the slack member from BELOW at default.
            PolicyFixture {
                name: "worker_upgrade",
                creep: creep(7, &[screeps::Part::Work, screeps::Part::Carry, screeps::Part::Move]),
                role: Role::Work { kind: WorkKind::Upgrade, supply_rate_e_t: 7.0 },
                ann: GoalAnnotation::default(),
                ttr: 10,
                ttl: 1500,
                threat: None,
            },
            // Non-binding member: the decision-(2) upkeep floor, HEAL+MOVE = 300/1500 = 0.2 e/t.
            PolicyFixture {
                name: "squad_non_binding",
                creep: creep(8, &[screeps::Part::Heal, screeps::Part::Move]),
                role: Role::Heal,
                ann: GoalAnnotation {
                    t_min: 50,
                    squad: Some(SquadRef {
                        alpha_class: 0.6,
                        required_class_power: 24.0,
                        fielded_class_power: 24.0,
                        binding: false,
                        ..squad(false)
                    }),
                    ..Default::default()
                },
                ttr: 100,
                ttl: 1000,
                threat: None,
            },
            // Scout: max(ε_intel, upkeep); a 1×MOVE scout's upkeep IS the default ε (50/1500).
            PolicyFixture {
                name: "scout",
                creep: creep(9, &[screeps::Part::Move]),
                role: Role::Scout,
                ann: GoalAnnotation::default(),
                ttr: 10,
                ttl: 1500,
                threat: None,
            },
        ]
    }

    /// The fixture names in descending contention-bid order under `params` — compared ONLY through
    /// [`crate::value::ordering_key`] (quantized milli-e/t + stable id), the determinism fence.
    fn policy_ranked_names(params: &PolicyParams, fixtures: &[PolicyFixture]) -> Vec<&'static str> {
        let mut keyed: Vec<_> = fixtures
            .iter()
            .map(|f| {
                let w = movement_intent_weight_with(
                    params, &f.creep, &f.role, &f.ann, f.ttr, f.ttl, true, f.threat.as_ref(),
                );
                (crate::value::ordering_key(w, f.creep.id), f.name)
            })
            .collect();
        keyed.sort_by_key(|&(key, _)| std::cmp::Reverse(key));
        keyed.into_iter().map(|(_, name)| name).collect()
    }

    /// The documented §D5.4 decision-block ordering at the DEFAULT point (each relation cites its
    /// decision): escape outbids everything (8); the deadline-tight claimer explodes (role table);
    /// the binding member outbids a loaded hauler (1); the slack-rich member decays below the
    /// hauler (2)+(3); the non-binding member bids the upkeep floor (1)+(2); the scout bids ε (3).
    const DEFAULT_POLICY_ORDER: [&str; 9] = [
        "wounded_escaper",   // 1002 e/t   escape drawdown
        "claim_tight",       // 50         V/S_REF
        "squad_binding",     // 40         R_O
        "hauler_loaded",     // 2          Q/T*
        "worker_build",      // 1.6        min(WORK·5, supply)·V_SINK
        "squad_slack",       // 1.404      R_O / (1 + 550/T_RAMP)
        "worker_upgrade",    // 1.0        min(WORK·1, supply)·V_SINK
        "squad_non_binding", // 0.2        upkeep floor
        "scout",             // 0.033      ε_intel
    ];

    /// The default point must reproduce the decision-block orderings — the checked-in pin the
    /// sweep measures rank changes against (also guards `PolicyParams`' default wiring end-to-end).
    #[test]
    fn default_policy_point_reproduces_the_decision_block_ordering() {
        let order = policy_ranked_names(&PolicyParams::default(), &policy_fixtures());
        assert_eq!(order, DEFAULT_POLICY_ORDER, "the §D5.4 decision-block ordering at defaults");
    }

    /// The decision-(3) sensitivity sweep (`#[ignore]`: run on demand —
    /// `cargo test -p screeps-rover-eval sweep_policy_params -- --ignored --nocapture`).
    /// Env overrides (comma lists): `SWEEP_T_RAMP`, `SWEEP_S_REF`, `SWEEP_V_SINK` (absolute
    /// values), `SWEEP_EPSILON_X` (multipliers of the default ε). One param varies per point
    /// (coordinate probes off the default, the param_sweep shape); the deliverable is which points
    /// are RANK-AFFECTING over the role-table fixture set. Recorded at the shipped defaults
    /// (2026-07-01): t_ramp 10 and 40 both flip ranks (the slack-decay knob is the sharp one),
    /// s_ref 200 drops a slack-0 claimer below a binding member, v_sink 0.5 drops workers below
    /// the slack member, ε_intel ×0.5/×2 flips nothing (the upkeep floor dominates until ε clears
    /// the next-lowest bid — it only orders scouts among sub-upkeep bids).
    #[test]
    #[ignore]
    fn sweep_policy_params() {
        let fixtures = policy_fixtures();
        let baseline = policy_ranked_names(&PolicyParams::default(), &fixtures);
        assert_eq!(baseline, DEFAULT_POLICY_ORDER, "sweep baseline == the decision-block ordering");
        eprintln!("[policy sweep] baseline {baseline:?}");

        let t_ramps = env_f64_list("SWEEP_T_RAMP", &[10.0, 20.0, 40.0]);
        let s_refs = env_f64_list("SWEEP_S_REF", &[50.0, 100.0, 200.0]);
        let eps_x = env_f64_list("SWEEP_EPSILON_X", &[0.5, 1.0, 2.0]);
        let v_sinks = env_f64_list("SWEEP_V_SINK", &[0.5, 1.0]);

        let d = PolicyParams::default();
        let points: Vec<(String, PolicyParams)> = t_ramps
            .iter()
            .map(|&v| (format!("t_ramp={v}"), PolicyParams { t_ramp: v, ..d }))
            .chain(s_refs.iter().map(|&v| (format!("s_ref={v}"), PolicyParams { s_ref: v, ..d })))
            .chain(eps_x.iter().map(|&x| {
                (format!("epsilon_intel=x{x}"), PolicyParams { epsilon_intel: d.epsilon_intel * x, ..d })
            }))
            .chain(v_sinks.iter().map(|&v| (format!("v_sink={v}"), PolicyParams { v_sink: v, ..d })))
            .collect();

        let mut rank_affecting: Vec<String> = Vec::new();
        for (name, params) in &points {
            let order = policy_ranked_names(params, &fixtures);
            if order == baseline {
                eprintln!("[policy sweep] {name:<20} order unchanged");
            } else {
                eprintln!("[policy sweep] {name:<20} RANK-AFFECTING → {order:?}");
                rank_affecting.push(name.clone());
            }
        }
        eprintln!("[policy sweep] rank-affecting points: {rank_affecting:?}");
    }
}
