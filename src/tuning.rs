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

use crate::base_traffic::{base_scenario, captured_layouts, energy_traffic_fleet};
use crate::haul::{run_haul_fleet_with, HaulAssignment};
use crate::stats::Summary;
use crate::value::quantize_w;
use screeps::{Position, RoomCoordinate, RoomName};
use screeps_sim_core::{MoverConfig, SimBody, SimTerrain};
use screeps_rover::StuckThresholds;

/// One corpus scenario: a named world + fleet + tick cap.
pub struct TuneScenario {
    pub name: String,
    pub terrain: SimTerrain,
    pub fleet: Vec<HaulAssignment>,
    pub tick_cap: u32,
}

/// The score of ONE [`MoverConfig`] point over the corpus (the `ParamScore` shape).
#[derive(Clone, Debug)]
pub struct TuneScore {
    /// The objective: pooled value-weighted efficiency over every trip in the corpus.
    pub h: f64,
    pub ci95: (f64, f64),
    pub p05: f64,
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

/// The default tuning corpus: the four synthetic mechanism scenarios (free-flow floor, one-gap
/// pinch, crossing routes, road-through-swamp) + three REAL layouts — one healthy room as the
/// regression floor and the two KNOWN-STARVED rooms as the healing targets.
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
        terrain: pinch,
        fleet: (0..4).map(|_| hauler(pos_in("W1N1", 10, 25), pos_in("W1N1", 20, 25))).collect(),
        tick_cap: 2_000,
    });

    // Crossing routes: two perpendicular 2-hauler routes through a shared center (resolver stress).
    scenarios.push(TuneScenario {
        name: "cross_traffic".into(),
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
        terrain: swamp_road,
        fleet: (0..2).map(|_| hauler(pos_in("W1N1", 10, 25), pos_in("W1N1", 20, 25))).collect(),
        tick_cap: 1_000,
    });

    // Real foreman-planned layouts: the first healthy room (regression floor) + both starved rooms
    // (the park-sealed-corridor repath-flap livelock — the healing targets).
    let picks = ["E11N13", "E13S29", "E11N17"];
    for layout in captured_layouts() {
        if !picks.contains(&layout.room.as_str()) {
            continue;
        }
        let terrain = base_scenario(&layout);
        let fleet = energy_traffic_fleet(&layout, &terrain);
        scenarios.push(TuneScenario {
            name: format!("base:{}", layout.room),
            terrain,
            fleet,
            tick_cap: 2_500,
        });
    }
    scenarios
}

/// Score one config over the corpus: pool every trip's `(η, W)` sample into ONE [`Summary`]
/// (value-weighted, exactly the §D5.4 aggregate), sum the audits/gates. PURE + deterministic —
/// same config + corpus + seed ⇒ byte-identical score.
pub fn evaluate_config(config: &MoverConfig, corpus: &[TuneScenario], seed: u32) -> TuneScore {
    let mut samples: Vec<(f64, f64)> = Vec::new();
    let mut completed = 0u32;
    let mut expected = 0u32;
    let mut deadlocks = 0u32;
    let mut coordination = 0u32;
    let mut parked = 0u32;
    let mut intents = 0u32;
    let mut ticks = 0u32;

    for s in corpus {
        let out = run_haul_fleet_with(&s.terrain, &s.fleet, s.tick_cap, seed, config)
            .unwrap_or_else(|| panic!("corpus scenario `{}` must be oracle-solvable", s.name));
        samples.extend_from_slice(&out.samples);
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
        completion: if expected > 0 { completed as f64 / expected as f64 } else { 1.0 },
        deadlocks,
        failed_coordination: coordination,
        failed_into_parked: parked,
        intents_issued: intents,
        total_ticks: ticks,
        gates_held: coordination == 0 && deadlocks == 0,
    }
}

/// Rank named config points over the corpus, best first ([`TuneScore::ranked_key`] descending).
pub fn run_tournament(
    points: Vec<(String, MoverConfig)>,
    corpus: &[TuneScenario],
    seed: u32,
) -> Vec<(String, MoverConfig, TuneScore)> {
    let mut ranked: Vec<(String, MoverConfig, TuneScore)> = points
        .into_iter()
        .map(|(name, config)| {
            let score = evaluate_config(&config, corpus, seed);
            (name, config, score)
        })
        .collect();
    ranked.sort_by_key(|(name, _, score)| (std::cmp::Reverse(score.ranked_key()), name.clone()));
    ranked
}

/// A [`StuckThresholds`] ladder built from its tier-1 base with the default tier SPACING ratios —
/// one scalar ("escalation speed") instead of six independent axes for the coarse stage.
pub fn ladder(avoid_friendly: u16) -> StuckThresholds {
    let d = StuckThresholds::default();
    let scale =
        |v: u16| ((v as u32 * avoid_friendly as u32).div_ceil(d.avoid_friendly_creeps as u32)) as u16;
    StuckThresholds {
        avoid_friendly_creeps: avoid_friendly.max(1),
        avoid_all_friendly_creeps: scale(d.avoid_all_friendly_creeps).max(avoid_friendly + 1),
        increase_ops: scale(d.increase_ops),
        enable_shoving: scale(d.enable_shoving),
        report_failure: scale(d.report_failure),
        no_progress_repath: scale(d.no_progress_repath),
    }
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

    /// Same config + corpus + seed ⇒ byte-identical score (the param_sweep determinism pin).
    #[test]
    fn evaluation_is_deterministic() {
        let a = evaluate_config(&MoverConfig::default(), &corpus(), 7);
        let b = evaluate_config(&MoverConfig::default(), &corpus(), 7);
        assert_eq!(a.ranked_key(), b.ranked_key());
        assert_eq!(a.h.to_bits(), b.h.to_bits(), "bit-identical H");
        assert_eq!(a.intents_issued, b.intents_issued);
    }

    /// The escalation knob must actually REACH rover end-to-end: an absurdly slow ladder on the
    /// pinch corpus (escalation effectively off) reproduces the parked-blocker livelock — wasted
    /// intents blow up vs the default. Proves config wiring, not just plumbing types.
    #[test]
    fn escalation_speed_knob_is_live() {
        let corpus: Vec<TuneScenario> =
            corpus().into_iter().filter(|s| s.name == "pinch").collect();
        let fast = evaluate_config(&MoverConfig::default(), &corpus, 1);
        let crippled = MoverConfig {
            stuck_thresholds: ladder(200),
            ..Default::default()
        };
        let slow = evaluate_config(&crippled, &corpus, 1);
        assert!(
            slow.failed_into_parked > 10 * fast.failed_into_parked.max(1),
            "a 100x-slower escalation must burn far more intents (default {} vs crippled {})",
            fast.failed_into_parked,
            slow.failed_into_parked
        );
        assert!(slow.completion < 1.0, "escalation-off starves the pinch");
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

    /// The STAGED tournament (coordinate descent over the tuning layers, incumbent carried
    /// forward). `#[ignore]`: run on demand —
    /// `cargo test -p screeps-rover-eval tune_rover_parameters -- --ignored --nocapture`
    /// Env overrides: `TUNE_ESCALATION` (avoid_friendly ladder bases), `TUNE_REUSE`,
    /// `TUNE_OPS`, `TUNE_SHOVE`, `TUNE_FRIENDLY_DIST` (comma lists).
    #[test]
    #[ignore]
    fn tune_rover_parameters() {
        let corpus = corpus();
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

        let mut incumbent = MoverConfig::default();
        let started = Instant::now();

        let stages: Vec<(&str, Vec<(String, MoverConfig)>)> = vec![
            (
                "escalation",
                escalations
                    .iter()
                    .map(|&af| {
                        (format!("ladder({af})"), MoverConfig {
                            stuck_thresholds: ladder(af),
                            ..incumbent.clone()
                        })
                    })
                    .collect(),
            ),
            (
                "commitment/budget",
                reuses
                    .iter()
                    .map(|&r| {
                        (format!("reuse({r})"), MoverConfig { reuse_path_length: r, ..incumbent.clone() })
                    })
                    .chain(opses.iter().map(|&o| {
                        (format!("ops({o})"), MoverConfig {
                            pathfinding_ops_budget: o,
                            ..incumbent.clone()
                        })
                    }))
                    .collect(),
            ),
            (
                "resolver",
                shoves
                    .iter()
                    .map(|&d| {
                        (format!("shove({d})"), MoverConfig { max_shove_depth: d, ..incumbent.clone() })
                    })
                    .chain(dists.iter().map(|&d| {
                        (format!("dist({d})"), MoverConfig {
                            friendly_creep_distance: d,
                            ..incumbent.clone()
                        })
                    }))
                    .collect(),
            ),
        ];

        for (stage_name, mut points) in stages {
            // Rebase this stage's points on the CURRENT incumbent (coordinate descent), then add
            // the incumbent itself so a stage can only improve or hold.
            for (_, cfg) in points.iter_mut() {
                let mut merged = incumbent.clone();
                match stage_name {
                    "escalation" => merged.stuck_thresholds = cfg.stuck_thresholds.clone(),
                    "commitment/budget" => {
                        merged.reuse_path_length = cfg.reuse_path_length;
                        merged.pathfinding_ops_budget = cfg.pathfinding_ops_budget;
                    }
                    _ => {
                        merged.max_shove_depth = cfg.max_shove_depth;
                        merged.friendly_creep_distance = cfg.friendly_creep_distance;
                    }
                }
                *cfg = merged;
            }
            points.push(("incumbent".into(), incumbent.clone()));

            let ranked = run_tournament(points, &corpus, 1);
            eprintln!("── stage: {stage_name} ──");
            for (name, _, s) in &ranked {
                eprintln!(
                    "  {name:<14} H={:.4} CI[{:.4},{:.4}] p05={:.3} done={:.3} parked={} coord={} dead={} ticks={} gates={}",
                    s.h, s.ci95.0, s.ci95.1, s.p05, s.completion, s.failed_into_parked,
                    s.failed_coordination, s.deadlocks, s.total_ticks, s.gates_held
                );
            }
            let (winner_name, winner_cfg, winner_score) = &ranked[0];
            assert!(winner_score.gates_held, "stage `{stage_name}` winner must hold the gates");
            eprintln!("  → incumbent := {winner_name}");
            incumbent = winner_cfg.clone();
        }

        let final_score = evaluate_config(&incumbent, &corpus, 1);
        eprintln!(
            "[final] {:?}\n  H={:.4} CI[{:.4},{:.4}] done={:.3} parked={} ({:.1?} total)",
            incumbent, final_score.h, final_score.ci95.0, final_score.ci95.1,
            final_score.completion, final_score.failed_into_parked, started.elapsed()
        );
        assert!(final_score.gates_held);
    }
}
