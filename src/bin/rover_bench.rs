//! ADR 0033 §M5-rest — the **corpus benchmark CLI**: run the tuning corpus through the shipped
//! harness stack and print the per-scenario lines, the per-family H report (§D5.4 decision (11)
//! primary view), the pooled H + CI + percentiles, and the failed-move audit totals. The manual
//! analogue of the checked-in `tuning.rs` gates — same corpora, same scorer
//! ([`screeps_rover_eval::tuning::corpus_outcomes`]/[`score_outcomes`]), so a number printed here
//! is byte-comparable to the ratchets' baselines.
//!
//!   cargo run --release -p screeps-rover-eval --bin rover_bench [-- fast|full]
//!
//! `fast` (default) = the 8-scenario checked-in corpus; `full` = all 13 real rooms + the 3 border
//! routes (20 scenarios). Overrides via env vars (the repo's env-driven bench idiom — single
//! values, not the sweep's comma lists; unset = the shipped default):
//!
//!   BENCH_SEED=1              bootstrap-CI seed (u32; determinism: same seed ⇒ byte-identical)
//!   BENCH_REUSE=20            MoverConfig::reuse_path_length
//!   BENCH_OPS=20000           MoverConfig::pathfinding_ops_budget
//!   BENCH_SHOVE=3             MoverConfig::max_shove_depth
//!   BENCH_FRIENDLY_DIST=5     MoverConfig::friendly_creep_distance
//!   BENCH_REGISTER_IDLE=1     MoverConfig::register_idle_creeps (0/1)
//!   BENCH_LADDER=8            per-request haul stuck ladder base (FleetOpts; 0 = none/system)
//!   BENCH_VALUE_PRIORITY=1    FleetOpts::value_priority (0/1 — §D5.4 decision (9) triage)

use screeps_rover_eval::haul::{ladder, FleetOpts};
use screeps_rover_eval::tuning::{
    corpus, corpus_full, corpus_outcomes, family_report, score_outcomes,
};
use screeps_sim_core::MoverConfig;
use std::time::Instant;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(default)
}
fn env_flag(key: &str, default: bool) -> bool {
    std::env::var(key).ok().map(|s| s.trim() != "0").unwrap_or(default)
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "fast".into());
    let scenarios = match which.as_str() {
        "fast" => corpus(),
        "full" => corpus_full(),
        other => {
            eprintln!("rover_bench: unknown corpus `{other}` (expected `fast` or `full`)");
            std::process::exit(2);
        }
    };

    let seed = env_u32("BENCH_SEED", 1);
    let defaults = MoverConfig::default();
    let config = MoverConfig {
        reuse_path_length: env_u32("BENCH_REUSE", defaults.reuse_path_length),
        pathfinding_ops_budget: env_u32("BENCH_OPS", defaults.pathfinding_ops_budget),
        max_shove_depth: env_u32("BENCH_SHOVE", defaults.max_shove_depth),
        friendly_creep_distance: env_u32("BENCH_FRIENDLY_DIST", defaults.friendly_creep_distance),
        register_idle_creeps: env_flag("BENCH_REGISTER_IDLE", defaults.register_idle_creeps),
        ..defaults
    };
    let ladder_base = env_u32("BENCH_LADDER", 8);
    let opts = FleetOpts {
        haul_stuck_thresholds: (ladder_base > 0).then(|| ladder(ladder_base as u16)),
        value_priority: env_flag("BENCH_VALUE_PRIORITY", true),
    };

    println!(
        "rover_bench: corpus={which} ({} scenarios) seed={seed} reuse={} ops={} shove={} dist={} register_idle={} haul_ladder={} value_priority={}",
        scenarios.len(),
        config.reuse_path_length,
        config.pathfinding_ops_budget,
        config.max_shove_depth,
        config.friendly_creep_distance,
        config.register_idle_creeps,
        if ladder_base > 0 { format!("ladder({ladder_base})") } else { "system".into() },
        opts.value_priority,
    );

    let started = Instant::now();
    let outcomes = corpus_outcomes(&config, &opts, &scenarios, seed);

    // Per-scenario lines (the base_traffic baseline-report shape).
    let (mut issued, mut failed, mut fatigued, mut wall, mut parked, mut coord) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    for (s, out) in scenarios.iter().zip(&outcomes) {
        let sum = &out.summary;
        let a = &out.audit;
        println!(
            "  [{:<9}] {:<22} H={:.4} CI95=[{:.4},{:.4}] p05={:.3} | {}/{} trips in {:>4} ticks | intents {:>4}, failed {} (fat {}, wall {}, park {}, coord {}){}",
            s.family, s.name, sum.weighted_mean, sum.ci95.0, sum.ci95.1, sum.p05,
            out.completed_trips, out.expected_trips, out.ticks,
            a.intents_issued, a.failed_moves, a.failed_fatigued, a.failed_wall,
            a.failed_into_parked, a.failed_coordination,
            if out.deadlocked { "  DEADLOCK" } else { "" },
        );
        issued += a.intents_issued as u64;
        failed += a.failed_moves as u64;
        fatigued += a.failed_fatigued as u64;
        wall += a.failed_wall as u64;
        parked += a.failed_into_parked as u64;
        coord += a.failed_coordination as u64;
    }

    let score = score_outcomes(&scenarios, &outcomes, seed);
    println!("per-family {}", family_report(&score));
    println!(
        "pooled H={:.4} CI95=[{:.4},{:.4}] p05={:.3} completion={:.3} deadlocks={} gates_held={}",
        score.h, score.ci95.0, score.ci95.1, score.p05, score.completion, score.deadlocks,
        score.gates_held,
    );
    println!(
        "audit totals: intents {issued}, failed {failed} (fatigued {fatigued}, wall {wall}, parked {parked}, coordination {coord}) | {} sim ticks in {:.1?}",
        score.total_ticks,
        started.elapsed(),
    );

    // Exit non-zero when the hard gates fail so the CLI is scriptable as a coarse check.
    if !score.gates_held {
        std::process::exit(1);
    }
}
