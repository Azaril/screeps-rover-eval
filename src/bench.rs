//! CPU / algorithmic bench for the rover mover (ADR 0033 §D5.3 / M5-rest): **op-counted primary,
//! wall-clock secondary**. The primary signal is rover's own [`MovementTickStats`] —
//! `ops_consumed` (pathfinding ops, 1 op ≈ 0.001 CPU live) + `repaths` per `process()` — surfaced
//! through the kernel driver's stats variant
//! ([`screeps_sim_core::resolve_moves_via_system_stats`], slice 7; this module's historical local
//! mirror driver deleted itself when that seam landed), the deterministic currency the scaling
//! curves and gates are pinned in. Wall clock is measured too but only ever gated with LOOSE
//! death-spiral bounds
//! (never tight thresholds, no criterion — the repo's hand-rolled `Instant`-loop convention,
//! `screeps-combat-eval/src/bench.rs`): **native host wall-clock is a RELATIVE proxy for Screeps
//! CPU** (wasm differs; there is no game CPU meter offline). What the loose bound catches is the
//! real risk class — the "CPU pathfinding death-spiral" algorithmic blowup — not microseconds.
//!
//! The scaling curves (the §D5.3 deliverable) live as `#[ignore]` tests that print the curve and
//! fit a rough log-log slope: `ops/tick` vs N creeps on parallel lanes (gate: ~linear — a
//! super-linear slope is the congestion-blowup class) and on a shared one-tile pinch (reported +
//! loosely bounded; serialization makes some super-linearity structural), and `ops/search` vs
//! route length.
//!
//! ## G-13 canary alignment (offline ↔ live, ADR 0033 §D8 L6 / M5-rest item 5)
//!
//! The live seg-57 canary emits two movement families the offline bench must stay definitionally
//! aligned with (READ-ONLY survey of the bot crate, 2026-07-01):
//!
//! - **Ops / repath telemetry — ALIGNED BY CONSTRUCTION.** Live records
//!   `MovementSystem::tick_stats()` verbatim (`screeps-ibex/src/pathing/movementsystem.rs:339` →
//!   `metrics.rs::record_movement_stats` → the seg-57 `pathing` block's
//!   `ops_used`/`ops_pool`/`repath_count`). This module reads the SAME struct from the SAME
//!   method, so the offline ops curves and the live ADR-0004 ops-saturation stream are the same
//!   quantity by definition — they cross-vouch with zero translation.
//! - **The G-13 wasted-move counter — a DEFINITIONAL GAP, flagged.** Live's `move_failures`
//!   (`pathing/movementsystem.rs:343-352` → `record_movement_failures`) counts rover
//!   **self-reported give-ups**: `MovementResult::Failed(_)` plus `Stuck { ticks ≥
//!   STUCK_REPORT_THRESHOLD = 10 }`, per tick. The offline sentinel
//!   ([`crate::crowd::IntentAudit`]) counts something stronger: the **issued-vs-executed
//!   reconciliation** — every `Direction` rover issued that the engine did not execute ("intent
//!   spent, no action"), partitioned `failed_fatigued` / `failed_wall` / `failed_into_parked` /
//!   `failed_coordination` by walking the blocking chain. Gap (1) below LANDED live in slice 7
//!   (seg-57 `pathing.wasted_moves` = the issued-vs-moved reconciliation via the CreepHandle
//!   wrapper seam; `move_failures` KEPT as the self-reported give-up level for comparison).
//!   The remaining gaps (2)/(3) stand as recorded:
//!   (1) an engine-rejected intent is INVISIBLE live until the creep accrues 10 consecutive
//!   immobile ticks — one-off rejections and the avoidable `failed_fatigued` burn never surface;
//!   the live analogue of the offline sentinel is a cheap issued-vs-moved check (position
//!   unchanged ∧ `fatigue == 0` after an issued move intent, one compare per moving creep);
//!   (2) `Stuck ≥ 10` is a per-tick LEVEL, so one stuck episode is re-counted every tick it
//!   persists, while the offline classes count discrete failures — rates are not comparable
//!   without an episode normalization; (3) live has no blocking-chain attribution, so the offline
//!   partition stays the diagnostic layer and the live counter the alarm layer — same event
//!   family, different resolution, which is fine ONCE (1) makes the event family itself match.

use crate::cost::WorldCostSource;
use crate::crowd::CrowdCreep;
use screeps::{Part, Position};
use screeps_sim_core::{
    resolve_movement, resolve_moves_via_system_stats, MoveIntents, MovementState, MoverConfig,
    SimBody, SimCreep, SimMoveCache, SimMoveRequest, SimTerrain,
};
use std::time::{Duration, Instant};

/// One bench run's accounting: op-counted primary, wall-clock secondary.
#[derive(Clone, Debug)]
pub struct OpsRun {
    pub creeps: usize,
    pub ticks: u32,
    pub all_arrived: bool,
    /// Move intents rover issued over the run.
    pub intents_issued: u64,
    /// Σ `MovementTickStats::ops_consumed` — the op-counted PRIMARY (deterministic).
    pub ops_consumed: u64,
    /// Σ `MovementTickStats::repaths` — really a SEARCH counter: rover increments it on every
    /// successful `generate_path`, first-time paths and segment re-searches included (the
    /// rover-side field doc now says exactly this — the doc bug this module flagged was fixed
    /// upstream, slice 7). So `ops_consumed / repaths` ≈ ops per search.
    pub repaths: u64,
    /// The configured per-tick ops budget the run held (`MoverConfig::pathfinding_ops_budget`).
    pub ops_budget_cap: u32,
    /// max over ticks of `ops_consumed` — the §D5.1(f) L4 invariant probe (≤ the per-tick cap).
    pub max_tick_ops: u32,
    /// Wall clock over the whole run — the SECONDARY, loose-bounds-only signal (module note).
    pub elapsed: Duration,
}

impl OpsRun {
    /// Mean pathfinding ops per simulated tick — the scaling curves' y-axis.
    pub fn ops_per_tick(&self) -> f64 {
        self.ops_consumed as f64 / (self.ticks.max(1)) as f64
    }
    /// Mean wall-clock microseconds per creep-tick (the combat-eval `per_block_tick_us` shape).
    pub fn us_per_creep_tick(&self) -> f64 {
        self.elapsed.as_secs_f64() * 1e6 / (self.creeps as f64 * self.ticks.max(1) as f64)
    }
}

/// Drive `creeps` to their goals through the real mover (crowd-shaped loop: arrived creeps park
/// and are registered per `config`), accounting ops + repaths + wall clock. Stops when all arrive
/// or `tick_cap` elapses — no deadlock detector here on purpose (a stall shows up as
/// `all_arrived == false` + the burned ticks; the QUALITY suites own deadlock gating).
pub fn run_ops_bench(
    terrain: &SimTerrain,
    creeps: &[CrowdCreep],
    tick_cap: u32,
    config: &MoverConfig,
) -> OpsRun {
    let n = creeps.len();
    let mut world = MovementState {
        terrain: terrain.clone(),
        creeps: creeps
            .iter()
            .enumerate()
            .map(|(i, c)| SimCreep {
                id: i as u32 + 1,
                owner: 0,
                pos: c.from,
                body: c.body.clone(),
                fatigue: 0,
                carry_used: c.carry_used,
            })
            .collect(),
        ..Default::default()
    };
    let goals: Vec<(Position, u32)> = creeps.iter().map(|c| (c.goal, c.range)).collect();
    let within = |p: Position, i: usize| p.get_range_to(goals[i].0) <= goals[i].1;

    let mut cache = SimMoveCache::new();
    let mut run = OpsRun {
        creeps: n,
        ticks: 0,
        all_arrived: false,
        intents_issued: 0,
        ops_consumed: 0,
        repaths: 0,
        ops_budget_cap: config.pathfinding_ops_budget,
        max_tick_ops: 0,
        elapsed: Duration::ZERO,
    };

    let started = Instant::now();
    for t in 0..tick_cap {
        if (0..n).all(|i| within(world.creeps[i].pos, i)) {
            break;
        }
        // Kernel-driver requests: `move_to` defaults (Normal priority, shove+swap consent) are
        // exactly what the deleted mirror's bare `data.move_to(id, target).range(range)` carried
        // (rover's `MovementRequest` defaults), so the rewire is behavior-identical — the
        // ops-determinism pin below and the re-run scaling curves vouch for it in the numbers.
        let reqs: Vec<SimMoveRequest> = (0..n)
            .filter(|&i| !within(world.creeps[i].pos, i))
            .map(|i| SimMoveRequest::move_to(world.creeps[i].id, goals[i].0, goals[i].1))
            .collect();
        let (dirs, stats) = resolve_moves_via_system_stats(
            &world,
            &reqs,
            &mut cache,
            WorldCostSource::new(terrain, &world),
            config,
        );
        run.intents_issued += dirs.len() as u64;
        run.ops_consumed += stats.ops_consumed as u64;
        run.repaths += stats.repaths as u64;
        run.max_tick_ops = run.max_tick_ops.max(stats.ops_consumed);

        let mut intents = MoveIntents::new();
        for (&id, &d) in &dirs {
            intents.set_move(id, d);
        }
        resolve_movement(&mut world, &intents);
        run.ticks = t + 1;
    }
    run.elapsed = started.elapsed();
    run.all_arrived = (0..n).all(|i| within(world.creeps[i].pos, i));
    run
}

/// A balanced 1:1 hauler body (full speed empty on plain/road) — the bench's standard creep.
fn balanced_hauler() -> SimBody {
    SimBody::unboosted(&[Part::Carry, Part::Carry, Part::Move, Part::Move])
}

fn pos(x: u8, y: u8) -> Position {
    crate::traverse::pos_in("W1N1".parse().unwrap(), x, y)
}

/// N creeps on PARALLEL lanes (free flow — the linear-scaling reference): lanes stacked down the
/// room (y = 1 + i mod 48); creep 48+ starts 2 tiles behind on an already-used lane with a goal
/// 2 tiles short of its leader's (a same-speed tandem — the leader vacates each tick and the
/// goals never collide, so still contention-free by construction).
pub fn parallel_lanes(n: usize) -> (SimTerrain, Vec<CrowdCreep>) {
    let creeps = (0..n)
        .map(|i| {
            let y = 1 + (i % 48) as u8;
            let offset = 2 * (i / 48) as u8;
            CrowdCreep::new(balanced_hauler(), pos(10 - offset, y), pos(40 - offset, y), 0)
        })
        .collect();
    (SimTerrain::default(), creeps)
}

/// N creeps forced through ONE wall gap at (15, 25) (the congestion / serialization case): starts
/// in a block left of the wall, per-creep distinct goals mirrored right of it, every route through
/// the gap.
pub fn shared_pinch(n: usize) -> (SimTerrain, Vec<CrowdCreep>) {
    let mut terrain = SimTerrain::default();
    for y in 0..=49u8 {
        if y != 25 {
            terrain.walls.insert((15, y));
        }
    }
    let creeps = (0..n)
        .map(|i| {
            let (col, row) = ((i % 8) as u8, (i / 8) as u8);
            CrowdCreep::new(
                balanced_hauler(),
                pos(3 + col, 18 + row),
                pos(46 - col, 18 + row),
                0,
            )
        })
        .collect();
    (terrain, creeps)
}

/// Least-squares slope of `ln(y)` on `ln(x)` — the rough scaling exponent the curves report
/// (points with `y <= 0` are skipped; a config that consumes zero ops contributes nothing to
/// fit). Deterministic: pure arithmetic over deterministic inputs.
pub fn log_log_slope(points: &[(f64, f64)]) -> f64 {
    let pts: Vec<(f64, f64)> =
        points.iter().filter(|&&(x, y)| x > 0.0 && y > 0.0).map(|&(x, y)| (x.ln(), y.ln())).collect();
    let n = pts.len() as f64;
    if pts.len() < 2 {
        return 0.0;
    }
    let (sx, sy): (f64, f64) = pts.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let (mut num, mut den) = (0.0, 0.0);
    for &(x, y) in &pts {
        num += (x - mx) * (y - my);
        den += (x - mx) * (x - mx);
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §D5.1(f) single-creep ops gates (checked-in): a lone creep's route consumes REAL, bounded
    /// ops — > 0 (the counter is actually wired end-to-end through the kernel driver's stats
    /// seam, `resolve_moves_via_system_stats`) and never over the per-tick cap on ANY tick (the
    /// L4 invariant: the budget is enforced, not advisory). Exact ops printed for the record.
    #[test]
    fn single_creep_ops_are_wired_and_capped() {
        let (terrain, creeps) = parallel_lanes(1);
        let config = MoverConfig::default();
        let run = run_ops_bench(&terrain, &creeps, 200, &config);
        println!(
            "[ops single] route 30: {} ops over {} ticks (max/tick {}, cap {}), {} repaths",
            run.ops_consumed, run.ticks, run.max_tick_ops, run.ops_budget_cap, run.repaths
        );
        assert!(run.all_arrived, "the lone creep arrives");
        assert!(run.ops_consumed > 0, "pathfinding ops are actually counted");
        assert!(
            run.max_tick_ops <= run.ops_budget_cap,
            "per-tick ops never exceed the configured budget ({} > {})",
            run.max_tick_ops,
            run.ops_budget_cap
        );
    }

    /// The ops accounting is bit-deterministic (the determinism fence extended to the CPU lane):
    /// same world + config ⇒ identical ops, repaths, intents, ticks. Runs on the kernel driver
    /// itself since slice 7 (the historical bench mirror deleted itself when the stats seam
    /// landed) — a behavioral change to the driver moves these integers.
    #[test]
    fn ops_accounting_is_deterministic() {
        let (terrain, creeps) = shared_pinch(8);
        let config = MoverConfig::default();
        let a = run_ops_bench(&terrain, &creeps, 400, &config);
        let b = run_ops_bench(&terrain, &creeps, 400, &config);
        assert_eq!(a.ops_consumed, b.ops_consumed, "bit-identical ops");
        assert_eq!(a.repaths, b.repaths);
        assert_eq!(a.intents_issued, b.intents_issued);
        assert_eq!(a.ticks, b.ticks);
        assert!(a.all_arrived, "the 8-creep pinch clears");
    }

    /// The wall-clock SECONDARY (checked-in death-spiral guard, the combat-eval `bench.rs` shape):
    /// 8 creeps through the pinch — congestion, stuck escalation, repaths, per-tick matrix
    /// rebuilds, the expensive path. The bound is deliberately LOOSE (native host wall-clock is a
    /// relative proxy, module note): it clears debug AND release with an order of magnitude to
    /// spare and trips only on the algorithmic blowup class. Exact us/creep-tick printed — THE
    /// regression baseline number; the tight signal stays the op-counted primary.
    #[test]
    fn wall_clock_stays_out_of_the_death_spiral() {
        let (terrain, creeps) = shared_pinch(8);
        let config = MoverConfig::default();
        let run = run_ops_bench(&terrain, &creeps, 400, &config);
        println!(
            "[wall-clock pinch(8)] {:.1} us/creep-tick over {} ticks ({} ops, {} repaths, {:.1?} total)",
            run.us_per_creep_tick(), run.ticks, run.ops_consumed, run.repaths, run.elapsed
        );
        assert!(run.all_arrived);
        // LOOSE death-spiral bound: measured ~510 us/creep-tick DEBUG (2026-07-01; release is
        // an order of magnitude under that) — ~40x headroom in the worst profile.
        const BUDGET_US_PER_CREEP_TICK: f64 = 20_000.0;
        assert!(
            run.us_per_creep_tick() < BUDGET_US_PER_CREEP_TICK,
            "movement wall-clock blew the death-spiral bound: {:.1} us/creep-tick",
            run.us_per_creep_tick()
        );
    }

    /// §D5.3 scaling curve 1 (`#[ignore]`, run on demand:
    /// `cargo test -p screeps-rover-eval scaling_ops -- --ignored --nocapture`):
    /// ops/tick vs N creeps on PARALLEL LANES — free flow, so total work is N independent
    /// creeps' searches and the fitted log-log exponent must stay ~linear. THE gate for the
    /// congestion-blowup class: a super-linear lanes slope means creep count alone (not
    /// contention) inflates per-creep search work. LOOSE bound: slope ≤ 1.5 (measured N^0.79 on
    /// 2026-07-01 — mildly SUB-linear: larger fleets finish in similar tick counts, so the
    /// per-tick average amortizes the fixed search charges; re-measured N^0.75 later the same
    /// day under the dense-crowd fix set — reserve + windowed damper + freed-tiles chains).
    #[test]
    #[ignore]
    fn scaling_ops_vs_parallel_creeps() {
        let config = MoverConfig::default();
        let mut points = Vec::new();
        for &n in &[1usize, 2, 4, 8, 16, 32, 64] {
            let (terrain, creeps) = parallel_lanes(n);
            let run = run_ops_bench(&terrain, &creeps, 400, &config);
            assert!(run.all_arrived, "lanes({n}) must arrive");
            println!(
                "[lanes] N={n:<3} ops/tick={:>10.1} total_ops={:>8} ticks={:<4} repaths={} {:.1} us/creep-tick",
                run.ops_per_tick(), run.ops_consumed, run.ticks, run.repaths, run.us_per_creep_tick()
            );
            points.push((n as f64, run.ops_per_tick()));
        }
        let slope = log_log_slope(&points);
        println!("[lanes] fitted ops/tick ~ N^{slope:.3}");
        assert!(
            slope <= 1.5,
            "parallel-lane ops must scale ~linearly in N (the congestion-blowup gate), got N^{slope:.3}"
        );
    }

    /// §D5.3 scaling curve 2 (`#[ignore]`): ops/tick vs N creeps through ONE shared gap — the
    /// congestion case. Serialization + stuck-escalation repaths make some super-linearity
    /// STRUCTURAL here (each blocked creep repaths as the queue drains), so this curve is
    /// primarily a printed regression record; the bound is a very loose backstop against the
    /// runaway class (resolver recursion / repath storms), not a linearity gate.
    ///
    /// **The dense-crowd breakdown (found 2026-07-01, capped this curve at N=32; ROOT-CAUSED +
    /// FIXED the same day — curve restored to N=64).** As found: N=40 broke the standing
    /// `failed_coordination == 0` gate (43 of 2111 intents); N≥48 partially LIVELOCKED (42/48,
    /// 37/56, 38/64 arrived; 15k–34k engine-rejected intents; the ops pool SATURATED at the full
    /// 20 000 ops/tick for 8000+ ticks at N=64 — the live "CPU pathfinding death-spiral"
    /// signature, deadlock detector silent). THREE interlocking mechanisms, all fixed in rover:
    /// (1) *stuck-repath storm* — `needs_repath` was a LEVEL, so every immobile-≥tier-1 creep
    /// re-searched EVERY tick; a dense stuck crowd drained the whole per-tick ops pool
    /// indefinitely → fixed by the FIRST-PATH POOL RESERVE (repaths are optional work and may
    /// not consume the last fifth of the pool — `needs_path` searches always find real budget)
    /// plus the per-episode storm damper (`StuckState::should_stuck_repath_with`: free cadence
    /// inside the designed escalation window, `ticks_immobile ≤ report_failure`, where tiers
    /// still flip and combat coordination needs the fast recovery — the drain-soak bed
    /// adjudicated that; exponential spacing past tier 4, where the ladder has nothing new and a
    /// long jam's searches are pure waste — the ADR 0004 repath-storm class, IBEX-016's
    /// repurposed `repath_count`);
    /// (2) *pathless starvation* — mid-order creeps got dreg ops allowances, their `needs_path`
    /// searches failed incomplete forever, and a pathless creep was ALSO invisible to the
    /// resolver (`process()` Pass 1's `Err` arm inserted no occupancy entry — the third instance
    /// of the stationary-occupant hole) → the wedge nuclei other creeps were granted THROUGH
    /// (the coordination flood) → fixed by the `stationary_occupant` entry on path errors + the
    /// reserve clearing the starvation itself;
    /// (3) *double-booked shove chains* — `try_shove`'s frame-entry `firmly_occupied` snapshot
    /// goes stale across chain recursion, so a deep chain member could land on a tile the chain
    /// had already promised away (its own vacated tiles included), double-booking the move-set →
    /// fixed at the source by the `freed_tiles` chain stack (no member may land on any tile the
    /// active chain is vacating — the chain picks another landing and SUCCEEDS) with
    /// post-recursion re-checks kept as defence-in-depth (resolver.rs).
    /// Post-fix: every probed N (40/48/56/64/96) fully arrives with the audit ALL-ZERO through
    /// 64; saturation only during the cold-start path-distribution window. The checked-in gate
    /// is `dense_pinch_crowd_holds_all_gates` (crowd.rs); this curve re-pins the slope over the
    /// restored range: **N^1.244 over 1..64** (2026-07-01, the final fix set — vs the pre-fix
    /// N^1.35 that was only measurable to N=32).
    #[test]
    #[ignore]
    fn scaling_ops_at_the_shared_pinch() {
        const NS: &[usize] = &[1, 2, 4, 8, 16, 32, 40, 48, 56, 64];
        let config = MoverConfig::default();
        let mut points = Vec::new();
        for &n in NS {
            let (terrain, creeps) = shared_pinch(n);
            let run = run_ops_bench(&terrain, &creeps, 2_000, &config);
            assert!(run.all_arrived, "pinch({n}) must eventually clear");
            println!(
                "[pinch] N={n:<3} ops/tick={:>10.1} total_ops={:>8} ticks={:<4} repaths={} {:.1} us/creep-tick",
                run.ops_per_tick(), run.ops_consumed, run.ticks, run.repaths, run.us_per_creep_tick()
            );
            points.push((n as f64, run.ops_per_tick()));
        }
        let slope = log_log_slope(&points);
        println!("[pinch] fitted ops/tick ~ N^{slope:.3}");
        assert!(
            slope <= 3.0,
            "pinch ops blew even the structural-serialization allowance, got N^{slope:.3}"
        );
    }

    /// §D5.3 scaling curve 3 (`#[ignore]`): total ops to complete a route vs route length
    /// (10..40 tiles on an open plain). `reuse_path_length` is raised past every route so no
    /// EXPIRY repath contributes — but the sum is still several searches, not one: rover plans in
    /// SEGMENTS (each search's path is length-capped; when a path is consumed it becomes
    /// `needs_path` and re-searches — and `MovementTickStats::repaths` counts every
    /// `generate_path`, first/segment searches included, its field doc notwithstanding —
    /// `movementsystem.rs:1237`). So the y-axis is honestly "pathfinding ops to route L tiles",
    /// with ops-per-search printed alongside. MEASURED 2026-07-01: exactly ONE search of exactly
    /// 2000 ops per point, flat across L = 10..40 (slope 0.000) — the ops charge is per-ROOM
    /// granular (the base per-room allocation, `movementsystem.rs:1310` rooms×2000), not
    /// per-tile, so the curve only rises when routes span more rooms or the search escalates its
    /// budget. The loose bound still guards the flood-regression class (heuristic loss ⇒
    /// quadratic frontier ⇒ escalated multi-charge searches).
    #[test]
    #[ignore]
    fn scaling_ops_vs_route_length() {
        let config = MoverConfig { reuse_path_length: 1_000, ..MoverConfig::default() };
        let mut points = Vec::new();
        for &len in &[10u8, 15, 20, 25, 30, 35, 40] {
            let creeps =
                vec![CrowdCreep::new(balanced_hauler(), pos(5, 25), pos(5 + len, 25), 0)];
            let run = run_ops_bench(&SimTerrain::default(), &creeps, 200, &config);
            assert!(run.all_arrived, "route({len}) must arrive");
            println!(
                "[route] L={len:<3} total_ops={:>7} searches={} ops/search={:>7.0} ticks={}",
                run.ops_consumed,
                run.repaths,
                run.ops_consumed as f64 / (run.repaths.max(1)) as f64,
                run.ticks
            );
            points.push((len as f64, run.ops_consumed as f64));
        }
        let slope = log_log_slope(&points);
        println!("[route] fitted total ops ~ L^{slope:.3}");
        assert!(
            slope <= 2.0,
            "route-length ops growth suggests heuristic loss (flood regression), got L^{slope:.3}"
        );
    }
}
