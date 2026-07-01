//! The hauler objective benchmark (ADR 0033 §D5.4, v1 of the objective function): a fleet of
//! haulers cycles cargo source→sink→source through the **real rover `MovementSystem`**, and each
//! round trip is scored `η = T*_rtt / T_rtt` against the fatigue-exact oracle round trip, weighted
//! by the value in flight (`W = ρ·T* = Q`, the cargo). The aggregate **`H = Σ Wη / Σ W`** — with a
//! seeded-bootstrap 95% CI and percentiles ([`Summary`]) — is the operator's *"maximum value
//! transport by distance"*, the economic special case the unified `w(creep)` weight reduces to.
//!
//! The M3 physics does the work here: the loaded leg carries `carry_used = q` (loaded-CARRY
//! fatigue), the return leg is empty — so under-MOVE'd haulers stall loaded and fly back empty,
//! exactly like live. Loading/unloading is instant (this benchmark prices movement only). `H` is
//! **regression-tracked, never a `== 1` gate** (contention makes `H < 1` structural); the hard
//! layer underneath stays the [`IntentAudit`] failed-move sentinel and the deadlock detector.

use crate::cost::WorldCostSource;
use crate::crowd::IntentAudit;
use crate::oracle::optimal_path;
use crate::stats::Summary;
use crate::traverse::{pos_in, traverse_cycle};
use screeps::Position;
use screeps_sim_core::resolve_movement;
use screeps_sim_core::{
    resolve_moves_via_system_with, MoveIntents, MovementState, MoverConfig, SimBody, SimCreep,
    SimMoveCache, SimMoveRequest, SimTerrain,
};

/// Arrival range at both endpoints (1 = adjacent, like a real `transfer`/`withdraw`).
const ENDPOINT_RANGE: u32 = 1;
/// Consecutive zero-movement ticks (work remaining) that count as a deadlock.
const DEADLOCK_TICKS: u32 = 12;

/// One hauler's task: cycle `q` cargo units per round trip between `source` and `sink`, `trips`
/// times. Multiple assignments may share endpoints — that contention is what the fleet run measures.
#[derive(Clone)]
pub struct HaulAssignment {
    pub body: SimBody,
    /// Cargo units per loaded leg (energy-equivalent; non-energy cargo arrives pre-priced).
    pub q: u32,
    pub source: Position,
    pub sink: Position,
    pub trips: u32,
}

/// The outcome of a haul-fleet run.
#[derive(Clone, Debug)]
pub struct HaulOutcome {
    /// One `(η, W)` sample per expected trip: `η = T*_rtt / T_rtt` (0 for trips not completed
    /// within the cap), `W = Q` (the cargo — the §D5.4 hauler sample weight).
    pub samples: Vec<(f64, f64)>,
    /// `H` = the weighted mean of the samples, with the bootstrap CI + percentiles.
    pub summary: Summary,
    /// The failed-move sentinel over the whole run (hard gate — independent of `H`).
    pub audit: IntentAudit,
    pub completed_trips: u32,
    pub expected_trips: u32,
    pub ticks: u32,
    pub deadlocked: bool,
}

/// The fatigue-exact optimal round-trip ticks for one assignment: greedy-walk the fatigue-optimal
/// route loaded (carry `q`) source→sink, then empty back — in ONE continuous simulation
/// ([`traverse_cycle`]), so residual fatigue at the turn is paid exactly as a real creep pays it.
/// The route is the fatigue-cost optimum (the `r_ticks` approximation, documented there). `None`
/// if unreachable. **Single-room only** ([`optimal_path`] is room-blind over `(x, y)`) — a
/// cross-room assignment must use [`t_star_rtt_solo`] instead (the fleet runner routes this).
pub fn t_star_rtt(terrain: &SimTerrain, a: &HaulAssignment) -> Option<u32> {
    let room = a.source.room_name();
    let cap = 10_000;
    // Loaded leg: source → within range of sink; empty leg: from the turn tile back.
    let out = optimal_path(terrain, a.source, a.sink, ENDPOINT_RANGE as u8)?;
    let out_route: Vec<Position> = out.iter().map(|&(x, y)| pos_in(room, x, y)).collect();
    let turn = out_route.last().copied().unwrap_or(a.source);
    let back = optimal_path(terrain, turn, a.source, ENDPOINT_RANGE as u8)?;
    let back_route: Vec<Position> = back.iter().map(|&(x, y)| pos_in(room, x, y)).collect();
    let cycle = traverse_cycle(
        terrain,
        a.body.clone(),
        a.source,
        &[(a.q, &out_route), (0, &back_route)],
        cap,
    );
    cycle.reached.then_some(cycle.ticks)
}

/// Generous cap for the SOLO baseline run (mirrors `t_star_rtt`'s oracle-walk cap): any sane
/// assignment's uncontended round trip is orders of magnitude shorter; hitting it means the
/// scenario is mis-built (unreachable/starved solo), which `t_star_rtt_solo` reports as `None`.
const SOLO_TICK_CAP: u32 = 10_000;

/// The **SOLO baseline** T*: the assignment run as a 1-hauler fleet (`trips = 1`) through the same
/// [`simulate_fleet`] machinery in an otherwise-empty world, taking its realized round-trip ticks.
///
/// **Semantic difference from [`t_star_rtt`], loudly:** the oracle T* is the fatigue-exact
/// *optimal* round trip, so oracle-based η also penalizes route-quality loss. Solo-T* baselines
/// against **rover's own uncontended behavior under the same `config`** — η then measures pure
/// CONTENTION/coordination loss, and a config that solo-paths a poor route is NOT penalized here
/// (route quality vs the true optimum stays Tier-A's job: [`crate::metrics`] `R_ticks`/`R_fatigue`
/// over [`crate::traverse`]). It exists because the single-room oracle cannot price a cross-room
/// route ([`optimal_path`] is a room-grid Dijkstra over `(x, y)` alone — fed a cross-room pair it
/// would silently price the coordinates as if co-roomed); the multi-room optimum is ADR 0033
/// §D5.4 open decision #10. Single-room assignments keep the stricter oracle. The baseline is the
/// FIRST round trip alone (cold path cache, zero fatigue) — later contended trips inherit
/// fatigue/cache state, which is part of what the fleet run measures.
pub fn t_star_rtt_solo(terrain: &SimTerrain, a: &HaulAssignment, config: &MoverConfig) -> Option<u32> {
    let solo = HaulAssignment { trips: 1, ..a.clone() };
    let sim = simulate_fleet(terrain, std::slice::from_ref(&solo), SOLO_TICK_CAP, config);
    if sim.deadlocked {
        return None;
    }
    sim.trip_ticks[0].first().copied()
}

/// Which leg of the round trip a hauler is on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Leg {
    ToSink,
    ToSource,
}

/// The raw realized fleet simulation, pre-scoring: per-assignment completed round-trip tick
/// lists + the audit. Shared by [`run_haul_fleet_with`] (which scores it against T*) and by
/// [`t_star_rtt_solo`] (which IS the cross-room baseline: the same sim, fleet of one).
struct FleetSim {
    /// Completed round-trip tick counts, per assignment, in completion order.
    trip_ticks: Vec<Vec<u32>>,
    audit: IntentAudit,
    ticks: u32,
    deadlocked: bool,
}

/// Run the fleet: every hauler starts at its source, loaded, and cycles until its trips are done or
/// `tick_cap` elapses. Returns `None` if any assignment's oracle round trip is unsolvable (a
/// mis-built scenario — distinct from rover failing it). `seed` drives the bootstrap CI.
pub fn run_haul_fleet(
    terrain: &SimTerrain,
    fleet: &[HaulAssignment],
    tick_cap: u32,
    seed: u32,
) -> Option<HaulOutcome> {
    run_haul_fleet_with(terrain, fleet, tick_cap, seed, &MoverConfig::default())
}

/// [`run_haul_fleet`] under explicit rover tunables — the evaluation primitive the parameter
/// tournament ([`crate::tuning`]) scores one [`MoverConfig`] point with.
pub fn run_haul_fleet_with(
    terrain: &SimTerrain,
    fleet: &[HaulAssignment],
    tick_cap: u32,
    seed: u32,
    config: &MoverConfig,
) -> Option<HaulOutcome> {
    let n = fleet.len();
    // T* per assignment: same-room ⇒ the fatigue-exact ORACLE round trip (strict — η also prices
    // route quality); cross-room ⇒ the SOLO baseline (rover's own uncontended round trip under the
    // same config — η prices pure contention/coordination loss; see `t_star_rtt_solo`'s loud
    // semantic note for why the oracle cannot serve here).
    let t_stars: Vec<u32> = fleet
        .iter()
        .map(|a| {
            if a.source.room_name() == a.sink.room_name() {
                t_star_rtt(terrain, a)
            } else {
                t_star_rtt_solo(terrain, a, config)
            }
        })
        .collect::<Option<Vec<_>>>()?;

    let sim = simulate_fleet(terrain, fleet, tick_cap, config);

    // Samples: one per EXPECTED trip — completed trips score T*/T, never-completed trips score 0
    // (the catastrophic tail stays visible in the distribution, per the objective design).
    let mut samples: Vec<(f64, f64)> = Vec::new();
    let mut completed = 0u32;
    for i in 0..n {
        let w = fleet[i].q as f64; // W = ρ·T* = Q — the hauler sample weight
        for &rt in &sim.trip_ticks[i] {
            samples.push(((t_stars[i] as f64 / rt.max(1) as f64).min(1.0), w));
            completed += 1;
        }
        for _ in sim.trip_ticks[i].len() as u32..fleet[i].trips {
            samples.push((0.0, w));
        }
    }

    Some(HaulOutcome {
        summary: Summary::of(&samples, seed),
        samples,
        audit: sim.audit,
        completed_trips: completed,
        expected_trips: fleet.iter().map(|a| a.trips).sum(),
        ticks: sim.ticks,
        deadlocked: sim.deadlocked,
    })
}

/// Drive the fleet through the real rover `MovementSystem` tick loop (no scoring). Multi-room
/// worlds: `MovementState.rooms` is left EMPTY, so `terrain` serves every room a route touches
/// (mirrored rooms — the kernel's `terrain_for` fallback). That matches [`WorldCostSource`], which
/// answers the same matrix for any room; per-room terrain needs a `cost.rs` extension first
/// (owned elsewhere — until then multi-room scenarios must share one terrain shape).
fn simulate_fleet(
    terrain: &SimTerrain,
    fleet: &[HaulAssignment],
    tick_cap: u32,
    config: &MoverConfig,
) -> FleetSim {
    let n = fleet.len();
    let mut world = MovementState {
        terrain: terrain.clone(),
        creeps: fleet
            .iter()
            .enumerate()
            .map(|(i, a)| SimCreep {
                id: i as u32 + 1,
                owner: 0,
                pos: a.source,
                body: a.body.clone(),
                fatigue: 0,
                carry_used: a.q, // starts loaded (loading is instant; movement is what's priced)
            })
            .collect(),
        ..Default::default()
    };

    let mut leg = vec![Leg::ToSink; n];
    let mut trips_done = vec![0u32; n];
    let mut trip_start = vec![0u32; n];
    let mut trip_ticks: Vec<Vec<u32>> = vec![Vec::new(); n];
    let done = |trips_done: &[u32], i: usize| trips_done[i] >= fleet[i].trips;

    let mut cache = SimMoveCache::new();
    let mut audit = IntentAudit::default();
    let mut ticks = 0u32;
    let mut stall = 0u32;
    let mut deadlocked = false;

    for t in 0..tick_cap {
        if (0..n).all(|i| done(&trips_done, i)) {
            break;
        }
        // One request per active hauler, toward its current leg's goal.
        let reqs: Vec<SimMoveRequest> = (0..n)
            .filter(|&i| !done(&trips_done, i))
            .map(|i| {
                let goal = match leg[i] {
                    Leg::ToSink => fleet[i].sink,
                    Leg::ToSource => fleet[i].source,
                };
                SimMoveRequest::move_to(world.creeps[i].id, goal, ENDPOINT_RANGE)
            })
            .collect();
        let requested: std::collections::HashSet<u32> = reqs.iter().map(|r| r.creep).collect();
        let dirs = resolve_moves_via_system_with(
            &world,
            &reqs,
            &mut cache,
            WorldCostSource::new(terrain, &world),
            config,
        );

        let before: Vec<Position> = world.creeps.iter().map(|c| c.pos).collect();
        let fatigued: Vec<bool> = world.creeps.iter().map(|c| c.fatigue > 0).collect();
        let mut intents = MoveIntents::new();
        for (&id, &d) in &dirs {
            intents.set_move(id, d);
        }
        let tick_report = resolve_movement(&mut world, &intents);
        ticks = t + 1;
        audit.reconcile(&world, &dirs, &tick_report, &before, &fatigued, &requested);

        // Leg transitions: arrive at the sink → dump cargo, turn around; arrive back at the source
        // → the round trip is complete, reload instantly and (if trips remain) head out again.
        let mut moved_any = false;
        for i in 0..n {
            if done(&trips_done, i) {
                continue;
            }
            if world.creeps[i].pos != before[i] {
                moved_any = true;
            }
            match leg[i] {
                Leg::ToSink => {
                    if world.creeps[i].pos.get_range_to(fleet[i].sink) <= ENDPOINT_RANGE {
                        world.creeps[i].carry_used = 0; // instant unload
                        leg[i] = Leg::ToSource;
                    }
                }
                Leg::ToSource => {
                    if world.creeps[i].pos.get_range_to(fleet[i].source) <= ENDPOINT_RANGE {
                        trip_ticks[i].push(ticks - trip_start[i]);
                        trips_done[i] += 1;
                        if !done(&trips_done, i) {
                            world.creeps[i].carry_used = fleet[i].q; // instant reload
                            leg[i] = Leg::ToSink;
                            trip_start[i] = ticks;
                        }
                    }
                }
            }
        }
        if moved_any {
            stall = 0;
        } else {
            stall += 1;
            if stall >= DEADLOCK_TICKS {
                deadlocked = true;
                break;
            }
        }
    }

    FleetSim {
        trip_ticks,
        audit,
        ticks,
        deadlocked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use screeps::{Part, RoomCoordinate, RoomName};

    fn pos(x: u8, y: u8) -> Position {
        let room: RoomName = "W1N1".parse().unwrap();
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
    }
    /// A 1:1-MOVE'd hauler (2×CARRY + 2×MOVE): full speed loaded AND empty on plains.
    fn balanced_hauler() -> SimBody {
        SimBody::unboosted(&[Part::Carry, Part::Carry, Part::Move, Part::Move])
    }

    #[test]
    fn lone_hauler_on_open_plain_is_perfectly_efficient() {
        let terrain = SimTerrain::default();
        let fleet = [HaulAssignment {
            body: balanced_hauler(),
            q: 100,
            source: pos(10, 25),
            sink: pos(20, 25),
            trips: 3,
        }];
        let out = run_haul_fleet(&terrain, &fleet, 500, 1).expect("solvable");
        assert_eq!(out.completed_trips, 3, "all trips complete");
        assert!(!out.deadlocked);
        assert_eq!(out.audit.failed_moves, 0, "a lone hauler wastes no intents");
        assert!(
            (out.summary.weighted_mean - 1.0).abs() < 1e-9,
            "uncontended open-plain hauling is optimal: H = {}, samples = {:?}",
            out.summary.weighted_mean,
            out.samples
        );
    }

    #[test]
    fn under_moved_hauler_is_slower_but_blameless() {
        // 2×CARRY + 1×MOVE: loaded weight 2 vs drain 2 → stalls every other tick loaded; empty it
        // flies. The oracle T* prices the SAME stalls, so η stays 1 — body-bound slowness is never
        // blamed on rover — while the optimal round trip is honestly longer than the balanced body's.
        let terrain = SimTerrain::default();
        let mk = |body: SimBody| HaulAssignment {
            body,
            q: 100,
            source: pos(10, 25),
            sink: pos(20, 25),
            trips: 1,
        };
        let slow = mk(SimBody::unboosted(&[Part::Carry, Part::Carry, Part::Move]));
        let fast = mk(balanced_hauler());
        let t_slow = t_star_rtt(&terrain, &slow).unwrap();
        let t_fast = t_star_rtt(&terrain, &fast).unwrap();
        assert!(t_slow > t_fast, "loaded-CARRY fatigue lengthens the optimal trip ({t_slow} vs {t_fast})");

        let out = run_haul_fleet(&terrain, &[slow], 500, 1).expect("solvable");
        assert!(
            (out.summary.weighted_mean - 1.0).abs() < 1e-9,
            "fatigue stalls are body-bound, rover blameless: H = {}",
            out.summary.weighted_mean
        );
        // And the rate the role table prices reflects the slower cycle: ρ = Q/T*.
        use crate::value::Role;
        let (r_slow, r_fast) = (Role::Haul { q: 100 }.rate_e_t(t_slow), Role::Haul { q: 100 }.rate_e_t(t_fast));
        assert!(r_slow < r_fast, "the under-MOVE'd hauler's e/t rate is lower ({r_slow:.3} < {r_fast:.3})");
    }

    #[test]
    fn sample_weights_are_the_cargo() {
        let terrain = SimTerrain::default();
        let fleet = [
            HaulAssignment { body: balanced_hauler(), q: 100, source: pos(10, 20), sink: pos(20, 20), trips: 2 },
            HaulAssignment { body: balanced_hauler(), q: 300, source: pos(10, 30), sink: pos(20, 30), trips: 1 },
        ];
        let out = run_haul_fleet(&terrain, &fleet, 500, 1).expect("solvable");
        let total_w: f64 = out.samples.iter().map(|&(_, w)| w).sum();
        assert!((total_w - 500.0).abs() < 1e-9, "Σ weights = Σ cargo over expected trips (2×100 + 1×300)");
        // The big-cargo hauler dominates H: its single trip carries 60% of the fleet weight.
        assert!(out.samples.iter().any(|&(_, w)| (w - 300.0).abs() < 1e-9));
    }

    fn pos_in_room(room: &str, x: u8, y: u8) -> Position {
        let room: RoomName = room.parse().unwrap();
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
    }

    #[test]
    fn cross_room_route_scores_against_the_solo_baseline() {
        // W1N1(10,25) → W2N1(40,25): out through W1N1's WEST exit (x=0 relocates to W2N1 x=49 —
        // the kernel edge-exit rule, tick.rs), ~19 steps out + ~19 back. One shared plain terrain
        // serves both rooms (MovementState.rooms empty — the mirrored-room design).
        let terrain = SimTerrain::default();
        let a = HaulAssignment {
            body: balanced_hauler(),
            q: 100,
            source: pos_in_room("W1N1", 10, 25),
            sink: pos_in_room("W2N1", 40, 25),
            trips: 1,
        };
        let solo = t_star_rtt_solo(&terrain, &a, &MoverConfig::default())
            .expect("the border route is solo-solvable");
        // The trip genuinely crosses the border: the geometric minimum is 35 ticks — out 18
        // (10 steps to the exit, which relocates to W2N1 x=49 same-tick, + 8 to range 1 of the
        // sink) + back 17 (8 to the exit + 9 to range 1 of the source). Realized 37 (deterministic).
        assert!(solo >= 35, "solo round trip spans both rooms, got {solo} ticks");

        // A LONE cross-room hauler scores η ≡ 1 by construction: the fleet run IS the baseline run
        // (same deterministic sim, fleet of one, first trip) — solo-T* measures contention only.
        let out = run_haul_fleet(&terrain, &[a], 500, 1).expect("solvable");
        assert_eq!(out.completed_trips, 1);
        assert!(!out.deadlocked);
        assert!(
            (out.summary.weighted_mean - 1.0).abs() < 1e-9,
            "uncontended cross-room η is exactly 1 against the solo baseline: H = {}",
            out.summary.weighted_mean
        );
    }

    #[test]
    fn contended_border_route_completes_with_gates_held() {
        // Two haulers cycling the same cross-room route: the solo baseline prices each trip, the
        // contention shows up as H ≤ 1, and the hard gates (no deadlock, zero unexplained
        // rejections) must hold across the border exactly as they do in-room.
        let terrain = SimTerrain::default();
        let fleet: Vec<HaulAssignment> = (0..2)
            .map(|_| HaulAssignment {
                body: balanced_hauler(),
                q: 100,
                source: pos_in_room("W1N1", 10, 25),
                sink: pos_in_room("W2N1", 40, 25),
                trips: 2,
            })
            .collect();
        let out = run_haul_fleet(&terrain, &fleet, 800, 1).expect("solvable");
        assert_eq!(out.completed_trips, out.expected_trips, "all border trips complete");
        assert!(!out.deadlocked);
        assert_eq!(out.audit.failed_coordination, 0, "no unexplained rejections at the border");
        let h = out.summary.weighted_mean;
        assert!(h > 0.0 && h <= 1.0, "solo-baselined H stays in (0, 1]: {h}");
    }

    #[test]
    fn contended_shared_route_degrades_h_but_never_wastes_intents() {
        // Four haulers cycling the SAME source/sink through a one-gap wall: congestion must show up
        // as H < 1 (delayed trips), while the failed-move sentinel stays 0 and nothing deadlocks.
        let mut terrain = SimTerrain::default();
        for y in 0..=49 {
            if y != 25 {
                terrain.walls.insert((15, y));
            }
        }
        let fleet: Vec<HaulAssignment> = (0..4)
            .map(|_| HaulAssignment {
                body: balanced_hauler(),
                q: 100,
                source: pos(10, 25),
                sink: pos(20, 25),
                trips: 2,
            })
            .collect();
        let out = run_haul_fleet(&terrain, &fleet, 2000, 1).expect("solvable");
        assert_eq!(out.completed_trips, out.expected_trips, "everyone finishes eventually");
        assert!(!out.deadlocked);
        // Mechanism-aware gates (root-caused via the head-on tick trace, 2026-07-01): rejections
        // whose blocking chain ends at a PARKED (finished, unrequested) creep are rover's designed
        // optimism cost — `ticks_immobile ≥ 2` of failed intents per blocking event before the
        // friendly-avoid repath fires; bounded per event, never linear. (The linear-forever case was
        // the kernel driver's missing CPU budget — rover treats an absent budget as EXHAUSTED and
        // never stuck-repaths; fixed in sim-core.) Rejections involving ACTIVE creeps the resolver
        // planned are unexplained divergence and gate at zero, always.
        assert_eq!(
            out.audit.failed_coordination, 0,
            "unexplained active-creep rejections: {} of {}",
            out.audit.failed_coordination, out.audit.intents_issued
        );
        assert!(
            out.audit.failed_into_parked <= 8,
            "parked-blocker optimism must stay bounded per event: {} of {}",
            out.audit.failed_into_parked,
            out.audit.intents_issued
        );
        assert!(
            out.audit.failed_move_rate() < 0.05,
            "wasted-intent rate must be marginal, got {:.4}",
            out.audit.failed_move_rate()
        );
        let h = out.summary.weighted_mean;
        assert!(h > 0.0 && h < 1.0, "shared-route contention costs real efficiency: H = {h}");
        assert!(
            out.summary.ci95.0 <= h && h <= out.summary.ci95.1,
            "the bootstrap CI brackets H"
        );
        assert!(out.summary.p05 <= out.summary.p95);
    }
}
