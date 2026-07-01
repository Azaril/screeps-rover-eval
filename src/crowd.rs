//! Tier-B multi-creep contention (ADR 0033 §D5/M4). Where [`crate::traverse`] walks ONE creep along a
//! fixed path, this drives MANY creeps through the **real rover `MovementSystem` resolver** each tick
//! — the kernel driver [`resolve_moves_via_system`] with a terrain cost source — then applies the
//! resolved directions via [`resolve_movement`]. It is the reuse the whole benchmark was built for: the
//! same unified mover the live bot runs, now measured for congestion.
//!
//! The payoff is that `fatigue_util`'s companion here — **congestion** — is finally a live signal: a
//! creep that sits still with **zero fatigue** and an unmet goal was blocked by *traffic* (lost a tile
//! contest / held by the resolver), not by its body. Single-creep worlds can't produce that.

use crate::cost::WorldCostSource;
use screeps::Position;
use screeps_sim_core::movement::step;
use screeps_sim_core::{
    resolve_moves_via_system, MoveIntents, MovementState, SimBody, SimCreep, SimMoveCache,
    SimMoveRequest, SimTerrain,
};
use screeps_sim_core::resolve_movement;

/// Consecutive zero-movement ticks (some creep still travelling) that count as a deadlock.
const DEADLOCK_TICKS: u32 = 12;

/// One creep in a crowd scenario.
#[derive(Clone)]
pub struct CrowdCreep {
    pub body: SimBody,
    pub carry_used: u32,
    pub from: Position,
    pub goal: Position,
    pub range: u32,
}

impl CrowdCreep {
    /// A convenience constructor for an empty (unloaded) creep.
    pub fn new(body: SimBody, from: Position, goal: Position, range: u32) -> Self {
        CrowdCreep { body, carry_used: 0, from, goal, range }
    }
}

/// The outcome of a crowd run.
#[derive(Clone, Debug, Default)]
pub struct CrowdReport {
    /// Ticks elapsed (until all arrived, deadlock, or the cap).
    pub ticks: u32,
    /// Per-creep arrival tick (`None` = never reached its goal).
    pub arrivals: Vec<Option<u32>>,
    /// Every creep is within range of its goal at the end.
    pub all_arrived: bool,
    /// Creep-ticks spent still travelling (denominator for congestion / throughput).
    pub active_creep_ticks: u32,
    /// Steps taken across all creeps.
    pub moves: u32,
    /// Creep-ticks a travelling creep sat still with ZERO fatigue — blocked by traffic (the
    /// resolver held it or it lost a tile contest). The congestion signal.
    pub traffic_idle: u32,
    /// Creep-ticks a travelling creep could not move because it was fatigued (body-forced).
    pub fatigue_idle: u32,
    /// The issued-intent vs executed-move reconciliation (the failed-move sentinel).
    pub audit: IntentAudit,
    /// No creep moved for [`DEADLOCK_TICKS`] consecutive ticks while some goal was unmet.
    pub deadlocked: bool,
}

/// **Intent spent, no action taken** — the reconciliation of every `Direction` rover issued against
/// what the engine actually executed. The algorithmic-failure signal: every sim creep is
/// rover-controlled (no uncontrollable NPCs, unlike live), so a failed move means rover's issued
/// move-set was not self-consistent — unanticipated coordinated movement. The offline analogue of
/// the live G-13 wasted-move canary; in live each failure also burns intent CPU. Shared by every
/// multi-creep driver ([`run_crowd`], the haul benchmark).
#[derive(Clone, Debug, Default)]
pub struct IntentAudit {
    /// Move intents rover issued (a `Direction` per creep per tick).
    pub intents_issued: u32,
    /// Issued intents the engine rejected (Σ of the three classes below).
    pub failed_moves: u32,
    /// Failed moves issued for a creep that was FATIGUED at tick start (the engine's `canMove`
    /// ignores it) — rover has the fatigue data, so these are avoidable wasted intents.
    pub failed_fatigued: u32,
    /// Failed moves issued INTO a wall or off the room edge — an outright pathing bug.
    pub failed_wall: u32,
    /// Failed moves whose blocking chain terminates at a **parked** creep — one that reached its
    /// goal and left the request set. What the mover knows about it is
    /// [`screeps_sim_core::MoverConfig::register_idle_creeps`]'s call. Under the shipped default
    /// (ON since the parked-creep-coordination-v2 slice, 2026-07-01 — live parity) parked tiles
    /// are handed to the resolver up front, contests against them resolve as resolver DENIALS
    /// (which feed the escalation ladder via denial-as-stuck) or as shoves of the synthesized
    /// idle entries — never as engine-rejected intents — so this class is **0 by mechanism and
    /// gated `== 0` alongside `failed_coordination`**: any nonzero value means the resolver
    /// issued a move into a tile it was TOLD was occupied, true divergence. The historical
    /// registration-OFF behavior (rover's optimistic first-path prices no friendlies, a later
    /// creep burned `ticks_immobile ≥ 2` rejected intents per blocking event before the
    /// friendly-avoid repath fired — the designed, bounded optimism cost, root-caused 2026-07-01
    /// via the head-on tube trace) stays reachable through `register_idle_creeps: false` A/B
    /// runs, where this class is the bounded-burn regression signal it originally was.
    pub failed_into_parked: u32,
    /// Failed moves the resolver had FULL knowledge to avoid: contention lost to, or chain-blocked
    /// behind, an **active** (requested) creep. The resolver plans exactly these creeps as one
    /// move-set, so any rejection here is an unexplained resolver↔engine model divergence —
    /// **gated `== 0` everywhere**; a nonzero value means the sim (or rover) is wrong and must be
    /// investigated, not tolerated.
    pub failed_coordination: u32,
}

impl IntentAudit {
    /// Reconcile one tick's issued `dirs` against the executed `moved` set, classifying each
    /// rejected intent. `before`/`fatigued` are the tick-start snapshots indexed by `id − 1`;
    /// `requested` is the set of creep ids the driver put in this tick's `MovementData` — the
    /// creeps the resolver planned MOVES for (under `register_idle_creeps` the resolver may also
    /// know parked tiles as stationary occupants; the audit classes by requested-or-not either
    /// way, so the parked class stays comparable across both modes). A rejection is walked down
    /// its blocking chain (a rejected occupant's own destination, transitively): a chain ending at
    /// an unrequested (parked) creep is the known optimism cost (`failed_into_parked`); anything
    /// else is `failed_coordination`.
    pub(crate) fn reconcile(
        &mut self,
        world: &MovementState,
        dirs: &std::collections::HashMap<u32, screeps::Direction>,
        tick_report: &screeps_sim_core::MovementReport,
        before: &[Position],
        fatigued: &[bool],
        requested: &std::collections::HashSet<u32>,
    ) {
        self.intents_issued += dirs.len() as u32;
        // Tick-start occupancy: tile → creep id (living creeps only; one per tile in a valid world).
        let occupant: std::collections::HashMap<Position, u32> = world
            .creeps
            .iter()
            .filter(|c| c.is_alive())
            .map(|c| (before[(c.id - 1) as usize], c.id))
            .collect();

        for (&id, &dir) in dirs {
            if tick_report.moved.contains_key(&id) {
                continue;
            }
            self.failed_moves += 1;
            let i = (id - 1) as usize;
            if fatigued[i] {
                self.failed_fatigued += 1;
                continue;
            }
            let mut dest = match step(before[i], dir) {
                None => {
                    self.failed_wall += 1; // off the room edge
                    continue;
                }
                Some(d) => d,
            };
            if world.terrain_for(dest.room_name()).is_wall(dest.x().u8(), dest.y().u8()) {
                self.failed_wall += 1;
                continue;
            }
            // Walk the blocking chain: while the blocker is an ACTIVE creep that was itself
            // rejected, follow ITS destination. Terminate at a parked (unrequested) creep →
            // into_parked; at anything else (active stayer, contention loss, cycle) → coordination.
            let mut visited = std::collections::HashSet::new();
            let class = loop {
                let Some(&occ_id) = occupant.get(&dest) else {
                    // Empty tile and still rejected → a pure contention loss to another mover.
                    break "coordination";
                };
                if !requested.contains(&occ_id) {
                    break "parked";
                }
                if tick_report.moved.contains_key(&occ_id) {
                    break "coordination"; // the blocker moved — our creep lost the vacated tile
                }
                let Some(&occ_dir) = dirs.get(&occ_id) else {
                    break "coordination"; // active blocker the resolver chose to hold
                };
                if !visited.insert(occ_id) {
                    break "coordination"; // a blocking cycle among active creeps
                }
                match step(before[(occ_id - 1) as usize], occ_dir) {
                    Some(next) => dest = next, // follow the chain to what blocked the blocker
                    None => break "coordination",
                }
            };
            if class == "parked" {
                self.failed_into_parked += 1;
            } else {
                self.failed_coordination += 1;
            }
        }
    }

    /// Failed-move rate: the share of issued move intents the engine rejected. The hard sentinel —
    /// `0` when rover's issued move-set is fully self-consistent under the engine's contention
    /// rules; `> 0` is the failure class live NPC interference would only worsen.
    pub fn failed_move_rate(&self) -> f64 {
        if self.intents_issued == 0 {
            return 0.0;
        }
        self.failed_moves as f64 / self.intents_issued as f64
    }
}

/// Drive `creeps` to their goals through rover's `MovementSystem` + resolver, one tick at a time, over
/// `terrain`. Arrived creeps stay put and become parked blockers — this driver runs the default
/// [`screeps_sim_core::MoverConfig`] (`register_idle_creeps: true` since the coordination-v2
/// slice), so the resolver knows every parked tile up front: contests against parkers resolve as
/// denials/shoves inside the resolver, and [`IntentAudit::failed_into_parked`] gates `== 0` (see
/// the field doc). Stops when all are within range, a deadlock is detected, or `tick_cap` elapses.
pub fn run_crowd(terrain: &SimTerrain, creeps: &[CrowdCreep], tick_cap: u32) -> CrowdReport {
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
    let mut report = CrowdReport { arrivals: vec![None; n], ..Default::default() };
    let mut stall = 0u32;

    for t in 0..tick_cap {
        if (0..n).all(|i| within(world.creeps[i].pos, i)) {
            break;
        }
        let reqs: Vec<SimMoveRequest> = (0..n)
            .filter(|&i| !within(world.creeps[i].pos, i))
            .map(|i| SimMoveRequest::move_to(world.creeps[i].id, goals[i].0, goals[i].1))
            .collect();
        let requested: std::collections::HashSet<u32> = reqs.iter().map(|r| r.creep).collect();
        let dirs = resolve_moves_via_system(&world, &reqs, &mut cache, WorldCostSource::new(terrain, &world));

        let before: Vec<Position> = world.creeps.iter().map(|c| c.pos).collect();
        let fatigued: Vec<bool> = world.creeps.iter().map(|c| c.fatigue > 0).collect();
        let mut intents = MoveIntents::new();
        for (&id, &d) in &dirs {
            intents.set_move(id, d);
        }
        let tick_report = resolve_movement(&mut world, &intents);
        report.ticks = t + 1;
        report.audit.reconcile(&world, &dirs, &tick_report, &before, &fatigued, &requested);

        let mut moved_any = false;
        for i in 0..n {
            if within(before[i], i) {
                if report.arrivals[i].is_none() {
                    report.arrivals[i] = Some(t);
                }
                continue; // already satisfied at tick start — not an active traveller this tick
            }
            report.active_creep_ticks += 1;
            let now = world.creeps[i].pos;
            if now != before[i] {
                report.moves += 1;
                moved_any = true;
            } else if fatigued[i] {
                report.fatigue_idle += 1;
            } else {
                report.traffic_idle += 1;
            }
            if within(now, i) && report.arrivals[i].is_none() {
                report.arrivals[i] = Some(t + 1);
            }
        }
        if moved_any {
            stall = 0;
        } else {
            stall += 1;
            if stall >= DEADLOCK_TICKS {
                report.deadlocked = true;
                break;
            }
        }
    }

    report.all_arrived = (0..n).all(|i| within(world.creeps[i].pos, i));
    report
}

/// Congestion: the share of active creep-ticks lost to *traffic* (zero-fatigue idle). `0` in free
/// flow; rises as creeps contend for tiles. The Tier-B analogue of a single creep's stalls.
pub fn congestion(r: &CrowdReport) -> f64 {
    if r.active_creep_ticks == 0 {
        return 0.0;
    }
    r.traffic_idle as f64 / r.active_creep_ticks as f64
}

/// Crowd `fatigue_util` = `1 − congestion`: the share of travel that was productive or fatigue-forced
/// (not lost to traffic). `1.0` in free flow; `< 1` exactly when the resolver made creeps wait.
pub fn crowd_fatigue_util(r: &CrowdReport) -> f64 {
    1.0 - congestion(r)
}


#[cfg(test)]
mod tests {
    use super::*;
    use screeps::{Part, RoomCoordinate, RoomName};

    fn pos(x: u8, y: u8) -> Position {
        let room: RoomName = "W1N1".parse().unwrap();
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
    }
    fn balanced() -> SimBody {
        SimBody::unboosted(&[Part::Attack, Part::Move])
    }

    /// Parallel lanes, two tiles apart — no creep ever contends for another's tile. Free flow:
    /// everyone arrives, zero congestion, no deadlock.
    #[test]
    fn parallel_lanes_flow_without_congestion() {
        let terrain = SimTerrain::default();
        let creeps: Vec<CrowdCreep> = [20u8, 22, 24, 26]
            .iter()
            .map(|&y| CrowdCreep::new(balanced(), pos(10, y), pos(20, y), 0))
            .collect();
        let r = run_crowd(&terrain, &creeps, 100);
        assert!(r.all_arrived, "all creeps reach their lanes' goals");
        assert!(!r.deadlocked);
        assert!(congestion(&r) < 1e-9, "parallel lanes never contend, got congestion {}", congestion(&r));
        assert!((crowd_fatigue_util(&r) - 1.0).abs() < 1e-9);
        assert_eq!(r.audit.failed_moves, 0, "free flow must waste zero intents ({} issued)", r.audit.intents_issued);
    }

    /// A wall spanning the room with a single one-tile gap: every creep must funnel through it, so the
    /// resolver serialises them — congestion is positive — but all still arrive without deadlock.
    #[test]
    fn corridor_pinch_serialises_but_all_arrive() {
        let mut terrain = SimTerrain::default();
        for y in 0..=49 {
            if y != 25 {
                terrain.walls.insert((15, y)); // wall at x=15 with a gap only at y=25
            }
        }
        let creeps: Vec<CrowdCreep> = [(23u8, 24u8), (24, 25), (26, 26), (27, 23)]
            .iter()
            .map(|&(fy, gy)| CrowdCreep::new(balanced(), pos(10, fy), pos(20, gy), 0))
            .collect();
        let r = run_crowd(&terrain, &creeps, 300);
        assert!(r.all_arrived, "all creeps thread the gap eventually; arrivals={:?}", r.arrivals);
        assert!(!r.deadlocked, "a single-gap pinch must serialise, not deadlock");
        // Only one creep can occupy the single gap tile per tick, so the crowd is forced to pass
        // one-at-a-time — observable as a spread in arrival ticks (serialisation), even though the
        // resolver shoves blocked creeps sideways rather than letting them idle (so `congestion` may
        // be ~0 — a *good* property of the mover; the metric is a regression sentinel, not a target).
        let arr: Vec<u32> = r.arrivals.iter().map(|a| a.expect("all arrived")).collect();
        let spread = arr.iter().max().unwrap() - arr.iter().min().unwrap();
        assert!(spread >= 3, "4 creeps through one gap must serialise (arrival spread ≥ 3), got {arr:?}");
        assert!((0.0..=1.0).contains(&congestion(&r)), "congestion is a valid fraction");
        // Measured baseline 2026-07-01: 52 intents, 0 failed — the resolver pre-resolves contention
        // and only issues self-consistent move-sets. Hard gate: a failed move here is a regression.
        assert_eq!(
            r.audit.failed_moves, 0,
            "pinch: rover wasted {} of {} intents (coordination={})",
            r.audit.failed_moves, r.audit.intents_issued, r.audit.failed_coordination
        );
    }

    /// Two creeps head-on in a one-tile-wide tube must swap past each other (rover `allow_swap`), and
    /// both reach the far end.
    #[test]
    fn head_on_in_a_tube_swaps_and_both_arrive() {
        let mut terrain = SimTerrain::default();
        for x in 9..=15 {
            terrain.walls.insert((x, 24));
            terrain.walls.insert((x, 26)); // a 1-wide tube along y=25, x in 10..=14
        }
        let creeps = vec![
            CrowdCreep::new(balanced(), pos(10, 25), pos(14, 25), 0),
            CrowdCreep::new(balanced(), pos(14, 25), pos(10, 25), 0),
        ];
        let r = run_crowd(&terrain, &creeps, 100);
        assert!(r.all_arrived, "head-on creeps pass each other; arrivals={:?}", r.arrivals);
        assert!(!r.deadlocked);
        // ROOT-CAUSED baseline (tick trace, 2026-07-01): rover does NOT swap — it retreats one creep
        // out of the tube and re-routes it around while the other marches through. The winner then
        // PARKS on its goal (the tube mouth), leaves the request set. Under the pre-tuning default
        // (`reuse_path_length: 5`) the re-routed creep re-optimized back onto the blocked short path
        // and burned exactly `ticks_immobile ≥ 2` failed intents at the parked blocker; the
        // tournament-tuned commitment default (20) made it stick to its detour, and registration-ON
        // (coordination-v2) now prices the parked mouth up front as well — both mechanisms
        // independently hold this at zero.
        assert_eq!(
            r.audit.failed_coordination, 0,
            "any active-creep rejection is an unexplained resolver↔engine divergence"
        );
        assert_eq!(
            r.audit.failed_into_parked, 0,
            "path commitment avoids the parked mouth blocker entirely: {} of {} intents",
            r.audit.failed_into_parked, r.audit.intents_issued
        );
    }

    /// The nastiest coordination pattern: 8 creeps on a ring, each targeting the diametrically
    /// opposite point, so every shortest path crosses the same centre tiles at the same time. rover
    /// must still emit only self-consistent move-sets — **zero failed moves** ("intent spent, no
    /// action" = the algorithmic-failure sentinel; every creep here is ours, no NPC excuse).
    #[test]
    fn crossing_swarm_wastes_no_intents() {
        let terrain = SimTerrain::default();
        let ring: [(u8, u8); 8] = [
            (25, 20), (28, 21), (30, 25), (28, 29), (25, 30), (22, 29), (20, 25), (22, 21),
        ];
        let creeps: Vec<CrowdCreep> = ring
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                let (gx, gy) = ring[(i + 4) % 8]; // the opposite point on the ring
                CrowdCreep::new(balanced(), pos(x, y), pos(gx, gy), 0)
            })
            .collect();
        let r = run_crowd(&terrain, &creeps, 200);
        assert!(r.all_arrived, "all 8 cross to the opposite side; arrivals={:?}", r.arrivals);
        assert!(!r.deadlocked);
        assert_eq!(
            r.audit.failed_coordination, 0,
            "crossing swarm: {} unexplained rejections of {} intents (fatigued={} wall={} parked={})",
            r.audit.failed_coordination, r.audit.intents_issued, r.audit.failed_fatigued,
            r.audit.failed_wall, r.audit.failed_into_parked
        );
        // Early finishers park ON the ring where late paths cross. Under registration-ON (the
        // shipped default) those parked tiles are resolver-known, so contests against them are
        // denials/shoves, never engine rejections — the class gates at zero like coordination
        // (was `<= 4` bounded optimism burn under the historical registration-OFF default).
        assert_eq!(
            r.audit.failed_into_parked, 0,
            "registered parkers never draw engine-rejected intents: {} of {}",
            r.audit.failed_into_parked, r.audit.intents_issued
        );
    }
}
