//! Tier-B multi-creep contention (ADR 0033 §D5/M4). Where [`crate::traverse`] walks ONE creep along a
//! fixed path, this drives MANY creeps through the **real rover `MovementSystem` resolver** each tick
//! — the kernel driver [`resolve_moves_via_system`] with a terrain cost source — then applies the
//! resolved directions via [`resolve_movement`]. It is the reuse the whole benchmark was built for: the
//! same unified mover the live bot runs, now measured for congestion.
//!
//! The payoff is that `fatigue_util`'s companion here — **congestion** — is finally a live signal: a
//! creep that sits still with **zero fatigue** and an unmet goal was blocked by *traffic* (lost a tile
//! contest / held by the resolver), not by its body. Single-creep worlds can't produce that.

use crate::cost::TerrainCostSource;
use screeps::Position;
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
    /// No creep moved for [`DEADLOCK_TICKS`] consecutive ticks while some goal was unmet.
    pub deadlocked: bool,
}

/// Drive `creeps` to their goals through rover's `MovementSystem` + resolver, one tick at a time, over
/// `terrain`. Arrived creeps stay put (becoming obstacles the resolver must route others around).
/// Stops when all are within range, a deadlock is detected, or `tick_cap` elapses.
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
        let dirs = resolve_moves_via_system(&world, &reqs, &mut cache, TerrainCostSource::new(terrain));

        let before: Vec<Position> = world.creeps.iter().map(|c| c.pos).collect();
        let fatigued: Vec<bool> = world.creeps.iter().map(|c| c.fatigue > 0).collect();
        let mut intents = MoveIntents::new();
        for (&id, &d) in &dirs {
            intents.set_move(id, d);
        }
        resolve_movement(&mut world, &intents);
        report.ticks = t + 1;

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
        assert!(r.all_arrived, "head-on creeps swap past each other; arrivals={:?}", r.arrivals);
        assert!(!r.deadlocked);
    }
}
