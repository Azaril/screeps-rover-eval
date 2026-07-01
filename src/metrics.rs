//! Pathing-quality metrics (ADR 0033 §D5).
//!
//! - **Route optimality** `R_fatigue` — the fatigue *cost* of rover's chosen route ÷ the
//!   fatigue-optimal cost. ≥ 1 always; `1.0` means rover found a fatigue-optimal route, `> 1` means
//!   rover's cost field disagrees with the engine fatigue field (rover's field is not body-aware).
//! - **Travel time** `R_ticks` — the *ticks* rover's creep takes to walk its route ÷ the ticks the
//!   same creep takes to walk the fatigue-optimal route. Unlike `R_fatigue` this is dynamic: it
//!   counts the ticks a fatigued creep sits still, so it sees a route that stalls an under-MOVE body.
//! - **`movement_eff`** — steps ÷ ticks over a traversal (`1.0` = moved every tick; `< 1` = stalled).
//! - **`fatigue_util`** — the share of a traversal's idle that is *fatigue-forced* (body-bound, rover
//!   blameless). `1.0` for a lone creep on a valid route; `< 1` flags rover-attributable idle (a
//!   move the creep could have made but didn't). Only becomes non-trivial under contention (Tier B).

use crate::cost::fatigue_rate;
use crate::oracle::optimal_path;
use crate::pathing::rover_path;
use crate::traverse::{pos_in, traverse, Traversal};
use screeps::Position;
use screeps_sim_core::{SimBody, SimTerrain};

/// Fatigue cost of a rover path (Vec of `Position`, excluding the start): Σ fatigue_rate(tile).
pub fn path_fatigue_cost(terrain: &SimTerrain, path: &[Position]) -> u32 {
    path.iter().map(|p| fatigue_rate(terrain, p.x().u8(), p.y().u8())).sum()
}

/// Fatigue cost of an oracle path (tile coords, excluding the start).
pub fn coords_fatigue_cost(terrain: &SimTerrain, path: &[(u8, u8)]) -> u32 {
    path.iter().map(|&(x, y)| fatigue_rate(terrain, x, y)).sum()
}

/// `R_fatigue` for a single-creep single-room route: rover's fatigue cost ÷ the optimum's.
/// `None` if rover's search was incomplete or the goal is unreachable by the oracle (an
/// oracle-unsolvable goal — a distinct outcome the caller inspects, not a ratio).
pub fn r_fatigue(terrain: &SimTerrain, from: Position, goal: Position, range: u32) -> Option<f64> {
    let (rover, incomplete) = rover_path(terrain, from, goal, range);
    if incomplete {
        return None;
    }
    let optimum = optimal_path(terrain, from, goal, range as u8)?;
    let rover_cost = path_fatigue_cost(terrain, &rover);
    let optimum_cost = coords_fatigue_cost(terrain, &optimum);
    if optimum_cost == 0 {
        return Some(1.0); // already within range
    }
    Some(rover_cost as f64 / optimum_cost as f64)
}

/// The tick cap for a single-room traversal — generous (a badly under-MOVE'd creep on swamp is slow
/// but bounded); reaching it means the route was not completed and the caller gets `None`.
const TRAVERSAL_TICK_CAP: u32 = 5_000;

/// Walk `body` (carrying `carry_used`) along rover's chosen route and report the [`Traversal`] —
/// the raw material for [`movement_eff`] / [`fatigue_util`]. `None` if rover's search was incomplete.
pub fn rover_traversal(
    terrain: &SimTerrain,
    body: &SimBody,
    carry_used: u32,
    from: Position,
    goal: Position,
    range: u32,
) -> Option<Traversal> {
    let (route, incomplete) = rover_path(terrain, from, goal, range);
    if incomplete {
        return None;
    }
    Some(traverse(terrain, body.clone(), carry_used, from, &route, TRAVERSAL_TICK_CAP))
}

/// `R_ticks`: the ticks rover's creep needs to walk its route ÷ the ticks the same creep needs to
/// walk the fatigue-optimal route. `≈ 1` when rover's route is as fast to traverse as the optimum;
/// `> 1` when rover picked a route that stalls this body more (e.g. through swamp an under-MOVE body
/// should have skirted). `None` if either search/traversal did not complete.
///
/// The denominator route is the *fatigue-cost* optimum (rover's `room_grid_dijkstra`), not a true
/// time-expanded (tile × fatigue) optimum — for reasonably-MOVE'd bodies they coincide, and the
/// fatigue-optimal route already minimises stalls; the time-expanded oracle is a later refinement.
pub fn r_ticks(
    terrain: &SimTerrain,
    body: &SimBody,
    carry_used: u32,
    from: Position,
    goal: Position,
    range: u32,
) -> Option<f64> {
    let (rover_route, incomplete) = rover_path(terrain, from, goal, range);
    if incomplete {
        return None;
    }
    let oracle_coords = optimal_path(terrain, from, goal, range as u8)?;
    let room = from.room_name();
    let oracle_route: Vec<Position> = oracle_coords.iter().map(|&(x, y)| pos_in(room, x, y)).collect();

    let rover = traverse(terrain, body.clone(), carry_used, from, &rover_route, TRAVERSAL_TICK_CAP);
    let optimum = traverse(terrain, body.clone(), carry_used, from, &oracle_route, TRAVERSAL_TICK_CAP);
    if !rover.reached || !optimum.reached {
        return None;
    }
    if optimum.ticks == 0 {
        return Some(1.0); // already within range
    }
    Some(rover.ticks as f64 / optimum.ticks as f64)
}

/// Fraction of ticks on which the creep actually stepped: `moves / ticks`. `1.0` = never stalled.
pub fn movement_eff(t: &Traversal) -> f64 {
    if t.ticks == 0 {
        return 1.0;
    }
    t.moves as f64 / t.ticks as f64
}

/// Share of a traversal's ticks that were *productive or fatigue-forced* (`1 − idle_free/ticks`).
/// `1.0` means every non-moving tick was a legitimate fatigue stall — all slowness is body-bound and
/// rover is blameless; `< 1` means the creep sat idle with a move available (rover-attributable).
pub fn fatigue_util(t: &Traversal) -> f64 {
    if t.ticks == 0 {
        return 1.0;
    }
    (t.moves + t.idle_fatigued) as f64 / t.ticks as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use screeps::{Position, RoomCoordinate, RoomName};
    use screeps_sim_core::SimTerrain;

    fn pos(x: u8, y: u8) -> Position {
        let room: RoomName = "W1N1".parse().unwrap();
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
    }

    /// A1 (open plain): rover walks a straight line — fatigue-optimal, `R_fatigue == 1`.
    #[test]
    fn open_plain_is_fatigue_optimal() {
        let terrain = SimTerrain::default(); // all plain
        let r = r_fatigue(&terrain, pos(10, 25), pos(20, 25), 0).expect("reachable, complete");
        assert!((r - 1.0).abs() < 1e-9, "open-plain route must be fatigue-optimal, got {r}");
    }

    /// A wall wall forces a detour; rover must find a near-optimal way around it (`R_fatigue ≈ 1`).
    #[test]
    fn wall_detour_is_near_optimal() {
        let mut terrain = SimTerrain::default();
        // A vertical wall at x=15, y in 20..=30, with a gap only at the top and bottom.
        for y in 21..=29 {
            terrain.walls.insert((15, y));
        }
        let r = r_fatigue(&terrain, pos(10, 25), pos(20, 25), 0).expect("reachable, complete");
        assert!(r >= 1.0, "rover can never beat the optimum, got {r}");
        assert!(r <= 1.05, "rover must route around the wall near-optimally, got {r}");
    }

    /// A swamp block: the fatigue-optimal route skirts the swamp (plain 2 ≪ swamp 10). rover, pricing
    /// the same field, should agree (`R_fatigue ≈ 1`).
    #[test]
    fn swamp_is_skirted_near_optimally() {
        let mut terrain = SimTerrain::default();
        for x in 13..=17 {
            for y in 23..=27 {
                terrain.swamps.insert((x, y));
            }
        }
        let r = r_fatigue(&terrain, pos(10, 25), pos(20, 25), 0).expect("reachable, complete");
        assert!((1.0..=1.05).contains(&r), "rover should skirt the swamp near-optimally, got {r}");
    }

    /// A road punched straight through a full-height swamp band: the fatigue-optimal route runs down
    /// the road (cost 1/tile), not around the band. rover, pricing roads at 1 (`road_cost`), agrees.
    #[test]
    fn road_corridor_across_a_swamp_is_taken() {
        let mut terrain = SimTerrain::default();
        for x in 13..=17 {
            for y in 0..=49 {
                terrain.swamps.insert((x, y)); // a swamp band spanning the room — skirting is costly
            }
        }
        for x in 13..=17 {
            terrain.roads.insert((x, 25)); // a straight road through the band at y=25
        }
        let r = r_fatigue(&terrain, pos(10, 25), pos(20, 25), 0).expect("reachable, complete");
        assert!((1.0..=1.05).contains(&r), "rover should take the road across the swamp band, got {r}");
    }

    use screeps::Part;
    use screeps_sim_core::SimBody;

    /// A well-MOVE'd body ([ATTACK, MOVE]: weight 1, +2/step, clears 2/tick) sustains full speed on
    /// plains — moves every tick, no stalls, and its straight route is time-optimal.
    #[test]
    fn well_moved_body_sustains_full_speed() {
        let terrain = SimTerrain::default();
        let body = SimBody::unboosted(&[Part::Attack, Part::Move]);
        let t = rover_traversal(&terrain, &body, 0, pos(10, 25), pos(20, 25), 0).expect("complete");
        assert!(t.reached);
        assert!((movement_eff(&t) - 1.0).abs() < 1e-9, "sustains 1 step/tick, got {}", movement_eff(&t));
        assert!((fatigue_util(&t) - 1.0).abs() < 1e-9, "no idle at all");
        let r = r_ticks(&terrain, &body, 0, pos(10, 25), pos(20, 25), 0).expect("complete");
        assert!((r - 1.0).abs() < 1e-9, "straight plains route is time-optimal, got {r}");
    }

    /// An under-MOVE body ([2×ATTACK, MOVE]: weight 2, +4/step, clears 2/tick) must wait every other
    /// tick — `movement_eff ≈ 0.5` — but every idle tick is a legitimate fatigue stall, so
    /// `fatigue_util == 1` (rover blameless) and its straight route is still time-optimal.
    #[test]
    fn under_moved_body_stalls_but_blamelessly() {
        let terrain = SimTerrain::default();
        let body = SimBody::unboosted(&[Part::Attack, Part::Attack, Part::Move]);
        let t = rover_traversal(&terrain, &body, 0, pos(10, 25), pos(20, 25), 0).expect("complete");
        assert!(t.reached);
        assert!(movement_eff(&t) > 0.45 && movement_eff(&t) < 0.55, "moves ~every other tick, got {}", movement_eff(&t));
        assert!((fatigue_util(&t) - 1.0).abs() < 1e-9, "all idle is fatigue-forced, got {}", fatigue_util(&t));
        assert_eq!(t.idle_free, 0, "a lone creep on a valid route never idles with a move available");
        let r = r_ticks(&terrain, &body, 0, pos(10, 25), pos(20, 25), 0).expect("complete");
        assert!((r - 1.0).abs() < 1e-9, "under-MOVE on open plains: straight route still optimal, got {r}");
    }

    /// A loaded hauler travels slower than an empty one: [2×CARRY, MOVE] is weightless empty (full
    /// speed) but weight-2 at 100 load (stalls every other tick), so the same route takes more ticks.
    #[test]
    fn loaded_hauler_is_slower_than_empty() {
        let terrain = SimTerrain::default();
        let body = SimBody::unboosted(&[Part::Carry, Part::Carry, Part::Move]);
        let empty = rover_traversal(&terrain, &body, 0, pos(10, 25), pos(20, 25), 0).expect("complete");
        let loaded = rover_traversal(&terrain, &body, 100, pos(10, 25), pos(20, 25), 0).expect("complete");
        assert!((movement_eff(&empty) - 1.0).abs() < 1e-9, "an empty hauler sustains full speed");
        assert!(movement_eff(&loaded) < 0.6, "a loaded hauler stalls, got {}", movement_eff(&loaded));
        assert!(loaded.ticks > empty.ticks, "loaded {} must exceed empty {}", loaded.ticks, empty.ticks);
    }

    /// `R_ticks` rewards avoiding a stall trap: an under-MOVE body should skirt a lone swamp tile (a
    /// plains diagonal, same distance) rather than eat the swamp's huge fatigue hit. rover does, so
    /// its route is as fast to traverse as the fatigue-optimal one (`R_ticks ≈ 1`).
    #[test]
    fn r_ticks_rewards_skirting_a_swamp_for_an_under_move_body() {
        let mut terrain = SimTerrain::default();
        terrain.swamps.insert((15, 25)); // one swamp tile straddling the direct line
        let body = SimBody::unboosted(&[Part::Attack, Part::Attack, Part::Move]);
        let r = r_ticks(&terrain, &body, 0, pos(10, 25), pos(20, 25), 0).expect("complete");
        assert!((1.0..=1.1).contains(&r), "rover skirts the swamp near-time-optimally, got {r}");
    }
}
