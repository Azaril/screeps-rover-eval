//! Pathing-quality metrics (ADR 0033 §D5). The first: **route optimality** `R_fatigue` — the
//! fatigue cost of rover's chosen route divided by the fatigue-optimal cost. ≥ 1 always; `1.0` means
//! rover found a fatigue-optimal route, `> 1` means rover's cost field disagrees with the engine
//! fatigue field (the single most actionable finding the bench exists to surface — rover's field is
//! not body-aware today).

use crate::cost::fatigue_rate;
use crate::oracle::optimal_path;
use crate::pathing::rover_path;
use screeps::Position;
use screeps_sim_core::SimTerrain;

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
        assert!(r >= 1.0 && r <= 1.05, "rover should skirt the swamp near-optimally, got {r}");
    }
}
