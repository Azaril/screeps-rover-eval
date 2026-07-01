//! The ground-truth optimum rover is scored against — rover's own `room_grid_dijkstra` over the
//! engine fatigue cost field. Single-room, exact (Dijkstra), so it is a true lower bound on
//! fatigue-weighted path cost. (Multi-creep optima are NP-hard — not this; that's a later regression
//! baseline, ADR 0033 §D4.)

use crate::cost::fatigue_rate;
use screeps::Position;
use screeps_rover::room_grid_dijkstra;
use screeps_sim_core::SimTerrain;
use std::collections::HashSet;

/// The fatigue-optimal single-room path from `from` to within `range` of `goal`, as tile coords
/// (excluding `from`). `None` if `goal` is unreachable. Uses the SAME fatigue field rover searched,
/// so the two are directly comparable.
pub fn optimal_path(terrain: &SimTerrain, from: Position, goal: Position, range: u8) -> Option<Vec<(u8, u8)>> {
    let walls: HashSet<(u8, u8)> = terrain.walls.iter().copied().collect();
    let enter_cost = |x: u8, y: u8| -> Option<u64> {
        if walls.contains(&(x, y)) {
            None // impassable
        } else {
            Some(fatigue_rate(terrain, x, y) as u64)
        }
    };
    room_grid_dijkstra(
        &enter_cost,
        (from.x().u8(), from.y().u8()),
        (goal.x().u8(), goal.y().u8()),
        range,
    )
}
