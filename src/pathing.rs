//! Drive rover's real `LocalPathfinder` over a `SimTerrain` (the mover UNDER TEST). No search is
//! reimplemented here — this is the production pathfinder, priced by [`crate::cost`].

use crate::cost::build_cost_matrix;
use screeps::{Position, RoomName};
use screeps_rover::{LocalPathfinder, PathfindingProvider};
use screeps_sim_core::constants::{FATIGUE_RATE_PLAIN, FATIGUE_RATE_SWAMP};
use screeps_sim_core::SimTerrain;

/// The pathfinder op ceiling (matches rover's `MAX_PATHFIND_OPS`).
pub const MAX_OPS: u32 = 20_000;

/// The route rover's `LocalPathfinder` finds from `from` to within `range` of `goal` over `terrain`,
/// searching the fatigue cost field. Returns the waypoints (rover's `path`, which excludes `from`)
/// and whether the search was `incomplete` (hit the op cap / unreachable → best-effort).
pub fn rover_path(terrain: &SimTerrain, from: Position, goal: Position, range: u32) -> (Vec<Position>, bool) {
    let mut room_cb = |r: RoomName| build_cost_matrix(terrain, r);
    let mut pf = LocalPathfinder;
    let result = pf.search(
        from,
        goal,
        range,
        &mut room_cb,
        MAX_OPS,
        FATIGUE_RATE_PLAIN as u8, // plain-tile cost (matrix-0 fallback)
        FATIGUE_RATE_SWAMP as u8, // ignored by the offline pathfinder; swamp is baked into the matrix
    );
    (result.path, result.incomplete)
}
