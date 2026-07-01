//! Walk a single creep along a fixed route through the shared kernel, tick by tick, so travel-TIME
//! metrics can see fatigue stalls (ADR 0033 §D5). Unlike [`crate::metrics::r_fatigue`] — a static
//! path-cost comparison — this drives `resolve_movement` and counts the ticks the creep actually
//! takes, including the ticks it sits fatigued. It reuses the kernel mover (no re-ported physics)
//! and follows a *precomputed* route (rover's or the oracle's); per-tick re-pathing is the Tier-B
//! `MovementSystem` driver (M1/M4), not this.

use screeps::{Direction, Position, RoomCoordinate, RoomName};
use screeps_sim_core::{MoveIntents, MovementState, SimBody, SimCreep, resolve_movement};

/// The outcome of walking a creep along a route.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Traversal {
    /// Total ticks elapsed until the creep reached the route's end (or hit the cap).
    pub ticks: u32,
    /// Ticks on which the creep stepped to the next waypoint.
    pub moves: u32,
    /// Ticks the creep could not move because it was fatigued at tick start (body-forced idle).
    pub idle_fatigued: u32,
    /// Ticks the creep was idle with *zero* fatigue — i.e. a move it *could* have made didn't happen
    /// (a wall/edge/rejection on the route). For a lone creep on a valid route this is 0; it is the
    /// rover-attributable idle that [`crate::metrics::fatigue_util`] flags.
    pub idle_free: u32,
    /// Whether the creep reached the end of the route within the tick cap.
    pub reached: bool,
}

/// Build a same-room position (routes here are single-room, Tier A).
pub fn pos_in(room: RoomName, x: u8, y: u8) -> Position {
    Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
}

/// The `Direction` from `a` to an ADJACENT tile `b` (Chebyshev range 1), or `None` if not adjacent /
/// not the same room. Route waypoints are consecutive adjacent tiles, so this is always `Some`.
fn direction_between(a: Position, b: Position) -> Option<Direction> {
    if a.room_name() != b.room_name() {
        return None;
    }
    let dx = b.x().u8() as i32 - a.x().u8() as i32;
    let dy = b.y().u8() as i32 - a.y().u8() as i32;
    Some(match (dx, dy) {
        (0, -1) => Direction::Top,
        (1, -1) => Direction::TopRight,
        (1, 0) => Direction::Right,
        (1, 1) => Direction::BottomRight,
        (0, 1) => Direction::Bottom,
        (-1, 1) => Direction::BottomLeft,
        (-1, 0) => Direction::Left,
        (-1, -1) => Direction::TopLeft,
        _ => return None,
    })
}

/// Walk a creep (`body` + `carry_used` resources aboard) from `from` along `route` (the waypoints
/// after the start, each adjacent to the last), driving the kernel one tick at a time. Every tick the
/// creep is issued a move toward its next unreached waypoint; the kernel enforces fatigue eligibility,
/// so a fatigued creep simply waits. Stops when the route end is reached or `tick_cap` ticks elapse.
pub fn traverse(
    terrain: &screeps_sim_core::SimTerrain,
    body: SimBody,
    carry_used: u32,
    from: Position,
    route: &[Position],
    tick_cap: u32,
) -> Traversal {
    let mut world = MovementState {
        terrain: terrain.clone(),
        creeps: vec![SimCreep { id: 1, owner: 0, pos: from, body, fatigue: 0, carry_used }],
        ..Default::default()
    };
    let mut t = Traversal { reached: route.is_empty(), ..Default::default() };
    let mut i = 0usize;
    while i < route.len() && t.ticks < tick_cap {
        let before = world.creeps[0].pos;
        let fatigued = world.creeps[0].fatigue > 0;
        let mut intents = MoveIntents::new();
        if let Some(dir) = direction_between(before, route[i]) {
            intents.set_move(1, dir);
        }
        resolve_movement(&mut world, &intents);
        t.ticks += 1;
        let after = world.creeps[0].pos;
        if after != before {
            t.moves += 1;
            if after == route[i] {
                i += 1;
            }
        } else if fatigued {
            t.idle_fatigued += 1;
        } else {
            t.idle_free += 1;
        }
    }
    t.reached = i >= route.len();
    t
}
