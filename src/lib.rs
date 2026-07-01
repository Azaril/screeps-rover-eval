//! # screeps-rover-eval
//!
//! Offline simulator + benchmark for [`screeps-rover`](https://github.com/Azaril/screeps-rover)
//! (ADR 0033). It drives the **real** rover pathfinder / mover over a
//! [`screeps_sim_core`] movement world and measures pathing quality — route optimality,
//! fatigue efficiency, congestion — plus algorithmic/ops CPU, against a ground-truth oracle. It is
//! the "validated separately" harness the combat sim (ADR 0023) fences off: the combat sim *runs*
//! rover but never *measures* it.
//!
//! Host-only; it reuses the `screeps-sim-core` kernel (no re-ported physics) and rover's own
//! `LocalPathfinder` + `room_grid_dijkstra` (no one-off search algorithms — pricing policy only).

pub mod base_traffic;
pub mod cost;
pub mod crowd;
pub mod haul;
pub mod metrics;
pub mod oracle;
pub mod pathing;
pub mod scenario;
pub mod stats;
pub mod traverse;
pub mod tuning;
pub mod value;

#[cfg(test)]
mod kernel_reuse_smoke {
    //! Proves the reuse wiring end-to-end: rover-eval drives the kernel's `MovementSim` over a
    //! `MovementState`, and a creep moves + accrues fatigue exactly as the shared mover dictates.
    use screeps::{Direction, Part, Position, RoomCoordinate, RoomName};
    use screeps_sim_core::{
        BodyPartDef, MoveIntents, MovementSim, MovementState, SimBody, SimCreep, Simulation,
    };

    fn pos(x: u8, y: u8) -> Position {
        let room: RoomName = "W1N1".parse().unwrap();
        Position::new(
            RoomCoordinate::new(x).unwrap(),
            RoomCoordinate::new(y).unwrap(),
            room,
        )
    }

    #[test]
    fn creep_steps_and_accrues_fatigue_via_the_shared_mover() {
        // An under-MOVE body (2 ATTACK + 1 MOVE): fatigue_weight = 2, plain rate = 2 → +4 on a step,
        // regen = 2×1 MOVE = 2, so net fatigue = 2 after one plain step.
        let body = SimBody::new(vec![
            BodyPartDef::new(Part::Attack),
            BodyPartDef::new(Part::Attack),
            BodyPartDef::new(Part::Move),
        ]);
        let mut world = MovementState {
            creeps: vec![SimCreep {
                id: 1,
                owner: 0,
                pos: pos(25, 25),
                body,
                fatigue: 0,
                carry_used: 0,
            }],
            ..Default::default()
        };

        let mut intents = MoveIntents::new();
        intents.set_move(1, Direction::Right);

        let report = MovementSim::step(&mut world, &intents);

        assert_eq!(world.creeps[0].pos, pos(26, 25), "the shared mover stepped the creep right");
        assert_eq!(world.creeps[0].fatigue, 2, "engine fatigue: +4 accrued − 2 regen = 2");
        assert_eq!(report.tick, 0, "report tick is captured pre-increment");
        assert_eq!(world.tick, 1, "the mover advanced the clock");
    }
}
