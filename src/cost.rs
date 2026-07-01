//! Terrain pricing for rover's pathfinder + the oracle. rover owns the search and the cost-matrix
//! machinery; rover-eval supplies only the per-tile terrain cost (pricing policy — ADR 0033 /
//! "no one-off pathfinding"). The cost field IS the engine fatigue field (road 1 / plain 2 /
//! swamp 10), so a fatigue-optimal route is exactly what rover *should* find.

use screeps::{LocalCostMatrix, RoomName};
use screeps_rover::{
    ConstructionSiteCostMatrixCache, CostMatrixCache, CostMatrixDataSource, CostMatrixOptions,
    CostMatrixSystem, CostMatrixWrite, CreepCostMatrixCache, LinearCostMatrix, StuctureCostMatrixCache,
};
use screeps_sim_core::constants::{FATIGUE_RATE_ROAD, FATIGUE_RATE_SWAMP};
use screeps_sim_core::SimTerrain;
use std::collections::{BTreeMap, HashSet};

/// The engine fatigue rate of a tile (road 1 / plain 2 / swamp 10). Delegates to the shared kernel
/// [`SimTerrain::fatigue_rate`] — the SAME field the sim mover accrues — so it is used unchanged as
/// BOTH rover's search cost and the oracle's edge cost, and `R_fatigue == 1` iff rover found a
/// fatigue-optimal path.
pub fn fatigue_rate(terrain: &SimTerrain, x: u8, y: u8) -> u32 {
    terrain.fatigue_rate(x, y)
}

/// A [`CostMatrixDataSource`] over a `SimTerrain`: walls impassable, swamps at their fatigue cost,
/// roads at cost 1 (overriding an underlying swamp). Owns its data so it is `'static` for rover's
/// `CostMatrixSystem`. Priced by terrain only — no creep costs, so multi-creep contention is left to
/// the `MovementSystem` resolver (what the Tier-B crowd sim measures), not baked into the field.
///
/// **Single-terrain by design**: it answers the SAME matrix for every room asked. It has no
/// `MovementState` to read per-room overrides from, and its consumers are the single-creep Tier-A
/// paths ([`crate::pathing`]/[`crate::metrics`]/[`build_cost_matrix`]) whose oracle is single-room
/// anyway. Multi-room worlds with distinct per-room terrain go through [`WorldCostSource`].
#[derive(Clone)]
pub struct TerrainCostSource {
    walls: Vec<(u8, u8)>,
    swamps: Vec<(u8, u8)>,
    roads: Vec<(u8, u8)>,
}

impl TerrainCostSource {
    /// Snapshot a `SimTerrain` into an owned, `'static` cost source for rover's `CostMatrixSystem`.
    pub fn new(terrain: &SimTerrain) -> Self {
        TerrainCostSource {
            walls: terrain.walls.iter().copied().collect(),
            swamps: terrain.swamps.iter().copied().collect(),
            roads: terrain.roads.iter().copied().collect(),
        }
    }
}

impl CostMatrixDataSource for TerrainCostSource {
    fn get_structure_costs(&self, _room: RoomName) -> Option<StuctureCostMatrixCache> {
        // rover applies `roads` first (at `road_cost` = 1) then overlays `other` via `apply_to`, which
        // OVERWRITES. So a road on a swamp must be omitted from `other`, else the swamp's 10 would clobber
        // the road's 1 — the reverse of the engine (`movement.js`: road wins). Exclude road tiles here.
        let road_set: HashSet<(u8, u8)> = self.roads.iter().copied().collect();
        let mut other = LinearCostMatrix::new();
        for &(x, y) in &self.swamps {
            if !road_set.contains(&(x, y)) {
                other.set(x, y, FATIGUE_RATE_SWAMP as u8);
            }
        }
        for &(x, y) in &self.walls {
            other.set(x, y, u8::MAX);
        }
        let mut roads = LinearCostMatrix::new();
        for &(x, y) in &self.roads {
            // The value is informational: rover's `apply_to_transformed` rewrites road tiles to
            // `CostMatrixOptions::road_cost` (default 1 = FATIGUE_RATE_ROAD) regardless of it.
            roads.set(x, y, FATIGUE_RATE_ROAD as u8);
        }
        Some(StuctureCostMatrixCache { roads, other })
    }
    fn get_construction_site_costs(&self, _room: RoomName) -> Option<ConstructionSiteCostMatrixCache> {
        None
    }
    fn get_creep_costs(&self, _room: RoomName) -> Option<CreepCostMatrixCache> {
        Some(CreepCostMatrixCache {
            friendly_creeps: LinearCostMatrix::new(),
            hostile_creeps: LinearCostMatrix::new(),
            source_keeper_agro: LinearCostMatrix::new(),
        })
    }
}

/// A [`CostMatrixDataSource`] over a `SimTerrain` PLUS current creep occupancy — the faithful
/// analogue of rover's live `ScreepsCostMatrixDataSource`, which marks every friendly creep's tile
/// `u8::MAX` in the `friendly_creeps` matrix. rover's default options path *through* friendlies
/// (`friendly_creeps: false`) and flip this matrix on only as **stuck-escalation** — so a parked
/// creep is routed around after a few blocked ticks, instead of never being priced at all. The
/// multi-creep drivers ([`crate::crowd`], [`crate::haul`]) MUST use this (rebuilt per tick);
/// omitting occupancy violates rover's usage contract and manufactures permanent livelock (the
/// failed-move sentinel's first catch, ADR 0033 §D5.4 note).
///
/// **Room-aware** (ADR 0033 M5 follow-up #4): per-room terrain overrides are snapshotted from
/// `MovementState::rooms` and answered per queried room — a room absent there prices as the
/// default `terrain`, mirroring [`screeps_sim_core::MovementState::terrain_for`] exactly — and
/// creep occupancy is keyed by each creep's ACTUAL room, so room B's matrix never carries room A's
/// walls or phantom A-roomed creep costs (the stuck-escalation cross-room detour distorter).
#[derive(Clone)]
pub struct WorldCostSource {
    /// The default/common terrain — what `terrain_for` falls back to for rooms with no override.
    terrain: TerrainCostSource,
    /// Per-room overrides (from `MovementState::rooms`). BTreeMap: lookup-keyed, and ordered by
    /// `RoomName` if it is ever iterated — no HashMap-iteration-order exposure (the fence).
    rooms: BTreeMap<RoomName, TerrainCostSource>,
    /// Every living creep's tile, keyed by the creep's actual room.
    friendly: Vec<(RoomName, u8, u8)>,
}

impl WorldCostSource {
    /// Snapshot the default terrain, `world.rooms`' per-room overrides, and every living creep's
    /// room + tile (all sim creeps are "friendly" — the whole fleet is rover-controlled). Callers
    /// pass the world's default terrain as `terrain` (the same object as `world.terrain`), so the
    /// absent-override fallback here IS `terrain_for`'s fallback.
    pub fn new(terrain: &SimTerrain, world: &screeps_sim_core::MovementState) -> Self {
        WorldCostSource {
            terrain: TerrainCostSource::new(terrain),
            rooms: world
                .rooms
                .iter()
                .map(|(&room, t)| (room, TerrainCostSource::new(t)))
                .collect(),
            friendly: world
                .living_creeps()
                .map(|c| (c.pos.room_name(), c.pos.x().u8(), c.pos.y().u8()))
                .collect(),
        }
    }

    /// The terrain source answering for `room` — the override if one exists, else the default
    /// (`MovementState::terrain_for` verbatim).
    fn terrain_for(&self, room: RoomName) -> &TerrainCostSource {
        self.rooms.get(&room).unwrap_or(&self.terrain)
    }
}

impl CostMatrixDataSource for WorldCostSource {
    fn get_structure_costs(&self, room: RoomName) -> Option<StuctureCostMatrixCache> {
        self.terrain_for(room).get_structure_costs(room)
    }
    fn get_construction_site_costs(&self, room: RoomName) -> Option<ConstructionSiteCostMatrixCache> {
        self.terrain_for(room).get_construction_site_costs(room)
    }
    fn get_creep_costs(&self, room: RoomName) -> Option<CreepCostMatrixCache> {
        let mut friendly_creeps = LinearCostMatrix::new();
        for &(creep_room, x, y) in &self.friendly {
            if creep_room == room {
                friendly_creeps.set(x, y, u8::MAX); // screeps_impl.rs:252 verbatim
            }
        }
        Some(CreepCostMatrixCache {
            friendly_creeps,
            hostile_creeps: LinearCostMatrix::new(),
            source_keeper_agro: LinearCostMatrix::new(),
        })
    }
}

/// Build the `LocalCostMatrix` rover's pathfinder reads for `room` — via rover's own
/// `CostMatrixSystem`, so rover-eval never hand-rolls a matrix beyond the pricing above. The default
/// `CostMatrixOptions` price roads at 1 (`road_cost`), matching `FATIGUE_RATE_ROAD`.
pub fn build_cost_matrix(terrain: &SimTerrain, room: RoomName) -> Option<LocalCostMatrix> {
    let source = TerrainCostSource::new(terrain);
    let mut cache = CostMatrixCache::default();
    let mut system = CostMatrixSystem::new(&mut cache, Box::new(source));
    system
        .build_local_cost_matrix(room, &CostMatrixOptions::default())
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use screeps::{Part, Position, RoomCoordinate, RoomXY};
    use screeps_sim_core::{MovementState, SimBody, SimCreep};

    fn pos_in(room: RoomName, x: u8, y: u8) -> Position {
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
    }
    fn xy(x: u8, y: u8) -> RoomXY {
        RoomXY::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap())
    }
    /// The final matrix rover's pathfinder would read for `room`, with creep costs ON (the
    /// stuck-escalation view — exactly where phantom cross-room costs used to distort detours).
    fn matrix_for(source: &WorldCostSource, room: RoomName) -> LocalCostMatrix {
        let mut cache = CostMatrixCache::default();
        let mut system = CostMatrixSystem::new(&mut cache, Box::new(source.clone()));
        let options = CostMatrixOptions { friendly_creeps: true, ..Default::default() };
        system.build_local_cost_matrix(room, &options).expect("matrix builds")
    }

    /// The room-awareness contract: two rooms with DIFFERENT walls + a creep in each. Room B's
    /// matrix must contain B's wall and B's creep, and neither A's wall nor A's creep (and vice
    /// versa) — `terrain_for`-mirroring per-room pricing, no phantom cross-room costs.
    #[test]
    fn per_room_matrices_never_leak_the_other_rooms_walls_or_creeps() {
        let room_a: RoomName = "W1N1".parse().unwrap();
        let room_b: RoomName = "W2N1".parse().unwrap();

        // Default terrain (= room A's, no override): a wall at (15,25).
        let mut terrain_a = SimTerrain::default();
        terrain_a.walls.insert((15, 25));
        // Room B's OVERRIDE: a wall at (30,30) instead.
        let mut terrain_b = SimTerrain::default();
        terrain_b.walls.insert((30, 30));

        let mut world = MovementState {
            terrain: terrain_a.clone(),
            creeps: vec![
                SimCreep {
                    id: 1,
                    owner: 0,
                    pos: pos_in(room_a, 10, 10),
                    body: SimBody::unboosted(&[Part::Move]),
                    fatigue: 0,
                    carry_used: 0,
                },
                SimCreep {
                    id: 2,
                    owner: 0,
                    pos: pos_in(room_b, 40, 40),
                    body: SimBody::unboosted(&[Part::Move]),
                    fatigue: 0,
                    carry_used: 0,
                },
            ],
            ..Default::default()
        };
        world.rooms.insert(room_b, terrain_b);

        let source = WorldCostSource::new(&terrain_a, &world);
        let a = matrix_for(&source, room_a);
        let b = matrix_for(&source, room_b);

        // Room A: its own wall + its own creep, nothing of B's.
        assert_eq!(a.get(xy(15, 25)), u8::MAX, "A's wall prices impassable in A");
        assert_eq!(a.get(xy(10, 10)), u8::MAX, "A's creep prices impassable in A");
        assert_ne!(a.get(xy(30, 30)), u8::MAX, "B's wall must NOT leak into A");
        assert_ne!(a.get(xy(40, 40)), u8::MAX, "B's creep must NOT leak into A");

        // Room B: the override + the B-roomed creep, nothing of A's.
        assert_eq!(b.get(xy(30, 30)), u8::MAX, "B's override wall prices impassable in B");
        assert_eq!(b.get(xy(40, 40)), u8::MAX, "B's creep prices impassable in B");
        assert_ne!(b.get(xy(15, 25)), u8::MAX, "A's (default-terrain) wall must NOT leak into B");
        assert_ne!(b.get(xy(10, 10)), u8::MAX, "A's creep must NOT leak into B");

        // A room with NO override and NO creeps prices as the default terrain — `terrain_for`'s
        // fallback semantics exactly.
        let room_c: RoomName = "W3N1".parse().unwrap();
        let c = matrix_for(&source, room_c);
        assert_eq!(c.get(xy(15, 25)), u8::MAX, "an unoverridden room inherits the default terrain");
        assert_ne!(c.get(xy(30, 30)), u8::MAX);
        assert_ne!(c.get(xy(10, 10)), u8::MAX, "creep occupancy is room-keyed, never default-terrain-keyed");
    }
}
