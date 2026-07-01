//! Terrain pricing for rover's pathfinder + the oracle. rover owns the search and the cost-matrix
//! machinery; rover-eval supplies only the per-tile terrain cost (pricing policy — ADR 0033 /
//! "no one-off pathfinding"). The cost field IS the engine fatigue field (road 1 / plain 2 /
//! swamp 10), so a fatigue-optimal route is exactly what rover *should* find.

use screeps::{LocalCostMatrix, RoomName};
use screeps_rover::{
    ConstructionSiteCostMatrixCache, CostMatrixCache, CostMatrixDataSource, CostMatrixOptions,
    CostMatrixSystem, CostMatrixWrite, CreepCostMatrixCache, LinearCostMatrix, StuctureCostMatrixCache,
};
use screeps_sim_core::constants::{FATIGUE_RATE_PLAIN, FATIGUE_RATE_SWAMP};
use screeps_sim_core::SimTerrain;

/// The engine fatigue rate of a tile (plain 2 / swamp 10; roads arrive in M3). Used as BOTH rover's
/// search cost and the oracle's edge cost, so `R_fatigue == 1` iff rover found a fatigue-optimal path.
pub fn fatigue_rate(terrain: &SimTerrain, x: u8, y: u8) -> u32 {
    if terrain.swamps.contains(&(x, y)) {
        FATIGUE_RATE_SWAMP
    } else {
        FATIGUE_RATE_PLAIN
    }
}

/// A [`CostMatrixDataSource`] over a `SimTerrain`: walls impassable, swamps at their fatigue cost,
/// roads/creeps empty. Owns its data so it is `'static` for rover's `CostMatrixSystem`.
struct TerrainCostSource {
    walls: Vec<(u8, u8)>,
    swamps: Vec<(u8, u8)>,
}

impl CostMatrixDataSource for TerrainCostSource {
    fn get_structure_costs(&self, _room: RoomName) -> Option<StuctureCostMatrixCache> {
        let mut other = LinearCostMatrix::new();
        for &(x, y) in &self.swamps {
            other.set(x, y, FATIGUE_RATE_SWAMP as u8);
        }
        for &(x, y) in &self.walls {
            other.set(x, y, u8::MAX);
        }
        Some(StuctureCostMatrixCache {
            roads: LinearCostMatrix::new(),
            other,
        })
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

/// Build the `LocalCostMatrix` rover's pathfinder reads for `room` — via rover's own
/// `CostMatrixSystem`, so rover-eval never hand-rolls a matrix beyond the pricing above.
pub fn build_cost_matrix(terrain: &SimTerrain, room: RoomName) -> Option<LocalCostMatrix> {
    let source = TerrainCostSource {
        walls: terrain.walls.iter().copied().collect(),
        swamps: terrain.swamps.iter().copied().collect(),
    };
    let mut cache = CostMatrixCache::default();
    let mut system = CostMatrixSystem::new(&mut cache, Box::new(source));
    system
        .build_local_cost_matrix(room, &CostMatrixOptions::default())
        .ok()
}
