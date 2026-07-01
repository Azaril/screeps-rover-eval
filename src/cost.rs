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
use std::collections::HashSet;

/// The engine fatigue rate of a tile (road 1 / plain 2 / swamp 10). Delegates to the shared kernel
/// [`SimTerrain::fatigue_rate`] — the SAME field the sim mover accrues — so it is used unchanged as
/// BOTH rover's search cost and the oracle's edge cost, and `R_fatigue == 1` iff rover found a
/// fatigue-optimal path.
pub fn fatigue_rate(terrain: &SimTerrain, x: u8, y: u8) -> u32 {
    terrain.fatigue_rate(x, y)
}

/// A [`CostMatrixDataSource`] over a `SimTerrain`: walls impassable, swamps at their fatigue cost,
/// roads at cost 1 (overriding an underlying swamp). Owns its data so it is `'static` for rover's
/// `CostMatrixSystem`.
struct TerrainCostSource {
    walls: Vec<(u8, u8)>,
    swamps: Vec<(u8, u8)>,
    roads: Vec<(u8, u8)>,
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

/// Build the `LocalCostMatrix` rover's pathfinder reads for `room` — via rover's own
/// `CostMatrixSystem`, so rover-eval never hand-rolls a matrix beyond the pricing above. The default
/// `CostMatrixOptions` price roads at 1 (`road_cost`), matching `FATIGUE_RATE_ROAD`.
pub fn build_cost_matrix(terrain: &SimTerrain, room: RoomName) -> Option<LocalCostMatrix> {
    let source = TerrainCostSource {
        walls: terrain.walls.iter().copied().collect(),
        swamps: terrain.swamps.iter().copied().collect(),
        roads: terrain.roads.iter().copied().collect(),
    };
    let mut cache = CostMatrixCache::default();
    let mut system = CostMatrixSystem::new(&mut cache, Box::new(source));
    system
        .build_local_cost_matrix(room, &CostMatrixOptions::default())
        .ok()
}
