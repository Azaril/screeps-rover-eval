//! ADR 0033 §D6 — the **offline full-layout base-capture tool**. Runs the foreman room planner on
//! combat-eval's committed real-terrain fixtures (ADR 0025 §12 Stage 1 — §D2 reuse) and writes the
//! COMPLETE structure placements — roads / containers / storage included — to
//! `resources/captured-layouts.json`, the cache `base_traffic` loads. Combat-eval's own
//! `capture_base` exists but its `CapturedBase` deliberately drops everything the combat sim
//! doesn't model (roads/containers/storage — exactly the logistics substrate a MOVEMENT benchmark
//! needs), and combat-eval is visibility-frozen for this work, so rover-eval mirrors the ~30-line
//! `PlannerRoomDataSource` shim instead of restructuring it. SLOW (the planner escalates anchor
//! beams, ~4–60s/room — see the foreman-bench perf notes) — run ONCE, manually, never in CI:
//!
//!   cargo run --release -p screeps-rover-eval --bin capture_layout -- [N]
//!
//! `N` (optional) caps how many fixtures to plan (default: all). Existing cached rooms are kept
//! and only missing ones are planned, so re-runs are incremental (persisted after each room).

use screeps_combat_eval::harness::terrain_import::{decode_fast, fixtures, TerrainFixture};
use screeps_common::plan_location::PlanLocation;
use screeps_foreman::room_data::PlannerRoomDataSource;
use screeps_foreman::terrain::FastRoomTerrain;
use screeps_rover_eval::base_traffic::{CapturedLayout, LayoutCache, PlannedStructure};
use std::time::Instant;

/// A `PlannerRoomDataSource` over a real-terrain fixture (mirror of combat-eval's private
/// `FixtureDataSource` — the fixture's object coords are already snapped to clear tiles, the
/// Stage-1 coordinate fix, so the planner gets valid terrain-aligned inputs).
struct FixtureSource {
    terrain: FastRoomTerrain,
    controllers: Vec<PlanLocation>,
    sources: Vec<PlanLocation>,
    minerals: Vec<PlanLocation>,
}

impl PlannerRoomDataSource for FixtureSource {
    fn get_terrain(&self) -> &FastRoomTerrain {
        &self.terrain
    }
    fn get_controllers(&self) -> &[PlanLocation] {
        &self.controllers
    }
    fn get_sources(&self) -> &[PlanLocation] {
        &self.sources
    }
    fn get_minerals(&self) -> &[PlanLocation] {
        &self.minerals
    }
}

/// Plan `fixture` and keep EVERY placed structure (kind = lowercased `StructureType` Debug name).
/// Output is sorted (kind, x, y) so the committed JSON is stable across planner HashMap iteration.
fn capture(fixture: &TerrainFixture) -> Result<CapturedLayout, String> {
    let source = FixtureSource {
        terrain: decode_fast(&fixture.terrain),
        controllers: vec![PlanLocation::new(
            fixture.controller.0 as i8,
            fixture.controller.1 as i8,
        )],
        sources: fixture
            .sources
            .iter()
            .map(|&(x, y)| PlanLocation::new(x as i8, y as i8))
            .collect(),
        minerals: fixture
            .mineral
            .into_iter()
            .map(|(x, y)| PlanLocation::new(x as i8, y as i8))
            .collect(),
    };
    let plan = screeps_foreman::planner::plan_room(&source)
        .map_err(|e| format!("planning {} failed: {e}", fixture.room))?;
    let mut structures: Vec<PlannedStructure> = plan
        .structures
        .iter()
        .flat_map(|(location, items)| {
            // Each RoomItem becomes its own PlannedStructure, so a Location carrying multiple
            // RoomItems (e.g. rampart-over-structure) keeps each kind paired with ITS OWN
            // `required_rcl` (screeps-foreman/src/plan.rs:27) — the ADR 0040 M1 capture extension.
            items.iter().map(|item| PlannedStructure {
                kind: format!("{:?}", item.structure_type()).to_ascii_lowercase(),
                x: location.x(),
                y: location.y(),
                required_rcl: item.required_rcl_opt(),
            })
        })
        .collect();
    structures.sort_by(|a, b| {
        (&a.kind, a.x, a.y, a.required_rcl).cmp(&(&b.kind, b.x, b.y, b.required_rcl))
    });
    Ok(CapturedLayout {
        room: fixture.room.clone(),
        terrain: fixture.terrain.clone(),
        controller: fixture.controller,
        sources: fixture.sources.clone(),
        mineral: fixture.mineral,
        structures,
    })
}

fn main() {
    let limit: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(usize::MAX);
    // Anchor to the crate dir so the cache lands in resources/ regardless of CWD.
    let out_path = concat!(env!("CARGO_MANIFEST_DIR"), "/resources/captured-layouts.json");

    // Incremental: keep already-cached rooms, only plan the rest.
    let mut cache: LayoutCache = std::fs::read_to_string(out_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let have: std::collections::HashSet<String> =
        cache.layouts.iter().map(|l| l.room.clone()).collect();

    let fx = fixtures();
    let todo: Vec<_> = fx
        .iter()
        .filter(|f| !have.contains(&f.room))
        .take(limit)
        .collect();
    println!(
        "capture_layout: {} fixtures, {} already cached, planning {} (foreman is SLOW)…",
        fx.len(),
        have.len(),
        todo.len()
    );

    for (i, fixture) in todo.iter().enumerate() {
        let t = Instant::now();
        match capture(fixture) {
            Ok(layout) => {
                let n = |k: &str| layout.structures.iter().filter(|s| s.kind == k).count();
                println!(
                    "  [{}/{}] {} planned in {:.1}s — {} road, {} container, {} storage, {} spawn, {} extension, {} total",
                    i + 1,
                    todo.len(),
                    layout.room,
                    t.elapsed().as_secs_f64(),
                    n("road"),
                    n("container"),
                    n("storage"),
                    n("spawn"),
                    n("extension"),
                    layout.structures.len()
                );
                cache.layouts.push(layout);
                // Persist after each room so a long run is resumable / partial results survive.
                std::fs::write(
                    out_path,
                    serde_json::to_string(&cache).expect("serialize cache"),
                )
                .expect("write cache");
            }
            Err(e) => println!("  [{}/{}] {} SKIPPED: {e}", i + 1, todo.len(), fixture.room),
        }
    }
    println!(
        "capture_layout: done — {} layouts cached in {out_path}",
        cache.layouts.len()
    );
}
