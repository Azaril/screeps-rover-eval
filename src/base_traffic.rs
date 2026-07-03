//! ADR 0033 §D6 — **realistic base-traffic scenarios** (the C-family substrate): real mmo:shard3
//! room terrain + a FOREMAN-planned base + the traffic pattern a real economy drives (haulers
//! cycling source containers → storage), replacing the too-simple synthetic rooms (operator
//! directive). §D2's planned REUSE of combat-eval's terrain/base machinery: the room fixtures and
//! the verified column-major terrain decoder come from `screeps_combat_eval::harness::
//! terrain_import`; the base layouts are planned by the same foreman planner combat-eval's
//! `capture_base` runs — but combat-eval's `CapturedBase` deliberately DROPS roads / containers /
//! storage (its `combat_kind` models combat, not logistics), which are exactly the structures a
//! MOVEMENT benchmark needs. So rover-eval carries its own committed cache with the FULL structure
//! set: `resources/captured-layouts.json`, produced once by the `capture_layout` bin (the planner
//! is SLOW — same committed-cache pattern as ADR 0025 §12 Stage 3a; tests never plan).
//!
//! What the realism buys (why synthetic rooms under-measured): real wall geometry (chokepoints the
//! resolver must thread), planned ROADS at fatigue 1 (ADR 0009c guarantees a hub-connected road
//! network, so the fatigue-optimal routes the oracle prices actually follow the base's roads), and
//! a movement-BLOCKING base core (spawn/extensions/storage/towers force real detours through the
//! stamp). [`base_scenario`] realizes a layout into a [`SimTerrain`]; [`energy_traffic_fleet`]
//! derives the source-container → storage hauler cycles the [`crate::haul`] benchmark runs.

use crate::haul::HaulAssignment;
use screeps::{Part, RoomName};
use screeps_combat_eval::harness::terrain_import::decode_terrain;
use screeps_sim_core::{SimBody, SimTerrain};

/// One planned structure from the foreman plan — ALL kinds kept (unlike combat-eval's
/// `CapturedStructure`), because roads/containers/storage are the logistics substrate. `kind` is
/// the lowercased `StructureType` Debug name (`"road"`, `"container"`, `"storage"`, `"spawn"`, …).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PlannedStructure {
    pub kind: String,
    pub x: u8,
    pub y: u8,
    /// The foreman plan's RCL schedule for this placement (`RoomItem::required_rcl`,
    /// screeps-foreman/src/plan.rs:27) — the level at which the structure exists. `None` only for
    /// items the planner left unresolved (foreman finalization defaults those to 1).
    /// `#[serde(default)]` so the pre-extension committed cache still parses (ADR 0040 M1: the
    /// economy sim realizes layouts "as of RCL R" from this field).
    #[serde(default)]
    pub required_rcl: Option<u8>,
}

/// A captured full base layout: real terrain (2500-char column-major encoding, ADR 0025a) + the
/// fixture's object coords (sources drive the traffic endpoints) + the foreman plan's complete
/// structure set. Serializable — the committed cache is an array of these.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CapturedLayout {
    pub room: String,
    pub terrain: String,
    pub controller: (u8, u8),
    pub sources: Vec<(u8, u8)>,
    pub mineral: Option<(u8, u8)>,
    pub structures: Vec<PlannedStructure>,
}

/// The committed cache shape (shared with the `capture_layout` bin, which reads/writes it
/// incrementally like combat-eval's `capture_base`).
#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct LayoutCache {
    pub layouts: Vec<CapturedLayout>,
}

/// The committed captured layouts (parsed from the embedded `resources/captured-layouts.json`).
/// Regenerate/extend with `cargo run --release -p screeps-rover-eval --bin capture_layout -- [N]`.
pub fn captured_layouts() -> Vec<CapturedLayout> {
    let cache: LayoutCache =
        serde_json::from_str(include_str!("../resources/captured-layouts.json"))
            .expect("embedded resources/captured-layouts.json parses");
    cache.layouts
}

/// Does a planned structure of `kind` block movement? Roads/containers are walkable, own ramparts
/// are walkable (rover-eval's whole fleet is "own"), and the extractor sits ON the mineral without
/// blocking. Everything else (spawn / extensions / storage / towers / links / labs / walls / …) is
/// an obstacle — movement-blocking is what matters for a movement benchmark.
fn blocks_movement(kind: &str) -> bool {
    !matches!(kind, "road" | "container" | "rampart" | "extractor")
}

/// Realize a captured layout into the benchmark terrain: the REAL walls/swamps, the base's planned
/// roads inserted into `terrain.roads` (fatigue 1 — the M3 physics that makes a planned base's
/// routes meaningfully cheaper), and every movement-blocking structure inserted into
/// `terrain.walls` (`SimTerrain` prices walkability, so a blocking structure IS a wall to the
/// mover, the pathfinder, and the oracle alike — all three stay consistent).
pub fn base_scenario(layout: &CapturedLayout) -> SimTerrain {
    let mut terrain = decode_terrain(&layout.terrain);
    for s in &layout.structures {
        if s.kind == "road" {
            terrain.roads.insert((s.x, s.y));
        } else if blocks_movement(&s.kind) {
            terrain.walls.insert((s.x, s.y));
        }
    }
    terrain
}

/// A balanced 1:1 hauler (8×CARRY + 8×MOVE, capacity 400): full speed loaded on plain AND road, so
/// any efficiency loss the benchmark reports is routing/contention, never body-bound fatigue.
fn balanced_hauler() -> SimBody {
    let mut parts = vec![Part::Carry; 8];
    parts.extend(std::iter::repeat_n(Part::Move, 8));
    SimBody::unboosted(&parts)
}
/// Cargo per loaded leg = the balanced hauler's full capacity (8×CARRY×50).
const HAUL_Q: u32 = 400;
/// Round trips per hauler — several, so per-assignment contention effects repeat and sample.
const HAUL_TRIPS: u32 = 3;
/// Haulers per source route (~2 per source is the live economy's steady-state staffing).
const HAULERS_PER_ROUTE: usize = 2;

/// The first walkable tile adjacent to `(x, y)` not already in `taken` (deterministic fixed-order
/// scan; interior only — never an exit tile). Places the second hauler of a route: it starts and
/// reloads one tile off the container, so fleet starts never stack (the sim resolves one creep per
/// tile) while both serve the same range-1 pickup point.
fn open_neighbor(terrain: &SimTerrain, x: u8, y: u8, taken: &[(u8, u8)]) -> Option<(u8, u8)> {
    const OFFSETS: [(i16, i16); 8] = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    OFFSETS.iter().find_map(|&(dx, dy)| {
        let (nx, ny) = (x as i16 + dx, y as i16 + dy);
        if !(1..=48).contains(&nx) || !(1..=48).contains(&ny) {
            return None;
        }
        let t = (nx as u8, ny as u8);
        (!terrain.is_wall(t.0, t.1) && !taken.contains(&t)).then_some(t)
    })
}

/// The traffic pattern that drives a real economy: haulers cycling **source-container → storage**.
/// Endpoints come from the plan itself — the sink is the planned storage (falling back to the
/// first spawn for a storage-less plan), and each source's pickup is the planned container nearest
/// it (Chebyshev ≤ 2 — the foreman harvest container is adjacent to its source; if a plan carries
/// no source containers, the 2 containers furthest from storage stand in). `terrain` must be the
/// [`base_scenario`] terrain (structure blocking decides neighbor walkability). Both endpoints may
/// sit on/next to blocked tiles — arrival is range 1 ([`crate::haul::run_haul_fleet`]), exactly a
/// real `withdraw`/`transfer`. ~2 haulers per route, balanced bodies, several trips each.
pub fn energy_traffic_fleet(layout: &CapturedLayout, terrain: &SimTerrain) -> Vec<HaulAssignment> {
    let room: RoomName = layout
        .room
        .parse()
        .expect("captured layout room name parses");
    let pos = |x: u8, y: u8| crate::traverse::pos_in(room, x, y);
    let of_kind = |k: &str| -> Vec<(u8, u8)> {
        layout
            .structures
            .iter()
            .filter(|s| s.kind == k)
            .map(|s| (s.x, s.y))
            .collect()
    };
    let chebyshev =
        |a: (u8, u8), b: (u8, u8)| (a.0.abs_diff(b.0) as u32).max(a.1.abs_diff(b.1) as u32);

    // The sink: the planned storage (the hub the road network is guaranteed to connect, ADR 0009c);
    // a storage-less plan falls back to its first spawn (same hub stamp).
    let Some(&sink) = of_kind("storage").first().or(of_kind("spawn").first()) else {
        return Vec::new(); // no hub ⇒ no economy to model (a degenerate capture; caller asserts)
    };

    // Pickup endpoints: per source, its harvest container (nearest planned container, Chebyshev ≤ 2),
    // deduped — two sources sharing one container is one route staffed once.
    let containers = of_kind("container");
    let mut endpoints: Vec<(u8, u8)> = Vec::new();
    for &src in &layout.sources {
        if let Some(&c) = containers
            .iter()
            .min_by_key(|&&c| (chebyshev(c, src), c.1, c.0))
        {
            if chebyshev(c, src) <= 2 && !endpoints.contains(&c) {
                endpoints.push(c);
            }
        }
    }
    if endpoints.is_empty() {
        // Fallback (per the C-family spec): the 2 containers FURTHEST from storage — the longest
        // planned logistics legs, the next-most-plausible economy endpoints.
        let mut far = containers.clone();
        far.sort_by_key(|&c| (std::cmp::Reverse(chebyshev(c, sink)), c.1, c.0));
        endpoints = far.into_iter().take(2).collect();
    }

    // ~2 haulers per route: the first starts ON the container, the second one tile off it (both
    // reload at range 1 of the same pickup). Starts are kept globally distinct — the sim world
    // spawns each hauler at its `source`, and tiles hold one creep. The staffing is deliberately
    // STATIC (never adapted to whether rover can currently sustain the route): the corpus must
    // stay identical across rover versions for `H` to be comparable. Two captured rooms are known
    // to STARVE a double-staffed pocket route under rover's current resolver (head-on-in-a-tube +
    // parked-mate seal + repath flapping — the tests' `KNOWN_STARVED` ledger has the full story);
    // they stay in the corpus as real-layout repros for that investigation.
    let mut taken: Vec<(u8, u8)> = endpoints.clone();
    let mut fleet = Vec::new();
    for &ep in &endpoints {
        let mut starts = vec![ep];
        for _ in 1..HAULERS_PER_ROUTE {
            if let Some(n) = open_neighbor(terrain, ep.0, ep.1, &taken) {
                taken.push(n);
                starts.push(n);
            }
        }
        for s in starts {
            fleet.push(HaulAssignment {
                body: balanced_hauler(),
                q: HAUL_Q,
                source: pos(s.0, s.1),
                sink: pos(sink.0, sink.1),
                trips: HAUL_TRIPS,
            });
        }
    }
    fleet
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haul::{run_haul_fleet, t_star_rtt};
    use crate::stats::Summary;

    /// Generous fleet tick cap: oracle round trips on a 50×50 room are ~O(100) ticks; 3 trips ×
    /// heavy contention still fits with an order of magnitude to spare. Not a performance gate —
    /// completion inside it just proves no livelock.
    const TICK_CAP: u32 = 6000;

    /// (a) WELL-POSED over every captured layout: the realized terrain carries the planned roads
    /// (a foreman plan always has a road network — ADR 0009c), the economy derives a non-empty
    /// fleet with distinct walkable starts, and every assignment's oracle round trip is solvable
    /// over the real layout (`t_star_rtt` = Some) — mis-built scenarios fail HERE, so a failure in
    /// the fleet run below is attributable to rover, never to the scenario.
    #[test]
    fn captured_layouts_are_well_posed() {
        let layouts = captured_layouts();
        assert!(
            !layouts.is_empty(),
            "the committed captured-layouts cache is non-empty (run the capture_layout bin)"
        );
        for l in &layouts {
            let terrain = base_scenario(l);
            assert!(
                !terrain.roads.is_empty(),
                "{}: the planned road network is wired into terrain.roads",
                l.room
            );
            assert!(
                terrain.walls.len() > decode_terrain(&l.terrain).walls.len(),
                "{}: movement-blocking base structures are wired into terrain.walls",
                l.room
            );
            let fleet = energy_traffic_fleet(l, &terrain);
            assert!(
                fleet.len() >= HAULERS_PER_ROUTE,
                "{}: the economy derives at least one staffed route ({} haulers)",
                l.room,
                fleet.len()
            );
            let mut starts = std::collections::HashSet::new();
            for a in &fleet {
                assert!(
                    !terrain.is_wall(a.source.x().u8(), a.source.y().u8()),
                    "{}: hauler start {:?} is walkable",
                    l.room,
                    (a.source.x().u8(), a.source.y().u8())
                );
                assert!(
                    starts.insert((a.source.x().u8(), a.source.y().u8())),
                    "{}: hauler starts are distinct (no stacked spawns)",
                    l.room
                );
                assert!(
                    t_star_rtt(&terrain, a).is_some(),
                    "{}: oracle round trip {:?}→{:?} is solvable over the real layout",
                    l.room,
                    (a.source.x().u8(), a.source.y().u8()),
                    (a.sink.x().u8(), a.sink.y().u8())
                );
            }
        }
    }

    /// Rooms whose double-staffed pocket route starves under the current resolver — real-layout
    /// repros of the head-on/parked-intent pathology, kept in the corpus rather than constructed
    /// away. EMPTY since the fixes landed (the ledger below); the ratchet cuts both ways.
    /// HISTORY (kept as the ledger): `E13S29` + `E11N17` starved under the pre-tuning default
    /// (`reuse_path_length: 5`) — the route's two haulers met head-on each cycle in the 1-wide
    /// road corridor that is the container pocket's only cheap mouth; the outer hauler won every
    /// exchange, finished, and PARKED on the mouth tile, after which the starved inner hauler
    /// flapped forever between the optimistic short path and the stuck-escalation detour — never
    /// committing to the cheap detour that EXISTED (oracle cost 22 vs 6), never globally still
    /// for 12 ticks (deadlock detector silent), racking up linear `failed_into_parked` (~1000 vs
    /// the bounded 2–6 elsewhere). **HEALED 2026-07-01 by the tournament-tuned commitment default
    /// (`reuse_path_length: 20`, rover `DEFAULT_REUSE_PATH_LENGTH`)**: the inner hauler now
    /// commits to its detour and completes. The list is EMPTY on purpose — the ratchet below
    /// fails any room that regresses back into starvation.
    const KNOWN_STARVED: &[&str] = &[];

    /// (b)+(c) The FIRST REAL-LAYOUT BASELINE: run the haul fleet over every captured layout —
    /// every trip must complete within the generous cap with no deadlock (the hard well-posedness
    /// gates) on every room except the [`KNOWN_STARVED`] repros (ratcheted both ways: a healthy
    /// room that regresses fails, a starved room that heals fails until delisted), and the
    /// resulting `H` + CI + percentiles + failed-move classes are REPORTED via eprintln
    /// (regression-tracked numbers, deliberately NOT gated here: contention makes `H < 1`
    /// structural; the hard failed-intent gates live in the crowd/haul suites, and the pooled-H
    /// floor lives in the tuning full-corpus ratchet).
    #[test]
    fn real_layout_fleet_completes_and_reports_baseline() {
        let layouts = captured_layouts();
        assert!(!layouts.is_empty(), "captured-layouts cache is non-empty");
        let mut all_samples: Vec<(f64, f64)> = Vec::new();
        let (mut issued, mut failed, mut fatigued, mut wall, mut parked, mut coord) =
            (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
        for l in &layouts {
            let terrain = base_scenario(l);
            let fleet = energy_traffic_fleet(l, &terrain);
            let out = run_haul_fleet(&terrain, &fleet, TICK_CAP, 1)
                .expect("well-posed (oracle-solvable) — the (a) test isolates scenario bugs");
            assert!(!out.deadlocked, "{}: fleet deadlocked", l.room);
            if KNOWN_STARVED.contains(&l.room.as_str()) {
                // The starvation repro: real progress happens (the healthy haulers finish), but
                // the pocket route starves. A completed run here means the resolver fix landed —
                // delist the room and let the full gate below take over.
                assert!(
                    out.completed_trips > 0,
                    "{}: even the starved repro makes partial progress",
                    l.room
                );
                assert!(
                    out.completed_trips < out.expected_trips,
                    "{}: KNOWN_STARVED room now completes ({}/{}) — the resolver fix landed; \
                     remove it from KNOWN_STARVED",
                    l.room,
                    out.completed_trips,
                    out.expected_trips
                );
            } else {
                assert_eq!(
                    out.completed_trips, out.expected_trips,
                    "{}: all trips complete within the generous cap ({} ticks)",
                    l.room, TICK_CAP
                );
            }
            let s = &out.summary;
            let a = &out.audit;
            eprintln!(
                "[base-traffic {:7}] H={:.4} CI95=[{:.4},{:.4}] p05={:.3} p50={:.3} p95={:.3} | {} haulers, {}/{} trips in {} ticks | intents {}, failed {} (fatigued {}, wall {}, parked {}, coordination {})",
                l.room, s.weighted_mean, s.ci95.0, s.ci95.1, s.p05, s.p50, s.p95,
                fleet.len(), out.completed_trips, out.expected_trips, out.ticks,
                a.intents_issued, a.failed_moves, a.failed_fatigued, a.failed_wall,
                a.failed_into_parked, a.failed_coordination
            );
            all_samples.extend(out.samples.iter().copied());
            issued += a.intents_issued;
            failed += a.failed_moves;
            fatigued += a.failed_fatigued;
            wall += a.failed_wall;
            parked += a.failed_into_parked;
            coord += a.failed_coordination;
        }
        let agg = Summary::of(&all_samples, 1);
        eprintln!(
            "[base-traffic BASELINE over {} rooms, {} trips] H={:.4} CI95=[{:.4},{:.4}] p05={:.3} p50={:.3} p95={:.3} min={:.3} | intents {}, failed {} (fatigued {}, wall {}, parked {}, coordination {})",
            layouts.len(), agg.n, agg.weighted_mean, agg.ci95.0, agg.ci95.1,
            agg.p05, agg.p50, agg.p95, agg.min,
            issued, failed, fatigued, wall, parked, coord
        );
    }
}
