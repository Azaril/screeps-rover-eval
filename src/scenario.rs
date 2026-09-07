//! A scenario catalog + procedural generator + validator for the rover pathing bench (ADR 0033 §D4).
//! This is the systematic corpus that scales the bench past ad-hoc `#[test]` fixtures: each
//! [`Scenario`] is a terrain + a single creep walking `from` → within `range` of `goal`, plus the
//! metric gates it must satisfy. [`validate`] runs the oracle (is the goal reachable?) and the
//! metrics, and reports pass/fail. [`catalog`] is the curated A/B-family set; [`generate`] draws a
//! seeded procedural room (reusing the kernel [`Rng`](screeps_sim_core::rng::Rng), no ambient entropy).
//!
//! Reuses the combat-eval Generator/Validator *shape* (ADR 0023a) without the combat coupling — a
//! movement benchmark needs terrain + a body + a route, not opponents.

use crate::metrics::{fatigue_util, movement_eff, r_fatigue, r_ticks, rover_traversal};
use crate::oracle::optimal_path;
use crate::traverse::pos_in;
use screeps::{Part, Position, RoomName};
use screeps_sim_core::rng::Rng;
use screeps_sim_core::{SimBody, SimTerrain};

fn room() -> RoomName {
    "W1N1".parse().unwrap()
}

/// A creep spec for a scenario: its body parts and the resources it carries (loaded-CARRY fatigue).
#[derive(Clone)]
pub struct CreepSpec {
    pub parts: Vec<Part>,
    pub carry_used: u32,
}

impl CreepSpec {
    /// A full-health unboosted body from this spec.
    pub fn body(&self) -> SimBody {
        SimBody::unboosted(&self.parts)
    }
}

/// One pathing scenario + the gates rover must satisfy on it.
pub struct Scenario {
    pub name: &'static str,
    pub terrain: SimTerrain,
    pub spec: CreepSpec,
    pub from: Position,
    pub goal: Position,
    pub range: u32,
    /// Upper bound on `R_fatigue` (route optimality). rover's chosen route's fatigue cost must be
    /// within this factor of the fatigue-optimal cost.
    pub max_r_fatigue: f64,
    /// Upper bound on `R_ticks` (travel-time optimality) for this scenario's body.
    pub max_r_ticks: f64,
}

/// The measured outcome of validating a [`Scenario`].
#[derive(Clone, Debug)]
pub struct ScenarioResult {
    pub name: &'static str,
    /// The oracle found a route to the goal (the scenario is well-posed).
    pub solvable: bool,
    pub r_fatigue: Option<f64>,
    pub r_ticks: Option<f64>,
    pub movement_eff: f64,
    pub fatigue_util: f64,
    /// All gates satisfied.
    pub passed: bool,
    /// One message per violated gate (empty iff `passed`).
    pub failures: Vec<String>,
}

/// Run the oracle + the metrics over `s` and check its gates. A lone creep on a valid route must
/// never idle with a move available, so `fatigue_util == 1` is asserted for every scenario (the
/// "rover blameless" invariant); `R_fatigue`/`R_ticks` are checked against the scenario's bounds.
pub fn validate(s: &Scenario) -> ScenarioResult {
    let body = s.spec.body();
    let solvable = optimal_path(&s.terrain, s.from, s.goal, s.range as u8).is_some();
    let r_fat = r_fatigue(&s.terrain, s.from, s.goal, s.range);
    let r_tk = r_ticks(&s.terrain, &body, s.spec.carry_used, s.from, s.goal, s.range);
    let trav = rover_traversal(&s.terrain, &body, s.spec.carry_used, s.from, s.goal, s.range);
    let (me, fu) = trav
        .as_ref()
        .map(|t| (movement_eff(t), fatigue_util(t)))
        .unwrap_or((0.0, 0.0));

    let mut failures = Vec::new();
    if !solvable {
        failures.push("oracle could not reach the goal".to_string());
    }
    match r_fat {
        Some(r) if r <= s.max_r_fatigue + 1e-9 => {}
        Some(r) => failures.push(format!("R_fatigue {r:.4} > {:.4}", s.max_r_fatigue)),
        None => failures.push("R_fatigue: rover search incomplete or goal unreachable".to_string()),
    }
    match r_tk {
        Some(r) if r <= s.max_r_ticks + 1e-9 => {}
        Some(r) => failures.push(format!("R_ticks {r:.4} > {:.4}", s.max_r_ticks)),
        None => failures.push("R_ticks: traversal did not complete".to_string()),
    }
    if trav.is_some() && (fu - 1.0).abs() > 1e-9 {
        failures.push(format!("fatigue_util {fu:.4} != 1 (rover-attributable idle)"));
    }

    ScenarioResult {
        name: s.name,
        solvable,
        r_fatigue: r_fat,
        r_ticks: r_tk,
        movement_eff: me,
        fatigue_util: fu,
        passed: failures.is_empty(),
        failures,
    }
}

/// A balanced body ([ATTACK, MOVE]: sustains full speed on plains/roads, stalls on swamp).
fn balanced() -> CreepSpec {
    CreepSpec { parts: vec![Part::Attack, Part::Move], carry_used: 0 }
}
/// An under-MOVE body ([2×ATTACK, MOVE]: stalls every other tick even on plains).
fn under_move() -> CreepSpec {
    CreepSpec { parts: vec![Part::Attack, Part::Attack, Part::Move], carry_used: 0 }
}

/// The curated corpus: open plains, a wall detour, a swamp skirt, a road corridor, and the
/// fatigue-bound (under-MOVE / loaded / swamp-trap) cases. Each carries the gates it must meet.
pub fn catalog() -> Vec<Scenario> {
    let a = pos_in(room(), 10, 25);
    let b = pos_in(room(), 20, 25);

    let mut wall = SimTerrain::default();
    for y in 21..=29 {
        wall.walls.insert((15, y));
    }

    let mut swamp = SimTerrain::default();
    for x in 13..=17 {
        for y in 23..=27 {
            swamp.swamps.insert((x, y));
        }
    }

    let mut road = SimTerrain::default();
    for x in 13..=17 {
        for y in 0..=49 {
            road.swamps.insert((x, y));
        }
    }
    for x in 13..=17 {
        road.roads.insert((x, 25));
    }

    let mut swamp_trap = SimTerrain::default();
    swamp_trap.swamps.insert((15, 25)); // one swamp tile straddling the direct line

    vec![
        Scenario { name: "open_plain", terrain: SimTerrain::default(), spec: balanced(), from: a, goal: b, range: 0, max_r_fatigue: 1.0, max_r_ticks: 1.0 },
        Scenario { name: "wall_detour", terrain: wall, spec: balanced(), from: a, goal: b, range: 0, max_r_fatigue: 1.05, max_r_ticks: 1.1 },
        Scenario { name: "swamp_skirt", terrain: swamp, spec: balanced(), from: a, goal: b, range: 0, max_r_fatigue: 1.05, max_r_ticks: 1.1 },
        Scenario { name: "road_corridor", terrain: road, spec: balanced(), from: a, goal: b, range: 0, max_r_fatigue: 1.05, max_r_ticks: 1.1 },
        Scenario { name: "under_move_open", terrain: SimTerrain::default(), spec: under_move(), from: a, goal: b, range: 0, max_r_fatigue: 1.0, max_r_ticks: 1.0 },
        Scenario { name: "loaded_hauler", terrain: SimTerrain::default(), spec: CreepSpec { parts: vec![Part::Carry, Part::Carry, Part::Move], carry_used: 100 }, from: a, goal: b, range: 0, max_r_fatigue: 1.0, max_r_ticks: 1.0 },
        Scenario { name: "swamp_trap_under_move", terrain: swamp_trap, spec: under_move(), from: a, goal: b, range: 0, max_r_fatigue: 1.05, max_r_ticks: 1.1 },
    ]
}

/// A seeded REALISTIC room (ADR 0044): the shared cellular-automata cave generator
/// ([`screeps_sim_core::terrain_gen`]) — clustered walls + swamp, forcing genuine detours (unlike
/// the open/patch corpus). Endpoints are two CONNECTED interior tiles (via `connected_open`, never
/// on an edge — that would relocate the creep), so the goal is always reachable through the cave.
/// This stresses rover's route-optimality on hard terrain — the regime the operator flagged the
/// trivial rooms never exercised.
pub fn generate_realistic(seed: u32) -> Scenario {
    use screeps_sim_core::terrain_gen::{connected_open, generate_terrain, Exits, TerrainGenParams};
    let terrain = generate_terrain(seed, &TerrainGenParams { exits: Exits::horizontal(), ..Default::default() });
    let region = connected_open(&terrain, (25, 25));
    // The connected interior tile nearest a target (excluding edge tiles that would trigger the
    // kernel's cross-room relocation).
    let pick = |tx: i32| {
        region
            .iter()
            .filter(|&&(x, y)| (1..=48).contains(&x) && (1..=48).contains(&y))
            .min_by_key(|&&(x, y)| (x as i32 - tx).pow(2) + (y as i32 - 25).pow(2))
            .copied()
            .unwrap_or((25, 25))
    };
    let (fx, fy) = pick(6);
    let (gx, gy) = pick(43);
    Scenario {
        name: "generated_realistic",
        terrain,
        spec: balanced(),
        from: pos_in(room(), fx, fy),
        goal: pos_in(room(), gx, gy),
        range: 0,
        // Loose bounds — realistic caves detour more; the point is rover stays NEAR-optimal.
        max_r_fatigue: 1.15,
        max_r_ticks: 1.5,
    }
}

/// A seeded procedural room: scattered swamp patches + an optional road stripe (both passable, so the
/// goal is always oracle-reachable — no walls), with a balanced creep crossing it. Loose gates: this
/// tests that rover routes a random fatigue field near-optimally, not a specific layout.
pub fn generate(seed: u32) -> Scenario {
    let mut rng = Rng::seeded(seed);
    let mut terrain = SimTerrain::default();

    let patches = rng.range(3, 8);
    for _ in 0..patches {
        let cx = rng.range(8, 41) as u8;
        let cy = rng.range(8, 41) as u8;
        let r = rng.range(1, 3) as u8;
        for x in cx.saturating_sub(r)..=(cx + r).min(48) {
            for y in cy.saturating_sub(r)..=(cy + r).min(48) {
                terrain.swamps.insert((x, y));
            }
        }
    }
    if rng.chance(50) {
        let ry = rng.range(12, 37) as u8;
        for x in 5..=44 {
            terrain.roads.insert((x, ry));
        }
    }

    let fy = rng.range(20, 30) as u8;
    let gy = rng.range(20, 30) as u8;
    Scenario {
        name: "generated",
        terrain,
        spec: balanced(),
        from: pos_in(room(), 5, fy),
        goal: pos_in(room(), 44, gy),
        range: 0,
        max_r_fatigue: 1.10,
        max_r_ticks: 1.25,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_catalog_scenario_passes_its_gates() {
        for s in catalog() {
            let r = validate(&s);
            assert!(r.passed, "scenario `{}` failed its gates: {:?}", r.name, r.failures);
        }
    }

    #[test]
    fn generated_rooms_are_solvable_and_near_optimal() {
        for seed in 0..25u32 {
            let s = generate(seed);
            let r = validate(&s);
            assert!(
                r.passed,
                "generated scenario (seed {seed}) failed: {:?} [R_fatigue={:?} R_ticks={:?}]",
                r.failures, r.r_fatigue, r.r_ticks
            );
        }
    }

    /// ADR 0044: rover stays NEAR-OPTIMAL and never hits the ops cap on REALISTIC cave terrain over
    /// a seed sweep — the deviation check for wiring the shared generator into the pathing sim. Prints
    /// the worst ratios so re-tuning (thresholds / ops cap) is evidence-driven.
    #[test]
    fn realistic_rooms_stay_near_optimal() {
        let (mut worst_fat, mut worst_tk, mut incomplete, mut unsolvable) = (0.0f64, 0.0f64, 0u32, 0u32);
        for seed in 0..30u32 {
            let s = generate_realistic(seed);
            let r = validate(&s);
            if !r.solvable {
                unsolvable += 1;
                continue;
            }
            match (r.r_fatigue, r.r_ticks) {
                (Some(f), Some(t)) => {
                    worst_fat = worst_fat.max(f);
                    worst_tk = worst_tk.max(t);
                }
                _ => incomplete += 1,
            }
        }
        eprintln!("realistic pathing over 30 seeds: worst R_fatigue={worst_fat:.4} worst R_ticks={worst_tk:.4} ops-cap/incomplete={incomplete} unsolvable={unsolvable}");
        assert_eq!(unsolvable, 0, "generated realistic rooms must be solvable (endpoints are connected)");
        assert_eq!(incomplete, 0, "rover hit the 20k ops cap on a single realistic room — raise the cap or the terrain is too dense");
        assert!(worst_fat <= 1.15, "rover route-optimality degraded on caves: worst R_fatigue={worst_fat}");
        assert!(worst_tk <= 1.5, "rover travel-time degraded on caves: worst R_ticks={worst_tk}");
    }

    #[test]
    fn validate_flags_an_unsolvable_scenario() {
        // Box the goal in with walls: the oracle cannot reach it → not solvable, gates fail.
        let mut terrain = SimTerrain::default();
        for (x, y) in [(19, 24), (19, 25), (19, 26), (20, 24), (20, 26), (21, 24), (21, 25), (21, 26)] {
            terrain.walls.insert((x, y));
        }
        let s = Scenario {
            name: "boxed_goal",
            terrain,
            spec: balanced(),
            from: pos_in(room(), 10, 25),
            goal: pos_in(room(), 20, 25),
            range: 0,
            max_r_fatigue: 1.0,
            max_r_ticks: 1.0,
        };
        let r = validate(&s);
        assert!(!r.solvable, "a walled-in goal must be unsolvable");
        assert!(!r.passed, "an unsolvable scenario must fail its gates");
    }
}
