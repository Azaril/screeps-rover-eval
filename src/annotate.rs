//! The §D5.4 decision-(5) LIVE-KERNEL BRIDGE: construct [`GoalAnnotation`]s from the REAL
//! force-sizing / objective-value kernels, proving the pre-priced-annotation contract end-to-end —
//! `p_win`/`est_ticks` from `force_sizing`'s oracle ([`assess`]/[`win_probability`]), `value_e` from
//! `objective_value` (ADR 0032's one energy-equivalent currency), `α_class` as spawn-cost shares over
//! the oracle's [`RequiredForce`].
//!
//! **Layering (why the bridge lives HERE, not in value.rs):** `value.rs` is the pure kernel — it
//! deliberately has NO combat deps (the same wall that keeps `SimBodyCombat` out of sim-core,
//! sim-core/src/body.rs:3-5), so `w(creep)` stays computable from a creep + a pre-priced annotation
//! alone. The pricing legitimately happens ONE layer up, where the two stacks already meet:
//! rover-eval is harness code that stands on combat-eval, so a direct `screeps-combat-decision` dep
//! here forks nothing — it CALLS the landed kernels, exactly as decision (5) demands ("the eval never
//! forks them"). This mirrors the bot's own seam: war.rs projects room intel into `ObjectiveIntel`
//! at the consumption point (war.rs:1720, the ObjectiveIntel pattern — cited, not read) and hands
//! the decision crate pre-priced facts; here the scenario harness plays war.rs's role and
//! [`movement_intent_weight`](crate::value::movement_intent_weight) plays the kernel's.
//!
//! Determinism: pure scalar arithmetic over slices in caller order (callers supply members
//! id-sorted, so the first-max-`ttr` binding tie-break IS the id tie-break); no maps, no clock, no
//! RNG. Ordering over the produced weights goes through [`crate::value::ordering_key`] (quantized),
//! never raw f64 — the fence.

use screeps::Part;
use screeps_combat_decision::force_sizing::RequiredForce;
use screeps_sim_core::SimBody;

use crate::value::{
    class_power, GoalAnnotation, Role, SquadRef, ATTACK_POWER, DISMANTLE_POWER, HEAL_POWER,
    RANGED_ATTACK_POWER,
};

/// An objective's pre-priced facts — the three scalars [`SquadRef`] shares across the squad.
/// Computed by the caller from the real kernels: `p_win` = `force_sizing::win_probability(fielded
/// heal, incoming)` (f32, widened), `value_e` = `objective_value::value_e(kind, intel)` (ADR 0032),
/// `est_ticks` = the oracle's `ForceAssessment::est_ticks`. Taken as inputs (not recomputed here)
/// because the kernels' INPUTS — the `DefenseProfile`, the room intel — are scenario facts the
/// harness owns; the tests demonstrate the kernel calls end-to-end.
#[derive(Clone, Copy, Debug)]
pub struct ObjectiveFacts {
    /// Energy-equivalent unlock value of the objective (`objective_value::value_e`, widened f32→f64).
    pub value_e: f64,
    /// Lanchester win probability (`force_sizing::win_probability`, widened f32→f64).
    pub p_win: f64,
    /// Estimated on-site ticks to convert (`ForceAssessment::est_ticks`).
    pub est_ticks: u32,
}

/// One fielded squad member as the harness knows it: its military [`Role`], the body actually
/// spawned (which may over/under-fill its class — the renormalization in [`SquadRef`] handles both),
/// and the per-member movement facts (`ttr` selects the binding laggard; `t_min`/`recycle_ev` pass
/// through to the annotation, decision (4)'s harness-owned-lifetimes idiom).
#[derive(Clone, Debug)]
pub struct SquadMemberInput {
    pub role: Role,
    pub body: SimBody,
    /// Fatigue-exact ticks-to-rally/objective — the max-`ttr` member is the critical-path laggard
    /// and bids the full `R_O` (decision (1)). Ties break to the FIRST max (callers pass id-sorted).
    pub ttr: u32,
    /// Minimum useful on-site ticks (the A gate's floor) — forming/positioning time for a squad.
    pub t_min: u32,
    /// Recycle-recoverable energy (the escape bid prices only the surplus over this).
    pub recycle_ev: f64,
}

/// The four military capability classes [`RequiredForce`] can demand of squad ROLES. Tough/Claim
/// parts also exist on `RequiredForce` but have no movement Role here — their spawn cost counts in
/// the α denominator (below) AND is REDISTRIBUTED to the members whose BODIES carry the parts
/// (ratified 2026-07-01, operator: allocate to the carrier, body-based, no class guessing — see
/// [`carried_share`]), so a fully-specified squad integrates `V_O` EXACTLY, no unclaimed remainder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Melee,
    Ranged,
    Heal,
    Dismantle,
}

const CLASSES: [Class; 4] = [Class::Melee, Class::Ranged, Class::Heal, Class::Dismantle];

fn class_of(role: &Role) -> Option<Class> {
    match role {
        Role::Melee => Some(Class::Melee),
        Role::Ranged => Some(Class::Ranged),
        Role::Heal => Some(Class::Heal),
        Role::Dismantle => Some(Class::Dismantle),
        _ => None,
    }
}

/// Required parts of a class, read off the ORACLE's force. Ranged is the SUM of the two RANGED-
/// currency terms — `immune_struct_parts` (anti-structure) + `anti_creep_parts` (anti-creep) — the
/// same fold `assemble_force` applies to its RangedDPS role (force_sizing.rs:513). CALLER CONTRACT:
/// pass the emitter-zeroed force (one structure weapon per objective — `dismantle_parts` XOR
/// `immune_struct_parts`, force_sizing.rs:536); both nonzero would double-count the structure kill
/// in the α denominator. Melee is 0: `RequiredForce` carries no ATTACK term (kill parts are sized in
/// RANGED currency), so an ATTACK member prices as capital (upkeep floor), never as a `V_O` share.
fn required_parts(required: &RequiredForce, class: Class) -> u32 {
    match class {
        Class::Melee => 0,
        Class::Ranged => required.immune_struct_parts + required.anti_creep_parts,
        Class::Heal => required.heal_parts,
        Class::Dismantle => required.dismantle_parts,
    }
}

/// The class's action power per part — value.rs's transcribed engine constants, REUSED (never
/// re-transcribed): the same table `class_power` reads, so `required_class_power` and the fielded
/// power are in one unit by construction.
fn power_per_part(class: Class) -> u32 {
    match class {
        Class::Melee => ATTACK_POWER,
        Class::Ranged => RANGED_ATTACK_POWER,
        Class::Heal => HEAL_POWER,
        Class::Dismantle => DISMANTLE_POWER,
    }
}

/// Spawn energy of a class's required parts (`Part::cost()` — the same `BODYPART_COST` table
/// `value::body_cost_e` sums). MOVE parts are deliberately absent: α is a share over the required
/// CAPABILITY parts (the oracle sizes capabilities, not chassis; MOVE ratios are a body-builder
/// concern), and shares only need a consistent basis.
fn class_cost_e(required: &RequiredForce, class: Class) -> f64 {
    let part = match class {
        Class::Melee => Part::Attack,
        Class::Ranged => Part::RangedAttack,
        Class::Heal => Part::Heal,
        Class::Dismantle => Part::Work,
    };
    (required_parts(required, class) * part.cost()) as f64
}

/// The α denominator: total spawn cost of EVERYTHING the oracle demands — the four role classes plus
/// the role-less Tough/Claim parts (Σ α over ROLE classes ≤ 1, = 1 iff the force is all-role). The
/// Tough/Claim remainder does NOT vanish: [`carried_share`] hands it to the carrying bodies, so the
/// squad's shares still sum to 1 when the force is fully fielded.
fn total_required_cost_e(required: &RequiredForce) -> f64 {
    CLASSES.iter().map(|&c| class_cost_e(required, c)).sum::<f64>()
        + (required.tough_parts * Part::Tough.cost()) as f64
        + (required.claim_parts * Part::Claim.cost()) as f64
}

/// The two role-less CARRIER part kinds a `RequiredForce` can demand (tough buffer,
/// force_sizing.rs:517; declaim weapon, force_sizing.rs:522).
const CARRIER_PARTS: [Part; 2] = [Part::Tough, Part::Claim];

fn required_carrier_parts(required: &RequiredForce, part: Part) -> u32 {
    match part {
        Part::Tough => required.tough_parts,
        Part::Claim => required.claim_parts,
        _ => 0,
    }
}

/// Carrier parts on a member's BODY (total, not alive-only: spawn cost is sunk at spawn — the same
/// basis as `value::body_cost_e`).
fn carried_parts(body: &SimBody, part: Part) -> u32 {
    body.parts.iter().filter(|p| p.part == part).count() as u32
}

/// The member's REDISTRIBUTED share of the role-less Tough/Claim spawn cost (ratified 2026-07-01,
/// operator): each carrier kind contributes `(required_parts × cost / total_cost) ×
/// (own_parts / max(required_parts, fielded_parts))` — the SAME renormalization shape as the role
/// classes (overfill dilutes carriers; underfill leaves the shortfall unclaimed), keyed purely off
/// the BODY (no class guessing: whoever spawned with the TOUGH plates owns their cost-share). With
/// the α denominator unchanged (the full `RequiredForce` cost), a fully-fielded squad's shares —
/// role classes + carriers — sum to exactly 1, so the sample weights integrate `V_O` exactly once.
fn carried_share(
    required: &RequiredForce,
    body: &SimBody,
    fielded_carrier: &[u32; 2],
    total_cost: f64,
) -> f64 {
    if total_cost <= 0.0 {
        return 0.0;
    }
    let mut share = 0.0;
    for (k, &part) in CARRIER_PARTS.iter().enumerate() {
        let req = required_carrier_parts(required, part);
        let denom = req.max(fielded_carrier[k]);
        let own = carried_parts(body, part);
        if denom > 0 && own > 0 {
            let cost_share = (req * part.cost()) as f64 / total_cost;
            share += cost_share * (own as f64 / denom as f64);
        }
    }
    share
}

/// Build the squad's per-member [`GoalAnnotation`]s from the objective's pre-priced facts + the
/// oracle's [`RequiredForce`] + the fielded members (decision (5), the completing half).
///
/// Every member shares one [`SquadRef`] shape (`p_win`/`value_e`/`est_ticks` identical); per member:
/// `α_class` = its class's spawn-cost share of the REQUIRED force, `required_class_power` from the
/// oracle's part counts × the transcribed powers, `fielded_class_power` = Σ [`class_power`] over the
/// fielded classmates (the over/under-fill renormalization denominator, value.rs `SquadRef` docs),
/// `binding` = the first max-`ttr` member (the critical-path laggard, decision (1); first = the id
/// tie-break under the caller's id-sorted order). `value_stock_e` is set to the member's cost-share
/// of `V_O`: the class share [`crate::value::squad_sample_weight`] integrates PLUS the member's
/// body-carried Tough/Claim redistribution ([`carried_share`], ratified 2026-07-01) — so the escape
/// bid (decision (8)) prices exactly the objective share this member's death forfeits, and the
/// squad's stocks sum to exactly `V_O` when fully fielded.
///
/// A non-military member (a tag-along claimer/scout) gets a squadless annotation: it bids the
/// upkeep floor through value.rs's role table, never a share of `V_O`.
pub fn annotate_squad(
    facts: &ObjectiveFacts,
    required: &RequiredForce,
    members: &[SquadMemberInput],
) -> Vec<GoalAnnotation> {
    let total_cost = total_required_cost_e(required);
    // Fielded power per class + fielded carrier parts — Σ over MILITARY members, caller order
    // (id-sorted ⇒ deterministic). Non-military stragglers ride their own rail (below) and neither
    // contribute to nor draw from the carrier redistribution.
    let mut fielded = [0.0f64; 4];
    let mut fielded_carrier = [0u32; 2];
    for m in members {
        if let Some(class) = class_of(&m.role) {
            fielded[class as usize] += class_power(&m.role, &m.body);
            for (k, &part) in CARRIER_PARTS.iter().enumerate() {
                fielded_carrier[k] += carried_parts(&m.body, part);
            }
        }
    }
    // The binding laggard: first max-ttr (strict `>` keeps the earliest, the id tie-break).
    let mut binding_ix = 0usize;
    for (i, m) in members.iter().enumerate() {
        if m.ttr > members[binding_ix].ttr {
            binding_ix = i;
        }
    }

    members
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let Some(class) = class_of(&m.role) else {
                // Squadless straggler rail (value.rs progress_rate): upkeep floor, no V_O share.
                return GoalAnnotation {
                    t_min: m.t_min,
                    recycle_ev: m.recycle_ev,
                    ..Default::default()
                };
            };
            let required_class_power =
                (required_parts(required, class) * power_per_part(class)) as f64;
            let alpha_class = if total_cost > 0.0 {
                class_cost_e(required, class) / total_cost
            } else {
                0.0
            };
            let squad = SquadRef {
                p_win: facts.p_win,
                value_e: facts.value_e,
                est_ticks: facts.est_ticks,
                alpha_class,
                required_class_power,
                fielded_class_power: fielded[class as usize],
                binding: i == binding_ix,
            };
            // The member's cost-share of V_O: the class share (mirrors squad_sample_weight,
            // value.rs) + the body-carried Tough/Claim redistribution (ratified 2026-07-01) — the
            // stock at risk == the full share this member's death forfeits, and Σ over a
            // fully-fielded squad == V_O exactly (no unclaimed carrier remainder).
            let denom = squad.required_class_power.max(squad.fielded_class_power);
            let class_share = if denom > 0.0 {
                squad.alpha_class * (class_power(&m.role, &m.body) / denom)
            } else {
                0.0
            };
            let value_stock_e = (class_share
                + carried_share(required, &m.body, &fielded_carrier, total_cost))
                * squad.objective_value_e();
            GoalAnnotation {
                rate_e_t: 0.0, // military is annotation-priced, not t_star-priced (value.rs role table)
                value_stock_e,
                t_min: m.t_min,
                deadline: None,
                squad: Some(squad),
                recycle_ev: m.recycle_ev,
            }
        })
        .collect()
}

/// The trivial economic arm, for symmetry: a hauler cycling `q` cargo units per fatigue-exact
/// optimal round trip prices at `ρ = q / t_star_rtt` (the §D5.4 hauler reduction — `rate_e_t` is
/// exactly what makes `w` collapse to `Q/T*` under open gates). `cargo_value_e` is the stock aboard
/// (energy at par; non-energy cargo scenario-pinned, decision (6)) feeding the escape bid.
pub fn annotate_hauler(
    q: u32,
    t_star_rtt: u32,
    cargo_value_e: f64,
    recycle_ev: f64,
) -> GoalAnnotation {
    GoalAnnotation {
        rate_e_t: Role::Haul { q }.rate_e_t(t_star_rtt),
        value_stock_e: cargo_value_e,
        t_min: 0, // a hauler is useful the tick it arrives; the cycle owns its own timing
        deadline: None,
        squad: None,
        recycle_ev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{
        movement_intent_weight, ordering_key, squad_sample_weight, upkeep_e_t,
    };
    use screeps::{Position, RoomCoordinate, RoomName};
    use screeps_combat_decision::force_sizing::{
        assess, tower_dps_at_assault, win_probability, AssaultMode, DefenseProfile,
        ForceAssessment, ForceBudget, TowerIntel, TowerThreat,
    };
    use screeps_combat_decision::objective_value::{self, ObjectiveIntel, ObjectiveValueKind};
    use screeps_sim_core::SimCreep;

    fn pos(x: u8, y: u8) -> Position {
        let room: RoomName = "W1N1".parse().unwrap();
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
    }

    fn creep(id: u32, parts: &[Part]) -> SimCreep {
        SimCreep {
            id,
            owner: 0,
            pos: pos(25, 25),
            body: SimBody::unboosted(parts),
            fatigue: 0,
            carry_used: 0,
        }
    }

    fn member(role: Role, parts: &[Part], ttr: u32) -> SquadMemberInput {
        SquadMemberInput { role, body: SimBody::unboosted(parts), ttr, t_min: 20, recycle_ev: 0.0 }
    }

    /// (a) A force_sizing-sized quad's annotations integrate `V_O` EXACTLY ONCE through
    /// `value::squad_sample_weight` — the decision-(1) sample-weight invariant, proven over the REAL
    /// R2 mapping (`RequiredForce::from_assessment`), bit-exact by a dyadic fixture.
    #[test]
    fn force_sized_quad_annotations_integrate_v_o_exactly_once() {
        // The real R2 inverse: 24 heal/tick ⇒ 2 HEAL (defender_heal_parts_for_dps, bodies.rs:120),
        // 250 gross dismantle ⇒ 5 WORK. Chosen so the class SPAWN costs are dyadic shares:
        // 2×250 = 500 (HEAL) vs 5×100 = 500 (WORK) ⇒ α = 0.5/0.5 exact in f64 — the assert_eq
        // precondition (shares × power-of-two scalings are exact; nothing else is assumed).
        let a = ForceAssessment {
            winnable: true,
            mode: AssaultMode::Breach,
            required_heal_per_tick: 24.0,
            required_dismantle_dps: 250.0,
            required_tank_hp: 0.0,
            est_ticks: 800,
            reason: "test",
        };
        let mut required = RequiredForce::from_assessment(&a);
        assert_eq!((required.heal_parts, required.dismantle_parts), (2, 5), "the real R2 mapping");
        // The emitter zeroes the structure weapon the objective can't use (force_sizing.rs:536):
        // a WORK-armed breach squad carries dismantle_parts, so the RANGED alternative is zeroed —
        // leaving both would double-count the structure kill in the α denominator.
        required.immune_struct_parts = 0;

        // V_O = 0.75 × 40_000 = 30_000 exact. The quad: 2×1-HEAL healers (each 12/24 = 0.5 of the
        // class), 1×5-WORK dismantler (the whole class), 1 melee (RequiredForce has no ATTACK term
        // ⇒ α = 0 ⇒ weight exactly 0 — capital, not a V_O share).
        let facts = ObjectiveFacts { value_e: 40_000.0, p_win: 0.75, est_ticks: 800 };
        let members = [
            member(Role::Heal, &[Part::Heal, Part::Move], 100),
            member(Role::Heal, &[Part::Heal, Part::Move], 105),
            member(
                Role::Dismantle,
                &[Part::Work, Part::Work, Part::Work, Part::Work, Part::Work, Part::Move],
                110,
            ),
            member(Role::Melee, &[Part::Attack, Part::Move], 90),
        ];
        let anns = annotate_squad(&facts, &required, &members);
        assert_eq!(anns.len(), 4);

        let mut sum = 0.0;
        for (i, (m, ann)) in members.iter().zip(&anns).enumerate() {
            let sq = ann.squad.as_ref().expect("military members carry a SquadRef");
            let c = creep(i as u32 + 1, &[]); // body swapped in below — squad_sample_weight reads it
            let c = SimCreep { body: m.body.clone(), ..c };
            let w = squad_sample_weight(&c, &m.role, sq);
            // The annotation's stock at risk IS the member's integrated share (the escape bid
            // forfeits exactly what the benchmark credits).
            assert_eq!(ann.value_stock_e, w, "member {i}: value_stock_e mirrors the sample weight");
            sum += w;
        }
        assert_eq!(sum, 30_000.0, "the quad integrates V_O exactly once (0.25+0.25+0.5+0 shares)");
    }

    /// (a′) TOUGH/CLAIM spawn-cost REDISTRIBUTION (ratified 2026-07-01, operator): the role-less
    /// carrier cost goes to the member whose BODY carries the parts — body-based, no class
    /// guessing — with the α denominator unchanged (the full `RequiredForce` cost), so the squad's
    /// stocks integrate `V_O` EXACTLY (the pre-ratification behavior left the TOUGH share
    /// permanently unclaimed). Dyadic fixture ⇒ assert_eq exact: required = 3 HEAL (750 e) +
    /// 25 TOUGH (250 e), total 1000 e ⇒ shares 0.75 / 0.25 exact in f64.
    #[test]
    fn tough_carrier_tank_claims_exactly_the_tough_cost_share_and_squad_totals_v_o() {
        let required = RequiredForce { heal_parts: 3, tough_parts: 25, ..Default::default() };
        // V_O = 0.5 × 64_000 = 32_000 exact.
        let facts = ObjectiveFacts { value_e: 64_000.0, p_win: 0.5, est_ticks: 800 };
        // The tank: an ATTACK-armed TOUGH carrier. Its melee CLASS share is exactly 0 —
        // `RequiredForce` has no ATTACK term (required_parts = 0 ⇒ α_melee = 0) — so its whole
        // stock is the carried redistribution: the pre-ratification share was 0.
        let tough_tank: Vec<Part> = std::iter::repeat_n(Part::Tough, 25)
            .chain([Part::Attack, Part::Move])
            .collect();
        let members = [
            member(Role::Heal, &[Part::Heal, Part::Heal, Part::Heal, Part::Move], 100),
            member(Role::Melee, &tough_tank, 110),
        ];
        let anns = annotate_squad(&facts, &required, &members);

        // The tank's class weight through squad_sample_weight is 0 (α_melee = 0)…
        let tank = SimCreep { body: members[1].body.clone(), ..creep(2, &[]) };
        let w_class = squad_sample_weight(&tank, &Role::Melee, anns[1].squad.as_ref().unwrap());
        assert_eq!(w_class, 0.0, "melee has no RequiredForce term: the class share is 0");
        // …so its stock GREW by exactly the TOUGH spawn-cost share: 25×10/1000 = 0.25 of V_O.
        assert_eq!(
            anns[1].value_stock_e,
            w_class + 0.25 * 32_000.0,
            "the carrier's share grew by exactly the TOUGH cost share (8000 e)"
        );
        // The non-carrier healer is untouched: stock == its class sample weight (0.75 · V_O).
        let healer = SimCreep { body: members[0].body.clone(), ..creep(1, &[]) };
        assert_eq!(
            anns[0].value_stock_e,
            squad_sample_weight(&healer, &Role::Heal, anns[0].squad.as_ref().unwrap()),
        );
        assert_eq!(anns[0].value_stock_e, 0.75 * 32_000.0);
        // The exact-integration invariant: Σ stocks == V_O, no unclaimed carrier remainder.
        assert_eq!(
            anns[0].value_stock_e + anns[1].value_stock_e,
            32_000.0,
            "role shares + carried shares integrate V_O exactly once"
        );
    }

    /// (a″) Carrier renormalization mirrors the class shape: OVERFILLED TOUGH (two tanks fielding
    /// 2× the required plates) dilutes each carrier — the carrier pool still integrates exactly the
    /// required TOUGH cost share, never more.
    #[test]
    fn overfilled_tough_carriers_dilute_but_never_exceed_the_carrier_share() {
        let required = RequiredForce { heal_parts: 3, tough_parts: 25, ..Default::default() };
        let facts = ObjectiveFacts { value_e: 64_000.0, p_win: 0.5, est_ticks: 800 };
        let tough_tank: Vec<Part> = std::iter::repeat_n(Part::Tough, 25)
            .chain([Part::Attack, Part::Move])
            .collect();
        let members = [
            member(Role::Heal, &[Part::Heal, Part::Heal, Part::Heal, Part::Move], 100),
            member(Role::Melee, &tough_tank, 110),
            member(Role::Melee, &tough_tank, 120), // fielded 50 TOUGH vs required 25
        ];
        let anns = annotate_squad(&facts, &required, &members);
        // Each tank: 0.25 × (25/max(25, 50)) = 0.125 of V_O — diluted, Σ carriers = 0.25 exactly.
        assert_eq!(anns[1].value_stock_e, 0.125 * 32_000.0);
        assert_eq!(anns[2].value_stock_e, 0.125 * 32_000.0);
        assert_eq!(
            anns[0].value_stock_e + anns[1].value_stock_e + anns[2].value_stock_e,
            32_000.0,
            "overfill dilutes carriers; the squad still integrates V_O exactly once"
        );
    }

    /// (b) Contention semantics (decision (1)): the binding (max-ttr) member bids the full `R_O`
    /// through `value::movement_intent_weight`; a non-binding member bids the upkeep floor; the
    /// quantized ordering ranks them accordingly.
    #[test]
    fn binding_member_bids_r_o_and_non_binding_bids_the_floor() {
        let mut required = RequiredForce {
            heal_parts: 2,
            dismantle_parts: 5,
            ..Default::default()
        };
        required.immune_struct_parts = 0; // emitter-zeroed (WORK is the weapon)
        let facts = ObjectiveFacts { value_e: 40_000.0, p_win: 0.75, est_ticks: 800 };
        let members = [
            member(Role::Heal, &[Part::Heal, Part::Move], 100),
            member(
                Role::Dismantle,
                &[Part::Work, Part::Work, Part::Work, Part::Work, Part::Work, Part::Move],
                120, // the critical-path laggard
            ),
        ];
        let anns = annotate_squad(&facts, &required, &members);
        assert!(!anns[0].squad.as_ref().unwrap().binding);
        assert!(anns[1].squad.as_ref().unwrap().binding, "max ttr ⇒ the binding laggard");

        // Open gates: ttl 900, ttr ≤ 120, t_min 20 ⇒ A = 1; window ≤ t_min + est_ticks = 820 ⇒
        // U = 1 (binding window); no threat ⇒ S = 1. So w == the raw r_bid, exactly.
        let healer = creep(1, &[Part::Heal, Part::Move]);
        let dism = creep(
            2,
            &[Part::Work, Part::Work, Part::Work, Part::Work, Part::Work, Part::Move],
        );
        let w_floor = movement_intent_weight(&healer, &Role::Heal, &anns[0], 100, 900, true, None);
        let w_bind =
            movement_intent_weight(&dism, &Role::Dismantle, &anns[1], 120, 900, true, None);
        assert_eq!(
            w_bind,
            anns[1].squad.as_ref().unwrap().objective_rate(),
            "the binding member bids the FULL R_O = p_win·value_e/est_ticks"
        );
        assert_eq!(w_bind, 30_000.0 / 800.0, "R_O = 37.5 e/t for this fixture");
        assert_eq!(
            w_floor,
            upkeep_e_t(&healer.body),
            "a non-binding member bids body_cost/1500 (decision (2))"
        );
        // The contention rail: quantized keys rank the laggard first — never raw f64 (the fence).
        assert!(ordering_key(w_bind, 2) > ordering_key(w_floor, 1));
    }

    /// (c) END-TO-END over the real kernels: `assess` sizes a breach against a defended target,
    /// `RequiredForce::from_assessment` maps it to parts, `win_probability` + `objective_value::
    /// value_e` price the facts, `annotate_squad` bridges — and every member's `w` is finite/
    /// positive under open gates with a deterministic quantized order.
    #[test]
    fn end_to_end_real_oracle_prices_a_defended_target_squad() {
        // One energized tower at range 20 (falloff floor 150 dps) + 30 enemy creep dps: incoming
        // 180, required heal 234 (×1.3 hold margin) — a winnable direct breach for this budget.
        let profile = DefenseProfile {
            towers: vec![TowerThreat { range_to_assault: 20, energy: 1000 }],
            breach_hits: 20_000,
            objective_hits: 50_000,
            repair_per_tick: 0.0,
            safe_mode: false,
            tower_intel: TowerIntel::Seen,
        };
        let enemy_dps = 30.0f32;
        let budget = ForceBudget {
            max_heal_per_tick: 600.0,
            max_dismantle_dps: 600.0,
            tank_effective_hp: 5_000.0,
            onsite_budget_ticks: 1200,
        };
        let a = assess(&profile, enemy_dps, &budget);
        assert!(a.winnable, "the oracle sizes this breach: {}", a.reason);
        assert_eq!(a.mode, AssaultMode::Breach);

        let mut required = RequiredForce::from_assessment(&a);
        required.immune_struct_parts = 0; // emitter-zeroed: WORK is this objective's weapon
        assert!(required.heal_parts > 0 && required.dismantle_parts > 0);

        // The real pricing kernels (f32 → f64 widening at the bridge boundary):
        // p_win over the FIELDED heal (parts × 12) vs the incoming the oracle assessed;
        // value_e = the FarmCore economic unlock (denied-reservation income — the combat-EV-economic
        // directive: the unlock, never force-elimination).
        let incoming = tower_dps_at_assault(&profile.towers) + enemy_dps;
        let fielded_heal = (required.heal_parts * HEAL_POWER) as f32;
        let p_win = win_probability(fielded_heal, incoming) as f64;
        let v_e = objective_value::value_e(
            ObjectiveValueKind::FarmCore,
            &ObjectiveIntel { income_per_tick: 10.0, horizon: 1500.0, ..Default::default() },
        ) as f64;
        assert!(p_win > 0.5, "fielded heal exceeds break-even ⇒ P(win) > 0.5: {p_win}");
        assert_eq!(v_e, 15_000.0, "the FarmCore arm: 10 e/t × 1500 ticks");
        let facts = ObjectiveFacts { value_e: v_e, p_win, est_ticks: a.est_ticks };

        // Field the sized force: heal split across two healers, dismantle across two workers —
        // fielded == required per class (halves are exact; the shares just don't need to be).
        let hp = required.heal_parts; // 20 for this fixture (234/12 ceil)
        let dp = required.dismantle_parts; // 12 (600 gross / 50)
        assert_eq!((hp, dp), (20, 12), "pin the fixture so the bodies below match the oracle");
        let heal_body: Vec<Part> =
            std::iter::repeat_n(Part::Heal, (hp / 2) as usize).chain([Part::Move; 3]).collect();
        let work_body: Vec<Part> =
            std::iter::repeat_n(Part::Work, (dp / 2) as usize).chain([Part::Move; 3]).collect();
        let members = [
            member(Role::Heal, &heal_body, 100),
            member(Role::Heal, &heal_body, 105),
            member(Role::Dismantle, &work_body, 110),
            member(Role::Dismantle, &work_body, 120), // the laggard
        ];
        let anns = annotate_squad(&facts, &required, &members);

        // Every member's w is finite + strictly positive under open gates (Δ=1, no threat): the
        // laggard bids toward R_O, the rest at least the upkeep floor — never 0, never NaN.
        let creeps: Vec<SimCreep> = members
            .iter()
            .enumerate()
            .map(|(i, m)| SimCreep { body: m.body.clone(), ..creep(i as u32 + 1, &[]) })
            .collect();
        let weights: Vec<f64> = members
            .iter()
            .zip(&anns)
            .zip(&creeps)
            .map(|((m, ann), c)| movement_intent_weight(c, &m.role, ann, m.ttr, 1400, true, None))
            .collect();
        for (i, w) in weights.iter().enumerate() {
            assert!(w.is_finite() && *w > 0.0, "member {i}: w = {w} must be finite/positive");
        }
        assert!(anns[3].squad.as_ref().unwrap().binding, "max ttr = the binding member");

        // Deterministic quantized order: two independent annotation+weight passes produce the same
        // key sequence, and the id tie-break totalizes it (the fence — no raw-f64 ordering).
        let keys = |ws: &[f64]| {
            let mut k: Vec<_> =
                ws.iter().zip(&creeps).map(|(w, c)| ordering_key(*w, c.id)).collect();
            k.sort_unstable_by(|x, y| y.cmp(x)); // descending: highest bid first
            k
        };
        let anns2 = annotate_squad(&facts, &required, &members);
        let weights2: Vec<f64> = members
            .iter()
            .zip(&anns2)
            .zip(&creeps)
            .map(|((m, ann), c)| movement_intent_weight(c, &m.role, ann, m.ttr, 1400, true, None))
            .collect();
        assert_eq!(keys(&weights), keys(&weights2), "re-annotation reproduces the exact order");
        let k = keys(&weights);
        assert!(k.windows(2).all(|w| w[0] > w[1]), "the id tie-break yields a strict total order");
    }

    /// The hauler arm (symmetry): the annotation carries `ρ = q/T*` so `w` collapses to the §D5.4
    /// hauler reduction under open gates, and the cargo stock feeds the escape bid unchanged.
    #[test]
    fn annotate_hauler_carries_the_t_star_rate_and_cargo_stock() {
        let ann = annotate_hauler(100, 50, 100.0, 0.0);
        assert_eq!(ann.rate_e_t, 2.0, "ρ = 100 cargo / 50-tick oracle round trip");
        assert_eq!(ann.value_stock_e, 100.0);
        assert!(ann.squad.is_none(), "haulers ride the economic rail, no SquadRef");
        let hauler = creep(1, &[Part::Carry, Part::Carry, Part::Move]);
        let w = movement_intent_weight(&hauler, &Role::Haul { q: 100 }, &ann, 200, 1500, true, None);
        assert_eq!(w, 2.0, "the hauler reduction: w == Q/T* with every gate open");
    }
}
