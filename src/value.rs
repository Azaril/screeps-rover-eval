//! The objective-function value kernel (ADR 0033 §D5.4): `w(creep)` — the expected
//! energy-equivalent value of spending one movement intent, the single scalar that collapses roles
//! so the benchmark (and eventually rover's own contention triage) can weight movements by what
//! they are worth.
//!
//! **Currency:** energy-equivalent per tick (e/t) for rates; energy-equivalent stocks for value at
//! risk. This is the codebase's one existing cross-goal currency (ADR 0032 `value_e`, ADR 0038
//! `room_net_roi`) — no new unit.
//!
//! **Contract:** `w = G · r_bid`, where `r_bid` is the role's slack-gated productive rate (the
//! §D5.4 role table) and `G = Δ_crit · A · S · U` are the criticality / arrive-in-time / survival /
//! slack gates. `w` is computable from a `SimCreep` + a per-creep [`GoalAnnotation`] the scenario
//! harness precomputes (fleet/squad context arrives pre-priced — the eval never forks the economic
//! kernels), + caller-supplied `ttr`/`ttl` and an optional [`ThreatView`]. Weights are quantized to
//! integer milli-e/t (+ a stable-id tie-break) before any ordering, per the determinism fence.
//!
//! **Scope:** the hauler arm (the operator's "maximum value transport by distance" anchor) plus the
//! MILITARY / worker / claimer / scout arms — the §D5.4 role table in full, with the 11 decisions
//! RATIFIED 2026-07-01 under live-constant alignment (the `## §D5.4 decisions` block below).
//! Entry point: [`movement_intent_weight`]; squad benchmark weights: [`squad_sample_weight`].

use screeps::Part;
use screeps_sim_core::world::CreepId;
use screeps_sim_core::{SimBody, SimCreep};

// ═════════════════════════════════════════════════════════════════════════════════════════════
// Engine constants, transcribed. rover-eval deliberately does NOT depend on `screeps-combat-engine`
// (layering, ADR 0033: the movement bench sits on `sim-core`, never on the combat stack — the same
// wall that keeps `SimBodyCombat` out of the kernel, sim-core/src/body.rs:3-5). Transcribing five
// scalar action powers is cheaper than a layering violation; drift risk is nil (engine ground truth,
// docs/references/engine-mechanics.md; unchanged since launch), and each cites its source line.
// ═════════════════════════════════════════════════════════════════════════════════════════════

/// Per-part action power (unboosted) — `screeps-combat-engine/src/constants.rs:15-19`, themselves
/// verified against `C:\code\screeps-engine` `common/constants.js`.
pub const ATTACK_POWER: u32 = 30; // ATTACK, melee, range 1
pub const RANGED_ATTACK_POWER: u32 = 10; // RANGED_ATTACK, range 3
pub const HEAL_POWER: u32 = 12; // HEAL adjacent, range 1
pub const RANGED_HEAL_POWER: u32 = 4; // HEAL at range, range 3 (declared for the full transcribed set)
pub const DISMANTLE_POWER: u32 = 50; // WORK dismantle, range 1

/// Creep lifetimes (`screeps-combat-engine/src/constants.rs:11-12`; engine `CREEP_LIFE_TIME` /
/// `CREEP_CLAIM_LIFE_TIME`). 1500 is the amortization denominator of the ubiquitous
/// `body_cost/CREEP_LIFE_TIME` upkeep rate (§D5.4 "the ubiquitous amortization").
pub const CREEP_LIFE_TIME: u32 = 1500;
pub const CREEP_CLAIM_LIFE_TIME: u32 = 600;

/// Worker energy conversion per WORK part per tick (engine `BUILD_POWER` = 5 energy/tick spent
/// building; `UPGRADE_CONTROLLER_POWER` = 1 energy/tick upgrading) — natively energy, no laundering.
pub const BUILD_POWER_E_T: f64 = 5.0;
pub const UPGRADE_POWER_E_T: f64 = 1.0;

// ═════════════════════════════════════════════════════════════════════════════════════════════
// ## §D5.4 decisions — RATIFIED (2026-07-01, operator). The 11 items (ADR 0033 §D5.4 "Open
// decisions") stand as written below, with LIVE-CONSTANT ALIGNMENT applied at ratification:
// `CLAIM_ARRIVAL_MARGIN` 100→50 (the live margin, screeps-ibex/src/missions/utility.rs:66) and the
// S-veto comparison ≥→strict `>` (the live kite.rs:713) — noted on decisions (3)/(8) and in the
// implementation-detail defaults list. Any future change remains a constant or match-arm change,
// never a schema change.
//
//  (1) Squad bid semantics — DECIDED binding-member-bids-`R_O` for CONTENTION, cost-shares
//      (`α_class·s_member`) for SAMPLE weights. The critical-path laggard delays the WHOLE
//      objective, so denying its move costs the full `R_O` (the healer a quad waits on outbids a
//      fat hauler — the historical squad-cohesion failure, fixed by construction); H's weights must
//      instead integrate each job's `V_O` exactly once, which the cost-shares do.
//  (2) Slack floor — DECIDED amortized upkeep (`body_cost/1500`), not 0. An alive creep is never
//      free to stall: its body is sunk capital depreciating at exactly that rate. Implemented as
//      the U-gate floor ratio `min(1, upkeep/r)`, so a member already bidding upkeep has U == 1.
//  (3) Policy constants — DECIDED `T_RAMP = 20` (sharp enough to matter within one squad forming
//      window, wide enough to avoid the binary-P(win) cliff §D5.4 warns about), `S_REF = 100`
//      (claim hazard smoothing: a claimer inside 100 ticks of its deadline prices as if at 100 —
//      caps the 1/slack explosion), `EPSILON_INTEL` = a 1×MOVE scout's upkeep (50/1500 e/t — VOI
//      has no landed kernel; a declared floor, not a buried magic number), `V_SINK = 1.0` (build
//      and upgrade energy both count at par until a sink-value kernel lands). The promised
//      sensitivity sweep is LANDED: [`PolicyParams`] (Default == these constants, bit-for-bit) +
//      `tuning.rs::sweep_policy_params` (env-driven, reports which points change quantized bid
//      ORDER over the role-table fixture set — rank-affecting ranges, not raw deltas). RATIFIED
//      as proposed, plus one alignment outside the four sweep constants: the claimer arrival
//      margin drops to the LIVE 50 (utility.rs:66) — see the defaults list below.
//  (4) ttl plumbing — DECIDED scenario-supplied (an explicit `ttl` parameter): zero `SimCreep` /
//      kernel schema change; the harness owns lifetimes like it owns terrain.
//  (5) Economic-facts bridge — DECIDED pre-priced in [`GoalAnnotation`] (the ObjectiveIntel
//      pattern): `p_win`/`value_e`/`α_class` arrive computed by the landed `force_sizing` /
//      `objective_value` / `room_net_roi` kernels; the eval never forks them (layering).
//  (6) Non-energy cargo — DECIDED scenario-pinned constants folded into `Haul::q` /
//      `rate_e_t` (seed-stable), not live FairValue (a market read is a determinism-fence breach).
//  (7) Renewal — DECIDED ttl is hard, no renewal modeling in v1: renewal is spawn-layer policy,
//      out of the movement bench's scope; a renewed creep is just a new (ttl, annotation) pair.
//  (8) Escape bid — DECIDED `w = max(w_progress, w_escape)` (one intent does one or the other —
//      summing would double-bid a single tile), with the wounded discount `V·hits/hits_max` (a
//      nearly-dead creep's remaining stock has already been destroyed; don't over-bid saving it).
//      RATIFIED with the S-veto handoff aligned to the live STRICT `>` (kite.rs:713): the veto —
//      and thus the escape takeover — fires only when net×horizon strictly exceeds hits; the
//      exact-lethal boundary now survives, matching live.
//  (9) Runtime adoption — DECIDED benchmark-only for now: replacing rover's High/Normal/Low
//      resolver priority with quantized `w` is the follow-up, gated on a combat-corpus tournament
//      (the same gate that held back `StuckThresholds` in M5).
// (10) Multi-room T*/ttr — DECIDED the approximation is accepted; `w` consumes `ttr` opaquely, so
//      the parallel multi-room T* work lands without touching this module.
// (11) H reporting — DECIDED per-family H primary + pooled secondary; LANDED in `tuning.rs`
//      (`TuneScenario::family` → `TuneScore::per_family` + `family_report`; the pooled H stays
//      the ranked key — this module only defines the (value, weight) samples).
//
// Implementation-detail defaults (documented so a veto knows what it is vetoing):
//  - U decay shape: past the binding window, `U = max(floor_ratio, 1/(1 + excess/T_RAMP))` —
//    continuous at the boundary, ~upkeep within a few hundred ticks of slack (the ADR's 800-ttl vs
//    200-tick-job squad bids ~upkeep and loses tiles to a loaded hauler, as specified).
//  - Continuous-cycle roles (Haul/Work/Scout) have `U == 1` always: their job consumes the whole
//    remaining life, so the window binds by construction (this is what makes the §D5.4 hauler
//    reduction exact).
//  - Military horizon_needed = `t_min + est_ticks` (arrive + fight): the window binds while the
//    creep cannot both reach and serve out the objective with room to spare.
//  - S-veto comparison is the live STRICT `>` (combat-decision/src/kite.rs:713): exactly-lethal
//    (`net × SURVIVAL_HORIZON == hits`) SURVIVES. Ratified 2026-07-01 — the pre-ratification
//    one-ulp `≥` tightening is dropped for live parity.
//  - `CLAIM_ARRIVAL_MARGIN = 50` == the live margin (screeps-ibex/src/missions/utility.rs:66, the
//    claim-intent latency). Ratified 2026-07-01 — the pre-ratification 2× bench margin (100,
//    arrival-positioning slack under contention) is dropped for live parity.
// ═════════════════════════════════════════════════════════════════════════════════════════════

/// A-gate ramp width (ticks) — decision (3). `A = clamp((ttl − ttr − t_min)/T_RAMP, 0, 1)`.
pub const T_RAMP: f64 = 20.0;
/// Claim hazard-smoothing reference slack (ticks) — decision (3): `min(V, V/max(slack, S_REF))`.
pub const S_REF: f64 = 100.0;
/// Survival-veto horizon (ticks) — combat-decision/src/kite.rs:24 (ADR 0019 Guard 4), generalized:
/// progress you don't survive to convert is not progress (kite.rs:704-714).
pub const SURVIVAL_HORIZON: u32 = 3;
/// Claimer arrival margin (ticks) — ratified 2026-07-01 to the LIVE 50
/// (screeps-ibex/src/missions/utility.rs:66; see the decisions block above).
pub const CLAIM_ARRIVAL_MARGIN: u32 = 50;
/// Scout intel floor (e/t) — decision (3): a 1×MOVE scout's amortized upkeep. A declared policy
/// constant standing in for a value-of-information kernel that does not exist yet.
pub const EPSILON_INTEL_E_T: f64 = 50.0 / CREEP_LIFE_TIME as f64;
/// Sink-value multiplier for worker energy — decision (3): build/upgrade energy at par (1.0).
pub const V_SINK: f64 = 1.0;

/// The decision-(3) policy constants as a swappable bundle — the sensitivity-sweep handle the
/// decision block promises ("ship defaults + sensitivity sweep"). `Default` IS the decided
/// constants above, bit-for-bit, so [`movement_intent_weight`] (which delegates with `Default`)
/// is byte-unchanged; the sweep (`tuning.rs::sweep_policy_params`) probes off-default points via
/// [`movement_intent_weight_with`] and reports which ranges actually CHANGE quantized bid order —
/// the evidence an operator veto of decision (3) would be priced against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyParams {
    /// A-gate ramp width + the U-gate decay scale (ticks) — [`T_RAMP`].
    pub t_ramp: f64,
    /// Claim hazard-smoothing reference slack (ticks) — [`S_REF`].
    pub s_ref: f64,
    /// Scout intel floor (e/t) — [`EPSILON_INTEL_E_T`].
    pub epsilon_intel: f64,
    /// Worker sink-value multiplier — [`V_SINK`].
    pub v_sink: f64,
}

impl Default for PolicyParams {
    fn default() -> Self {
        PolicyParams {
            t_ramp: T_RAMP,
            s_ref: S_REF,
            epsilon_intel: EPSILON_INTEL_E_T,
            v_sink: V_SINK,
        }
    }
}

/// What a WORK part converts per tick, by job — selects the §D5.4 worker `k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkKind {
    /// `k = BUILD_POWER` (5 e/t/WORK).
    Build,
    /// `k = UPGRADE_CONTROLLER_POWER` (1 e/t/WORK).
    Upgrade,
}

impl WorkKind {
    fn energy_per_work_tick(self) -> f64 {
        match self {
            WorkKind::Build => BUILD_POWER_E_T,
            WorkKind::Upgrade => UPGRADE_POWER_E_T,
        }
    }
}

/// A creep's role/goal, carrying only role-LOCAL parameters; fleet/squad economics arrive via
/// [`GoalAnnotation`] (decision (5)). The §D5.4 role table in full.
#[derive(Clone, Debug)]
pub enum Role {
    /// A hauler cycling `q` cargo units per round trip (Q = 50 × CARRY parts loaded, or the
    /// scenario-pinned value-units for non-energy cargo — decision (6)).
    Haul { q: u32 },
    /// Squad melee (ATTACK) — priced on the squad rail via `GoalAnnotation::squad`.
    Melee,
    /// Squad ranged (RANGED_ATTACK) — squad rail.
    Ranged,
    /// Squad healer (HEAL) — squad rail. Heal→energy converts ONLY through the objective it
    /// unlocks (`R_O`), never a per-HP energy constant (§D5.4 role table, deliberately).
    Heal,
    /// Squad dismantler (WORK@`DISMANTLE_POWER`) — squad rail.
    Dismantle,
    /// Builder/upgrader: `min(WORK·k, supply_rate)·V_SINK`, WORK counted live from the body (so a
    /// chewed-up worker's bid degrades with its alive parts).
    Work { kind: WorkKind, supply_rate_e_t: f64 },
    /// Claimer: the hard reach gate + hazard-smoothed stock rate (its own rail — A/U do not apply).
    Claim,
    /// Scout: `max(EPSILON_INTEL, upkeep)` — decision (3)'s declared policy constant.
    Scout,
}

impl Role {
    /// The t_star rail: the productive rate `r` in e/t for roles priced entirely by the
    /// fatigue-exact optimal cycle duration (`t_star` — for a hauler, the oracle round-trip time).
    /// **Haul only**; every other role is annotation/body-priced by [`movement_intent_weight`] and
    /// returns 0 here (military sample weights come from [`squad_sample_weight`], decision (1)).
    pub fn rate_e_t(&self, t_star: u32) -> f64 {
        match self {
            // ρ = Q / T*_rtt — room_net_roi's haul term inverted. Saturation caps
            // (marginal_route_share) arrive via the scenario annotation when fleets share a route.
            Role::Haul { q } => {
                if t_star == 0 {
                    0.0
                } else {
                    *q as f64 / t_star as f64
                }
            }
            _ => 0.0,
        }
    }

    /// The benchmark sample weight for one movement episode of optimal duration `t_star`:
    /// `W = r · T*` — the energy-tick value in flight over one optimal episode. For a hauler this
    /// is exactly `Q` (the cargo), making the aggregate `H` the operator's cargo-weighted
    /// "value transport by distance" (§D5.4 hauler reduction). Squad members use
    /// [`squad_sample_weight`] instead (cost-shares summing each job's `V_O` exactly once).
    pub fn sample_weight(&self, t_star: u32) -> f64 {
        self.rate_e_t(t_star) * t_star as f64
    }
}

/// The squad-objective context a military creep's annotation carries — pre-priced by the harness
/// from the landed kernels (decision (5)): `p_win`/`est_ticks` from `force_sizing`'s oracle,
/// `value_e` from `objective_value`/`room_net_roi` (the economic unlock, per the
/// combat-EV-economic directive — never force-elimination), `alpha_class` = this member's
/// capability class's spawn-cost share of the REQUIRED force (`RequiredForce`,
/// combat-decision/src/force_sizing.rs:499).
#[derive(Clone, Debug)]
pub struct SquadRef {
    /// Lanchester win probability for the squad's objective.
    pub p_win: f64,
    /// Energy-equivalent value of the objective's unlock (stock).
    pub value_e: f64,
    /// Estimated ticks to convert the objective once on site.
    pub est_ticks: u32,
    /// Spawn-cost share of this member's capability class within the required force (Σ over
    /// classes = 1 for a fully-specified force).
    pub alpha_class: f64,
    /// Required power of this member's class (e.g. Σ needed HEAL/tick), from `RequiredForce`.
    pub required_class_power: f64,
    /// Total FIELDED power of the class — the renormalization denominator: an overfilled class
    /// dilutes each member so the class never integrates more than `alpha_class` (≤ 1 per class).
    pub fielded_class_power: f64,
    /// This member is the squad's critical-path laggard (max ttr): it bids the full `R_O` under
    /// contention; everyone else bids the upkeep floor (decision (1)).
    pub binding: bool,
}

impl SquadRef {
    /// `V_O = p_win · value_e` — the expected stock the objective is worth (e).
    pub fn objective_value_e(&self) -> f64 {
        self.p_win * self.value_e
    }

    /// `R_O = p_win · value_e / est_ticks` — the squad's objective rate (e/t): what one tick of
    /// delay to the WHOLE squad costs.
    pub fn objective_rate(&self) -> f64 {
        if self.est_ticks == 0 {
            0.0
        } else {
            self.objective_value_e() / self.est_ticks as f64
        }
    }
}

/// The per-creep goal annotation the scenario harness precomputes (§D5.4). Everything economic is
/// pre-priced (decision (5)); `w` is then a pure function of (creep, role, annotation, ttr, ttl,
/// threat) — no kernel forks, no map iteration, no clock.
#[derive(Clone, Debug, Default)]
pub struct GoalAnnotation {
    /// The pre-priced productive rate for self-contained economic roles (haul: `Q/T*_rtt`, already
    /// capped by `marginal_route_share` when fleets share a route). Ignored by annotation-priced
    /// roles (military/claim/scout compute their own rails).
    pub rate_e_t: f64,
    /// The stock at risk aboard/embodied (e): a hauler's cargo value, a claimer's `claim_value`,
    /// a squad member's cost-share of `V_O`. Feeds the claim rail + the escape bid.
    pub value_stock_e: f64,
    /// Minimum on-site ticks for arrival to matter (the A gate's usefulness floor; for a squad,
    /// pre-engagement forming/positioning time).
    pub t_min: u32,
    /// External deadline (absolute ticks-from-now), if any — e.g. a controller downgrade timer for
    /// a claimer. Tightens the claim rail's effective window.
    pub deadline: Option<u32>,
    /// Squad-objective context for military roles (None for economic roles or a squadless
    /// straggler, which bids the upkeep floor).
    pub squad: Option<SquadRef>,
    /// Energy recoverable by recycling instead of escaping (e) — the escape bid prices only the
    /// SURPLUS over recycling (`max(0, V − recycle_ev)`).
    pub recycle_ev: f64,
}

/// The caller's threat view at the creep's position — already net of squad heal (pre-priced, the
/// same `threat_field − squad_heal` shape as kite.rs:712).
#[derive(Clone, Copy, Debug)]
pub struct ThreatView {
    /// Net incoming damage per tick after friendly heal (0 = safe).
    pub net_incoming_per_tick: u32,
}

/// Spawn cost of the body (e) — `Σ Part::cost()` (screeps-game-api's engine `BODYPART_COST` table;
/// boost minerals are lab-side capital and deliberately excluded — squad shares arrive pre-priced).
pub fn body_cost_e(body: &SimBody) -> u32 {
    body.parts.iter().map(|p| p.part.cost()).sum()
}

/// The amortized upkeep rate `body_cost / CREEP_LIFE_TIME` (e/t) — the §D5.4 slack floor
/// (decision (2)) and the non-binding squad member's contention bid (decision (1)).
pub fn upkeep_e_t(body: &SimBody) -> f64 {
    body_cost_e(body) as f64 / CREEP_LIFE_TIME as f64
}

/// This member's power within its capability class (boost-aware, alive-parts-only via
/// `SimBody::effective_power` — the engine's `calcBodyEffectiveness`): melee ATTACK·30,
/// ranged RANGED_ATTACK·10, heal HEAL·12, dismantle WORK·50. 0 for non-military roles.
pub fn class_power(role: &Role, body: &SimBody) -> f64 {
    match role {
        Role::Melee => body.effective_power(Part::Attack, ATTACK_POWER) as f64,
        Role::Ranged => body.effective_power(Part::RangedAttack, RANGED_ATTACK_POWER) as f64,
        Role::Heal => body.effective_power(Part::Heal, HEAL_POWER) as f64,
        Role::Dismantle => body.effective_power(Part::Work, DISMANTLE_POWER) as f64,
        _ => 0.0,
    }
}

/// The squad member's benchmark SAMPLE weight (decision (1)): `α_class · s_member · V_O`, where
/// `s_member = own_power / max(required, fielded)` — renormalized so each class integrates at most
/// `α_class` (an overfilled class dilutes its members; an underfilled one leaves the shortfall
/// unclaimed) and a fully-specified squad's weights sum to exactly `V_O`, once.
pub fn squad_sample_weight(creep: &SimCreep, role: &Role, squad: &SquadRef) -> f64 {
    let denom = squad.required_class_power.max(squad.fielded_class_power);
    if denom <= 0.0 {
        return 0.0;
    }
    squad.alpha_class * (class_power(role, &creep.body) / denom) * squad.objective_value_e()
}

/// A — the arrive-in-time ramp `clamp((ttl − ttr − t_min)/T_RAMP, 0, 1)` (§D5.4): a creep that can
/// no longer arrive alive-and-useful has `w = 0`; ramped, not stepped, to avoid the binary-P(win)
/// cliff. Exactly 0 when `ttl ≤ ttr + t_min` (the property-test boundary).
pub fn gate_a(ttl: u32, ttr: u32, t_min: u32) -> f64 {
    gate_a_with(&PolicyParams::default(), ttl, ttr, t_min)
}

/// [`gate_a`] under explicit [`PolicyParams`] (the sensitivity-sweep entry).
pub fn gate_a_with(params: &PolicyParams, ttl: u32, ttr: u32, t_min: u32) -> f64 {
    ((ttl as f64 - ttr as f64 - t_min as f64) / params.t_ramp).clamp(0.0, 1.0)
}

/// S — the survival veto (kite.rs:704-714 generalized): `net_incoming × SURVIVAL_HORIZON > hits`
/// ⇒ 0 (progress you don't survive to convert is not progress; the escape bid takes over). The
/// LIVE strict `>` of kite.rs:713 (ratified 2026-07-01): exactly-lethal survives — see the
/// decisions block. No threat view ⇒ 1.
pub fn gate_s(threat: Option<&ThreatView>, effective_hits: u32) -> f64 {
    match threat {
        Some(t)
            if t.net_incoming_per_tick > 0
                && t.net_incoming_per_tick.saturating_mul(SURVIVAL_HORIZON) > effective_hits =>
        {
            0.0
        }
        _ => 1.0,
    }
}

/// U — the slack gate (§D5.4): delay costs `r` only while the window binds. Continuous-cycle roles
/// (Haul/Work/Scout) bind by construction (their job consumes the whole remaining life) ⇒ 1.
/// Military roles bind while `ttl − ttr ≤ t_min + est_ticks` (arrive + serve out the objective);
/// past that, `U = max(floor_ratio, 1/(1 + excess/T_RAMP))` — decaying toward the amortized-upkeep
/// floor `min(1, upkeep/r)` (decision (2): never 0 — an alive body is depreciating capital). A
/// member already bidding the upkeep floor gets `floor_ratio == 1` ⇒ `U == 1` (no double-flooring).
pub fn gate_u(role: &Role, creep: &SimCreep, ann: &GoalAnnotation, ttr: u32, ttl: u32, r: f64) -> f64 {
    gate_u_with(&PolicyParams::default(), role, creep, ann, ttr, ttl, r)
}

/// [`gate_u`] under explicit [`PolicyParams`] (the decay scale is `t_ramp`).
pub fn gate_u_with(
    params: &PolicyParams,
    role: &Role,
    creep: &SimCreep,
    ann: &GoalAnnotation,
    ttr: u32,
    ttl: u32,
    r: f64,
) -> f64 {
    let horizon_needed = match (role, &ann.squad) {
        (Role::Melee | Role::Ranged | Role::Heal | Role::Dismantle, Some(sq)) => {
            ann.t_min.saturating_add(sq.est_ticks)
        }
        // Continuous-cycle roles (and squadless military, already floored at upkeep): binding.
        _ => return 1.0,
    };
    let window = ttl.saturating_sub(ttr);
    if window <= horizon_needed || r <= 0.0 {
        return 1.0;
    }
    let excess = (window - horizon_needed) as f64;
    let decay = 1.0 / (1.0 + excess / params.t_ramp);
    let floor_ratio = (upkeep_e_t(&creep.body) / r).min(1.0);
    decay.max(floor_ratio)
}

/// The claim rail (§D5.4 role table, its own gate — A/U do not apply): the hard reach gate
/// `ttr + CLAIM_ARRIVAL_MARGIN ≤ min(ttl, 600, deadline)` (the utility.rs:94 shape), then the
/// hazard-smoothed stock rate `min(V, V/max(slack, S_REF))` — deadline-tight claimers explode in
/// priority (capped at V), unreachable ones drop to exactly 0.
fn claim_rate(params: &PolicyParams, ann: &GoalAnnotation, ttr: u32, ttl: u32) -> f64 {
    let effective_deadline = ttl
        .min(CREEP_CLAIM_LIFE_TIME)
        .min(ann.deadline.unwrap_or(u32::MAX));
    let arrival = match ttr.checked_add(CLAIM_ARRIVAL_MARGIN) {
        Some(a) if a <= effective_deadline => a,
        _ => return 0.0, // reach gate failed (or ttr overflow — unreachable either way)
    };
    let slack = (effective_deadline - arrival) as f64; // ≥ 0: the gate passed
    let v = ann.value_stock_e;
    v.min(v / slack.max(params.s_ref))
}

/// The role's contention bid rate `r_bid` (e/t), before the external gates. The §D5.4 role table:
/// haul = the pre-priced `Q/T*` (annotation); worker = `min(WORK·k, supply)·V_SINK` from the live
/// body; scout = `max(ε_intel, upkeep)`; military = binding member bids the full `R_O`, everyone
/// else (and any squadless straggler) the upkeep floor (decision (1)); claim = its own rail.
fn progress_rate(
    params: &PolicyParams,
    creep: &SimCreep,
    role: &Role,
    ann: &GoalAnnotation,
    ttr: u32,
    ttl: u32,
) -> f64 {
    match role {
        Role::Haul { .. } => ann.rate_e_t,
        Role::Work { kind, supply_rate_e_t } => {
            let work = creep.body.alive_part_count(Part::Work) as f64;
            (work * kind.energy_per_work_tick()).min(*supply_rate_e_t) * params.v_sink
        }
        Role::Scout => upkeep_e_t(&creep.body).max(params.epsilon_intel),
        Role::Melee | Role::Ranged | Role::Heal | Role::Dismantle => match &ann.squad {
            Some(sq) if sq.binding => sq.objective_rate(),
            _ => upkeep_e_t(&creep.body),
        },
        Role::Claim => claim_rate(params, ann, ttr, ttl),
    }
}

/// The escape bid (§D5.4): under fire, moving away prices the stock-drawdown RATE
/// `max(0, V − recycle_ev)/τ_die` with `τ_die = hits/net_incoming` and the wounded discount
/// `V = value_stock_e · hits/hits_max` (decision (8)). 0 when not under net fire.
pub fn escape_weight(creep: &SimCreep, ann: &GoalAnnotation, threat: Option<&ThreatView>) -> f64 {
    let net = threat.map_or(0, |t| t.net_incoming_per_tick);
    if net == 0 || creep.body.hits == 0 {
        return 0.0;
    }
    let hits_share = creep.body.hits as f64 / creep.body.hits_max() as f64;
    let v = ann.value_stock_e * hits_share;
    let surplus = (v - ann.recycle_ev).max(0.0);
    let tau_die = creep.body.hits as f64 / net as f64;
    surplus / tau_die
}

/// `w(creep)` — the §D5.4 scalar: the marginal energy-equivalent value destroyed if this creep's
/// one movement intent is denied this tick. `w = max(Δ_crit·A·S·U·r_bid, w_escape)` (decision (8));
/// the claim rail replaces A/U with its own hard reach gate. `delta_crit` is caller-supplied: did
/// the move reduce the fatigue-exact `ttr` by ≥ 1 (0 if fatigued/lateral/off-path/in-position —
/// the in-transit derivative; holding a slot is priced by the caller passing its hold Δ). Quantize
/// via [`ordering_key`] before ANY ordering (the determinism fence).
pub fn movement_intent_weight(
    creep: &SimCreep,
    role: &Role,
    ann: &GoalAnnotation,
    ttr: u32,
    ttl: u32,
    delta_crit: bool,
    threat: Option<&ThreatView>,
) -> f64 {
    movement_intent_weight_with(&PolicyParams::default(), creep, role, ann, ttr, ttl, delta_crit, threat)
}

/// [`movement_intent_weight`] under explicit [`PolicyParams`] — the decision-(3) sensitivity-sweep
/// entry point. The default-params path is bit-identical to [`movement_intent_weight`] (pinned by
/// the smoke test); everything except the four policy constants is shared code.
#[allow(clippy::too_many_arguments)] // the §D5.4 signature + one params handle; a struct would hide the contract
pub fn movement_intent_weight_with(
    params: &PolicyParams,
    creep: &SimCreep,
    role: &Role,
    ann: &GoalAnnotation,
    ttr: u32,
    ttl: u32,
    delta_crit: bool,
    threat: Option<&ThreatView>,
) -> f64 {
    let delta = if delta_crit { 1.0 } else { 0.0 };
    let s = gate_s(threat, creep.body.hits);
    let r = progress_rate(params, creep, role, ann, ttr, ttl);
    let (a, u) = match role {
        // The reach gate inside the claim rail replaces A/U (§D5.4 role table, "its own rail").
        Role::Claim => (1.0, 1.0),
        _ => (
            gate_a_with(params, ttl, ttr, ann.t_min),
            gate_u_with(params, role, creep, ann, ttr, ttl, r),
        ),
    };
    let w_progress = delta * a * s * u * r;
    // Escape is NOT Δ_crit-gated: fleeing is lateral by nature; any intent spent escaping bids the
    // drawdown rate (and is exactly what takes over when S vetoes progress).
    w_progress.max(escape_weight(creep, ann, threat))
}

/// Quantize a weight to integer milli-e/t — every ordering over weights MUST compare these (never
/// raw f64) per the determinism fence; ties break on the stable creep id via [`ordering_key`].
pub fn quantize_w(w: f64) -> i64 {
    (w * 1000.0).round() as i64
}

/// The total order for contention triage: quantized weight first (descending by caller), stable
/// creep id as the deterministic tie-break — no float compare, no map-iteration order, anywhere.
pub fn ordering_key(w: f64, id: CreepId) -> (i64, CreepId) {
    (quantize_w(w), id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use screeps::{Position, RoomCoordinate, RoomName};

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

    /// A squad ref with `V_O = 0.8·45000 = 36000` (exact in f64) and `R_O = 36000/900 = 40` exact.
    fn squad(alpha: f64, required: f64, fielded: f64, binding: bool) -> SquadRef {
        SquadRef {
            p_win: 0.8,
            value_e: 45000.0,
            est_ticks: 900,
            alpha_class: alpha,
            required_class_power: required,
            fielded_class_power: fielded,
            binding,
        }
    }

    #[test]
    fn hauler_rate_is_cargo_over_optimal_round_trip() {
        let role = Role::Haul { q: 100 };
        assert!((role.rate_e_t(50) - 2.0).abs() < 1e-9, "100 cargo / 50-tick cycle = 2 e/t");
        assert_eq!(role.rate_e_t(0), 0.0, "a degenerate zero-length cycle prices at 0");
    }

    #[test]
    fn hauler_sample_weight_reduces_to_cargo() {
        // W = r·T* = (Q/T*)·T* = Q — the §D5.4 hauler reduction, exactly.
        let role = Role::Haul { q: 150 };
        for t_star in [10u32, 37, 400] {
            assert!(
                (role.sample_weight(t_star) - 150.0).abs() < 1e-9,
                "the hauler sample weight is its cargo, independent of route length"
            );
        }
    }

    #[test]
    fn hauler_reduction_w_is_exactly_q_over_t_star_all_gates_open() {
        // The §D5.4 hauler reduction through the FULL w pipeline: Δ=1, A=1 (ttl−ttr−t_min ≫ 20),
        // S=1 (no threat), U=1 (continuous-cycle) ⇒ w == ρ == Q/T*, bit-exact.
        let role = Role::Haul { q: 100 };
        let t_star = 50;
        let ann = GoalAnnotation { rate_e_t: role.rate_e_t(t_star), t_min: 1, ..Default::default() };
        let hauler = creep(1, &[Part::Carry, Part::Carry, Part::Move]);
        let w = movement_intent_weight(&hauler, &role, &ann, 200, 1500, true, None);
        assert_eq!(w, 2.0, "w degenerates to Q/T* = 100/50 with every gate open");
        // Δ_crit = 0 (lateral / fatigued / in-position) zeroes the progress bid entirely.
        assert_eq!(movement_intent_weight(&hauler, &role, &ann, 200, 1500, false, None), 0.0);
    }

    #[test]
    fn squad_cost_share_weights_sum_to_exactly_v_o_once() {
        // Two classes, α .5 + .5, each exactly filled by two members ⇒ Σ W == V_O == 36000, exact
        // (all shares dyadic: 30/60 and 12/24 are 0.5).
        let melee = creep(1, &[Part::Attack, Part::Move]); // ATTACK power 30
        let healer = creep(2, &[Part::Heal, Part::Move]); // HEAL power 12
        let sq_melee = squad(0.5, 60.0, 60.0, false);
        let sq_heal = squad(0.5, 24.0, 24.0, false);
        let sum = squad_sample_weight(&melee, &Role::Melee, &sq_melee) * 2.0
            + squad_sample_weight(&healer, &Role::Heal, &sq_heal) * 2.0;
        assert_eq!(sum, 36000.0, "a fully-specified squad's sample weights integrate V_O once");

        // An OVERFILLED class renormalizes (s = own/max(required, fielded)): 4 healers of 12
        // against required 24 ⇒ each s = 12/48, class still integrates exactly α·V_O.
        let sq_over = squad(0.5, 24.0, 48.0, false);
        let over_sum = squad_sample_weight(&healer, &Role::Heal, &sq_over) * 4.0;
        assert_eq!(over_sum, 0.5 * 36000.0, "overfill dilutes members, never double-counts V_O");

        // An UNDERFILLED class leaves the shortfall unclaimed: one healer of 12 against 24.
        let sq_under = squad(0.5, 24.0, 12.0, false);
        assert_eq!(
            squad_sample_weight(&healer, &Role::Heal, &sq_under),
            0.25 * 36000.0,
            "underfill claims only the fielded fraction of the class share"
        );
    }

    #[test]
    fn binding_member_bids_full_r_o_and_non_binding_bids_the_upkeep_floor() {
        // Gates open: ttl 1000, ttr 100, t_min 50 ⇒ A = 42.5-clamped = 1; window 900 ≤ horizon
        // 50+900 = 950 ⇒ U = 1 (binding window); no threat ⇒ S = 1.
        let ann_binding = GoalAnnotation {
            t_min: 50,
            squad: Some(squad(0.4, 60.0, 60.0, true)),
            ..Default::default()
        };
        let melee = creep(1, &[Part::Attack, Part::Move]); // body cost 80+50 = 130
        let w_bind = movement_intent_weight(&melee, &Role::Melee, &ann_binding, 100, 1000, true, None);
        assert_eq!(w_bind, 40.0, "the critical-path member bids the full R_O = 36000/900");

        let ann_other = GoalAnnotation {
            t_min: 50,
            squad: Some(squad(0.6, 24.0, 24.0, false)),
            ..Default::default()
        };
        let healer = creep(2, &[Part::Heal, Part::Move]); // body cost 250+50 = 300
        let w_other = movement_intent_weight(&healer, &Role::Heal, &ann_other, 100, 1000, true, None);
        assert_eq!(w_other, 300.0 / 1500.0, "everyone else bids body_cost/1500 (decision (2))");
    }

    #[test]
    fn slack_rich_binding_member_decays_toward_upkeep_but_never_below() {
        // window 1500 > horizon 950: excess 550 ⇒ U = 1/(1 + 550/20) = 1/28.5, floored at
        // upkeep/R_O — the ADR's slack-rich squad bids ~upkeep and loses tiles to a loaded hauler.
        let ann = GoalAnnotation { t_min: 50, squad: Some(squad(0.4, 60.0, 60.0, true)), ..Default::default() };
        let melee = creep(1, &[Part::Attack, Part::Move]);
        let w = movement_intent_weight(&melee, &Role::Melee, &ann, 0, 1500, true, None);
        assert!((w - 40.0 / 28.5).abs() < 1e-9, "hyperbolic decay past the binding window: {w}");
        assert!(w < 40.0 && w >= upkeep_e_t(&melee.body), "between R_O and the upkeep floor");
    }

    #[test]
    fn a_gate_zeroes_w_when_ttl_cannot_cover_ttr_plus_t_min() {
        let ann = GoalAnnotation { t_min: 50, squad: Some(squad(1.0, 60.0, 60.0, true)), ..Default::default() };
        let melee = creep(1, &[Part::Attack, Part::Move]);
        // ttl == ttr + t_min exactly ⇒ A = clamp(0/20) = 0 ⇒ w = 0 (the boundary is dead).
        assert_eq!(gate_a(150, 100, 50), 0.0);
        assert_eq!(movement_intent_weight(&melee, &Role::Melee, &ann, 100, 150, true, None), 0.0);
        // One ramp width of headroom ⇒ A = 1 again.
        assert_eq!(gate_a(170, 100, 50), 1.0);
    }

    #[test]
    fn s_veto_zeroes_progress_under_lethal_incoming_and_escape_takes_over() {
        // 10 parts ⇒ 1000 hits. The LIVE strict `>` boundary (kite.rs:713, ratified 2026-07-01):
        // net 334: 334×3 = 1002 > 1000 ⇒ lethal; net 333: 999 not > 1000 ⇒ survives.
        let body: Vec<Part> = std::iter::repeat_n(Part::Attack, 9).chain([Part::Move]).collect();
        let melee = creep(1, &body);
        let lethal = ThreatView { net_incoming_per_tick: 334 };
        let survivable = ThreatView { net_incoming_per_tick: 333 };
        assert_eq!(gate_s(Some(&lethal), melee.body.hits), 0.0);
        assert_eq!(gate_s(Some(&survivable), melee.body.hits), 1.0);
        // The strict boundary pinned at EXACT equality: 3 parts ⇒ 300 hits, net 100 ⇒ 100×3 == 300
        // — not strictly greater ⇒ SURVIVES (live parity; `≥` would have vetoed); one more point
        // of net (101×3 = 303 > 300) flips it lethal.
        let boundary = creep(2, &[Part::Attack, Part::Attack, Part::Move]);
        assert_eq!(gate_s(Some(&ThreatView { net_incoming_per_tick: 100 }), boundary.body.hits), 1.0);
        assert_eq!(gate_s(Some(&ThreatView { net_incoming_per_tick: 101 }), boundary.body.hits), 0.0);

        // No stock aboard ⇒ no escape surplus ⇒ w == 0 under lethal fire: pure veto.
        let ann = GoalAnnotation { t_min: 50, squad: Some(squad(1.0, 270.0, 270.0, true)), ..Default::default() };
        assert_eq!(movement_intent_weight(&melee, &Role::Melee, &ann, 100, 1000, true, Some(&lethal)), 0.0);

        // With stock: escape bids the drawdown rate (V − recycle)/τ_die = 3000·334/1000 = 1002 e/t
        // (full-health ⇒ no wounded discount), overriding the vetoed progress bid (decision (8)).
        let ann_stock = GoalAnnotation { value_stock_e: 3000.0, ..ann };
        let w = movement_intent_weight(&melee, &Role::Melee, &ann_stock, 100, 1000, true, Some(&lethal));
        assert!((w - 1002.0).abs() < 1e-9, "escape prices the stock-drawdown rate: {w}");
        // The wounded discount: at half hits, V halves AND τ_die halves ⇒ the rate is unchanged in
        // this construction, but the recycle floor bites sooner.
        let ann_recycled = GoalAnnotation { recycle_ev: 3000.0, ..ann_stock };
        assert_eq!(
            movement_intent_weight(&melee, &Role::Melee, &ann_recycled, 100, 1000, true, Some(&lethal)),
            0.0,
            "no surplus over recycling ⇒ no escape bid"
        );
    }

    #[test]
    fn claimer_reach_gate_is_exact_at_the_boundary() {
        let claimer = creep(1, &[Part::Claim, Part::Move]);
        // ttl 1500 caps at CREEP_CLAIM_LIFE_TIME 600: ttr 550 + margin 50 (the LIVE utility.rs:66
        // margin, ratified 2026-07-01) == 600 ⇒ passes, slack 0 ⇒ w = min(V, V/max(0, S_REF)) =
        // 5000/100 = 50.
        let ann = GoalAnnotation { value_stock_e: 5000.0, ..Default::default() };
        let w = movement_intent_weight(&claimer, &Role::Claim, &ann, 550, 1500, true, None);
        assert_eq!(w, 50.0, "at the exact boundary the claimer still bids (hazard-smoothed)");
        // One tick past ⇒ the hard gate drops it to exactly 0.
        assert_eq!(movement_intent_weight(&claimer, &Role::Claim, &ann, 551, 1500, true, None), 0.0);
        // An external deadline tightens the window below the 600 lifetime: ttr + 50 ≤ 550.
        let ann_dl = GoalAnnotation { deadline: Some(550), ..ann.clone() };
        assert!(movement_intent_weight(&claimer, &Role::Claim, &ann_dl, 500, 1500, true, None) > 0.0);
        assert_eq!(movement_intent_weight(&claimer, &Role::Claim, &ann_dl, 501, 1500, true, None), 0.0);
    }

    #[test]
    fn scout_and_worker_rails_price_as_declared() {
        // Scout: max(ε_intel, upkeep). A 1×MOVE scout IS the ε floor (50/1500).
        let scout = creep(1, &[Part::Move]);
        let ann = GoalAnnotation { t_min: 0, ..Default::default() };
        let w_scout = movement_intent_weight(&scout, &Role::Scout, &ann, 10, 1500, true, None);
        assert_eq!(w_scout, EPSILON_INTEL_E_T, "the declared intel floor (decision (3))");

        // Worker: min(WORK·k, supply)·v_sink — 2×WORK builder = min(10, 7) = 7 e/t supply-capped.
        let builder = creep(2, &[Part::Work, Part::Work, Part::Carry, Part::Move]);
        let role = Role::Work { kind: WorkKind::Build, supply_rate_e_t: 7.0 };
        let w_build = movement_intent_weight(&builder, &role, &ann, 10, 1500, true, None);
        assert_eq!(w_build, 7.0, "supply-capped build rate");
        // The same body upgrading: k = 1 ⇒ WORK·k = 2 binds below the supply cap.
        let role_up = Role::Work { kind: WorkKind::Upgrade, supply_rate_e_t: 7.0 };
        assert_eq!(movement_intent_weight(&builder, &role_up, &ann, 10, 1500, true, None), 2.0);
    }

    /// Decision (3)'s sweep handle must be inert at its default: `PolicyParams::default()` IS the
    /// decided constants (bit-equal), and the `_with` path reproduces `movement_intent_weight`
    /// bit-for-bit across every role rail (so the sweep probes ONLY what it claims to probe).
    #[test]
    fn default_policy_params_are_the_decided_constants_bit_for_bit() {
        let p = PolicyParams::default();
        assert_eq!(p.t_ramp.to_bits(), T_RAMP.to_bits());
        assert_eq!(p.s_ref.to_bits(), S_REF.to_bits());
        assert_eq!(p.epsilon_intel.to_bits(), EPSILON_INTEL_E_T.to_bits());
        assert_eq!(p.v_sink.to_bits(), V_SINK.to_bits());

        // One fixture per policy-touched rail: hauler (t_ramp via A), binding+slack-rich squad
        // (t_ramp via U decay), claimer (s_ref), scout (epsilon_intel), worker (v_sink), escaper
        // (no params — must still match).
        type Case = (SimCreep, Role, GoalAnnotation, u32, u32, bool, Option<ThreatView>);
        let lethal = ThreatView { net_incoming_per_tick: 400 };
        let cases: Vec<Case> = vec![
            (
                creep(1, &[Part::Carry, Part::Carry, Part::Move]),
                Role::Haul { q: 100 },
                GoalAnnotation { rate_e_t: 2.0, t_min: 1, ..Default::default() },
                200, 1500, true, None,
            ),
            (
                creep(2, &[Part::Attack, Part::Move]),
                Role::Melee,
                GoalAnnotation { t_min: 50, squad: Some(squad(0.4, 60.0, 60.0, true)), ..Default::default() },
                0, 1500, true, None, // slack-rich: exercises the U decay's t_ramp
            ),
            (
                creep(3, &[Part::Claim, Part::Move]),
                Role::Claim,
                GoalAnnotation { value_stock_e: 5000.0, ..Default::default() },
                550, 1500, true, None, // slack 0 (ttr 550 + the live margin 50 == 600): exercises s_ref
            ),
            (
                creep(4, &[Part::Move]),
                Role::Scout,
                GoalAnnotation::default(),
                10, 1500, true, None,
            ),
            (
                creep(5, &[Part::Work, Part::Work, Part::Carry, Part::Move]),
                Role::Work { kind: WorkKind::Build, supply_rate_e_t: 7.0 },
                GoalAnnotation::default(),
                10, 1500, true, None,
            ),
            (
                creep(6, &[Part::Attack, Part::Move]),
                Role::Melee,
                GoalAnnotation { value_stock_e: 3000.0, ..Default::default() },
                100, 1000, true, Some(lethal),
            ),
        ];
        for (c, role, ann, ttr, ttl, delta, threat) in &cases {
            let w = movement_intent_weight(c, role, ann, *ttr, *ttl, *delta, threat.as_ref());
            let w_with = movement_intent_weight_with(&p, c, role, ann, *ttr, *ttl, *delta, threat.as_ref());
            assert_eq!(
                w.to_bits(),
                w_with.to_bits(),
                "default-params `_with` must be bit-identical (creep {})",
                c.id
            );
        }
    }

    #[test]
    fn quantized_ordering_is_deterministic() {
        // Two weights closer than 1 milli-e/t collapse to the same quantum; the id breaks the tie.
        let a = ordering_key(1.2340004, 7);
        let b = ordering_key(1.2340001, 3);
        assert_eq!(a.0, b.0, "sub-milli differences are noise, not order");
        assert_ne!(a, b, "the stable id still yields a total order");
        assert!(ordering_key(2.0, 1).0 > ordering_key(1.999, 9).0);
        // The contention rail end-to-end: a binding squad member outranks a loaded hauler outranks
        // a slack-rich straggler — compared ONLY through quantized keys (the fence).
        let keys = [ordering_key(40.0, 1), ordering_key(2.0, 2), ordering_key(0.2, 3)];
        assert!(keys[0] > keys[1] && keys[1] > keys[2]);
    }
}
