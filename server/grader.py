# grader.py
# ──────────────────────────────────────────────────────────
# Reward engineering for ContentGuard.
# Provides shaped mid-episode rewards (evidence discovery,
# conflict resolution, duplicate penalties) and a terminal
# reward matrix with efficiency bonus.
# ──────────────────────────────────────────────────────────

VERDICT_BINS: dict[str, tuple[float, float]] = {
    "remove":   (0.00, 0.30),
    "monetize": (0.30, 0.45),
    "escalate": (0.45, 0.65),
    "keep":     (0.65, 1.00),
}

REWARD_MATRIX: dict[tuple[str, str], float] = {
    ("remove",   "remove"):   +1.00,
    ("remove",   "monetize"): -0.50,
    ("remove",   "escalate"): +0.20,
    ("remove",   "keep"):     -1.00,  # catastrophic — infringing content stays live

    ("monetize", "monetize"): +1.00,
    ("monetize", "remove"):   +0.30,
    ("monetize", "escalate"): +0.60,
    ("monetize", "keep"):     -0.40,

    ("escalate", "escalate"): +1.00,
    ("escalate", "remove"):   +0.20,
    ("escalate", "monetize"): +0.10,
    ("escalate", "keep"):     +0.20,

    ("keep",     "keep"):     +1.00,
    ("keep",     "escalate"): +0.60,
    ("keep",     "monetize"): -0.30,
    ("keep",     "remove"):   -0.80,  # wrongful takedown
}

# ── Reward constants ───────────────────────────────────────
STEP_COST = -0.02
EVIDENCE_DISCOVERY_BONUS = 0.05
CONFLICT_RESOLUTION_BONUS = 0.08
DUPLICATE_ACTION_PENALTY = -0.05
EFFICIENCY_BONUS = 0.10

# ── Per-archetype critical actions ─────────────────────────
# Which investigation actions reveal the most decision-relevant
# evidence for each archetype, derived from the 4-factor
# fair-use rubric weights and archetype-specific fields.
ARCHETYPE_CRITICAL_ACTIONS: dict[str, set[str]] = {
    "verbatim_commercial":            {"assess_transformation", "query_rights_db"},
    "commentary_clip_noncommercial":  {"assess_transformation", "check_usage_context"},
    "parody_high_overlap":            {"assess_transformation", "check_fingerprint"},
    "educational_excerpt":            {"assess_transformation", "check_usage_context"},
    "background_music_commercial":    {"assess_transformation", "check_usage_context"},
    "expired_license_disputed":       {"query_rights_db", "check_usage_context"},
    "multi_claimant_non_overlapping": {"query_rights_db", "cross_ref_history"},
    "orphaned_work":                  {"query_rights_db", "check_fingerprint"},
    "creative_commons_misapplication": {"query_rights_db", "assess_transformation"},
    "transformative_large_amount":    {"assess_transformation", "check_fingerprint"},
    "noncommercial_direct_substitute": {"assess_transformation", "check_usage_context"},
    "educational_verbatim_complete":  {"assess_transformation", "check_usage_context"},
    "live_sports_gameplay_disguise":  {"check_fingerprint", "query_rights_db"},
    "ai_audio_reconstruction":        {"check_fingerprint", "assess_transformation"},
}

# ── Optimal step counts per archetype ──────────────────────
# Minimum investigation actions + decide for a perfect agent.
ARCHETYPE_OPTIMAL_STEPS: dict[str, int] = {
    "verbatim_commercial":            3,
    "commentary_clip_noncommercial":  3,
    "parody_high_overlap":            4,
    "educational_excerpt":            3,
    "background_music_commercial":    3,
    "expired_license_disputed":       3,
    "multi_claimant_non_overlapping": 3,
    "orphaned_work":                  3,
    "creative_commons_misapplication": 3,
    "transformative_large_amount":    4,
    "noncommercial_direct_substitute": 3,
    "educational_verbatim_complete":  3,
    "live_sports_gameplay_disguise":  4,
    "ai_audio_reconstruction":        4,
}


def get_correct_bin(ground_truth: float) -> str:
    for verdict, (lo, hi) in VERDICT_BINS.items():
        if lo <= ground_truth < hi:
            return verdict
    return "keep"  # fallback for exactly 1.0


def unresolved_conflicts(state) -> bool:
    """
    Returns True if the case has conflict_flag=1 (multiple claimants)
    but the agent never called query_rights_db to resolve it.
    """
    has_conflict = state.resolved_fields.get("conflict_flag_value", 0) == 1
    investigated = "query_rights_db" in state.actions_taken
    return has_conflict and not investigated


def step_reward(action: str, case: dict, state) -> float:
    """
    Mid-episode reward with evidence-based shaping.

    Base:  -0.02 per step (budget pressure).
    +0.05  first time calling a critical investigation action.
    +0.08  for resolving a conflict via query_rights_db.
    -0.05  for duplicate investigation (same action called twice).
    """
    reward = STEP_COST

    archetype = case.get("archetype", "")
    critical = ARCHETYPE_CRITICAL_ACTIONS.get(archetype, set())

    # Duplicate action: penalize redundant investigation
    if state.actions_taken.count(action) > 1:
        reward += DUPLICATE_ACTION_PENALTY
        return round(reward, 4)

    # Evidence discovery: first time calling a critical action
    if action in critical:
        reward += EVIDENCE_DISCOVERY_BONUS

    # Conflict resolution: proactively investigating conflict
    if action == "query_rights_db" and case.get("conflict_flag", 0) == 1:
        reward += CONFLICT_RESOLUTION_BONUS

    return round(reward, 4)


def terminal_reward(
    agent_verdict: str,
    ground_truth: float,
    state,
    case: dict | None = None,
) -> float:
    correct_bin = get_correct_bin(ground_truth)
    base = REWARD_MATRIX[(correct_bin, agent_verdict)]
    process = -0.40 if unresolved_conflicts(state) else 0.0
    step_costs = STEP_COST * len(state.actions_taken)

    # Efficiency bonus: correct verdict within optimal step budget
    efficiency = 0.0
    if case and correct_bin == agent_verdict:
        archetype = case.get("archetype", "")
        optimal = ARCHETYPE_OPTIMAL_STEPS.get(archetype, 5)
        if len(state.actions_taken) <= optimal:
            efficiency = EFFICIENCY_BONUS

    # NO CLAMP — negatives are valid reward signals
    return round(base + process + step_costs + efficiency, 4)
