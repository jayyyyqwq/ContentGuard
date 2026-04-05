"""Tests for server/grader.py — reward matrix, step rewards, terminal rewards."""
import pytest
from server.grader import (
    ARCHETYPE_CRITICAL_ACTIONS,
    ARCHETYPE_OPTIMAL_STEPS,
    DUPLICATE_ACTION_PENALTY,
    EFFICIENCY_BONUS,
    EVIDENCE_DISCOVERY_BONUS,
    CONFLICT_RESOLUTION_BONUS,
    REWARD_MATRIX,
    STEP_COST,
    VERDICT_BINS,
    get_correct_bin,
    step_reward,
    terminal_reward,
    unresolved_conflicts,
)
from models import ContentGuardState


# ── helpers ────────────────────────────────────────────────────────────────
def _state(
    actions: list[str] | None = None,
    conflict_flag_value: int = 0,
    difficulty: str = "medium",
) -> ContentGuardState:
    return ContentGuardState(
        actions_taken=actions or [],
        resolved_fields={"conflict_flag_value": conflict_flag_value},
        difficulty=difficulty,
    )


# ── get_correct_bin ────────────────────────────────────────────────────────
class TestGetCorrectBin:
    @pytest.mark.parametrize("gt, expected", [
        (0.00, "remove"),
        (0.15, "remove"),
        (0.29, "remove"),
        (0.30, "monetize"),
        (0.37, "monetize"),
        (0.44, "monetize"),
        (0.45, "escalate"),
        (0.55, "escalate"),
        (0.64, "escalate"),
        (0.65, "keep"),
        (0.80, "keep"),
        (0.99, "keep"),
        (1.00, "keep"),  # fallback
    ])
    def test_bins(self, gt: float, expected: str):
        assert get_correct_bin(gt) == expected


# ── reward matrix completeness ─────────────────────────────────────────────
class TestRewardMatrix:
    VERDICTS = list(VERDICT_BINS.keys())

    def test_all_16_entries_exist(self):
        for correct in self.VERDICTS:
            for agent in self.VERDICTS:
                assert (correct, agent) in REWARD_MATRIX

    def test_correct_verdicts_give_plus_one(self):
        for v in self.VERDICTS:
            assert REWARD_MATRIX[(v, v)] == 1.0

    def test_catastrophic_negatives(self):
        assert REWARD_MATRIX[("remove", "keep")] == -1.0
        assert REWARD_MATRIX[("keep", "remove")] == -0.80


# ── unresolved_conflicts ───────────────────────────────────────────────────
class TestUnresolvedConflicts:
    def test_no_conflict(self):
        assert not unresolved_conflicts(_state(conflict_flag_value=0))

    def test_conflict_investigated(self):
        s = _state(actions=["query_rights_db"], conflict_flag_value=1)
        assert not unresolved_conflicts(s)

    def test_conflict_not_investigated(self):
        s = _state(actions=["assess_transformation"], conflict_flag_value=1)
        assert unresolved_conflicts(s)


# ── step_reward ────────────────────────────────────────────────────────────
class TestStepReward:
    def test_base_cost(self):
        case = {"archetype": "verbatim_commercial", "conflict_flag": 0}
        s = _state(actions=["cross_ref_history"])  # non-critical action
        r = step_reward("cross_ref_history", case, s)
        assert r == STEP_COST

    def test_evidence_discovery_bonus(self):
        case = {"archetype": "verbatim_commercial", "conflict_flag": 0}
        s = _state(actions=["assess_transformation"])  # critical for this archetype
        r = step_reward("assess_transformation", case, s)
        assert r == pytest.approx(STEP_COST + EVIDENCE_DISCOVERY_BONUS)

    def test_conflict_resolution_bonus(self):
        case = {"archetype": "multi_claimant_non_overlapping", "conflict_flag": 1}
        s = _state(actions=["query_rights_db"])
        r = step_reward("query_rights_db", case, s)
        # query_rights_db is critical for this archetype AND conflict_flag=1
        assert r == STEP_COST + EVIDENCE_DISCOVERY_BONUS + CONFLICT_RESOLUTION_BONUS

    def test_conflict_bonus_only_when_flag_set(self):
        case = {"archetype": "verbatim_commercial", "conflict_flag": 0}
        s = _state(actions=["query_rights_db"])
        r = step_reward("query_rights_db", case, s)
        # critical action but no conflict
        assert r == pytest.approx(STEP_COST + EVIDENCE_DISCOVERY_BONUS)

    def test_duplicate_penalty(self):
        case = {"archetype": "verbatim_commercial", "conflict_flag": 0}
        s = _state(actions=["assess_transformation", "assess_transformation"])
        r = step_reward("assess_transformation", case, s)
        assert r == STEP_COST + DUPLICATE_ACTION_PENALTY

    def test_duplicate_no_stacking_with_critical(self):
        """Duplicate penalty short-circuits — no discovery bonus on repeat."""
        case = {"archetype": "verbatim_commercial", "conflict_flag": 0}
        s = _state(actions=["assess_transformation", "assess_transformation"])
        r = step_reward("assess_transformation", case, s)
        # Should NOT include EVIDENCE_DISCOVERY_BONUS
        assert r == STEP_COST + DUPLICATE_ACTION_PENALTY


# ── terminal_reward ────────────────────────────────────────────────────────
class TestTerminalReward:
    def test_correct_verdict_no_steps(self):
        """Perfect verdict with zero investigation = base reward only."""
        s = _state(actions=["decide"], difficulty="easy")
        case = {"archetype": "verbatim_commercial"}
        r = terminal_reward("remove", 0.10, s, case)
        # base=1.0, process=0, steps=-0.02*1, efficiency=+0.10 (1 step <= 3 optimal)
        assert r == round(1.0 + 0.0 - 0.02 + 0.10, 4)

    def test_wrong_verdict(self):
        s = _state(actions=["decide"])
        r = terminal_reward("keep", 0.10, s)
        # base=-1.0, steps=-0.02
        assert r == round(-1.0 - 0.02, 4)

    def test_process_penalty(self):
        s = _state(actions=["decide"], conflict_flag_value=1)
        r = terminal_reward("escalate", 0.50, s)
        # base=1.0, process=-0.40, steps=-0.02
        assert r == round(1.0 - 0.40 - 0.02, 4)

    def test_step_cost_accumulates(self):
        actions = ["query_rights_db", "assess_transformation", "check_fingerprint", "decide"]
        s = _state(actions=actions)
        r = terminal_reward("remove", 0.10, s)
        expected_steps = STEP_COST * 4
        assert r == round(1.0 + expected_steps, 4)

    def test_efficiency_bonus_on_optimal(self):
        actions = ["assess_transformation", "query_rights_db", "decide"]
        s = _state(actions=actions, difficulty="easy")
        case = {"archetype": "verbatim_commercial"}
        r = terminal_reward("remove", 0.10, s, case)
        # 3 steps <= optimal 3 for verbatim_commercial, correct verdict
        expected = 1.0 + (STEP_COST * 3) + EFFICIENCY_BONUS
        assert r == round(expected, 4)

    def test_no_efficiency_bonus_on_wrong_verdict(self):
        actions = ["decide"]
        s = _state(actions=actions, difficulty="easy")
        case = {"archetype": "verbatim_commercial"}
        r = terminal_reward("keep", 0.10, s, case)
        # Wrong verdict — no efficiency bonus
        assert r == round(-1.0 + STEP_COST, 4)

    def test_no_efficiency_bonus_over_optimal(self):
        actions = ["query_rights_db", "assess_transformation", "check_fingerprint",
                   "check_usage_context", "cross_ref_history", "decide"]
        s = _state(actions=actions, difficulty="easy")
        case = {"archetype": "verbatim_commercial"}
        r = terminal_reward("remove", 0.10, s, case)
        # 6 steps > optimal 3 — no efficiency bonus
        expected = 1.0 + (STEP_COST * 6)
        assert r == round(expected, 4)

    def test_backward_compat_no_case(self):
        """terminal_reward still works without case param."""
        s = _state(actions=["decide"])
        r = terminal_reward("remove", 0.10, s)
        assert r == round(1.0 + STEP_COST, 4)


# ── archetype coverage ─────────────────────────────────────────────────────
class TestArchetypeCoverage:
    def test_critical_actions_cover_all_archetypes(self):
        from server.case_generator import ARCHETYPES
        for arch in ARCHETYPES:
            assert arch in ARCHETYPE_CRITICAL_ACTIONS, f"Missing critical actions for {arch}"

    def test_optimal_steps_cover_all_archetypes(self):
        from server.case_generator import ARCHETYPES
        for arch in ARCHETYPES:
            assert arch in ARCHETYPE_OPTIMAL_STEPS, f"Missing optimal steps for {arch}"

    def test_critical_actions_are_valid_operations(self):
        valid_ops = {
            "query_rights_db", "assess_transformation", "check_fingerprint",
            "check_usage_context", "cross_ref_history",
        }
        for arch, actions in ARCHETYPE_CRITICAL_ACTIONS.items():
            assert actions.issubset(valid_ops), f"Invalid action in {arch}: {actions - valid_ops}"
