"""Tests for server/environment.py — reset, step, state isolation."""
import pytest
from server.environment import ContentGuardEnvironment, ACTION_UNLOCKS
from models import ContentGuardAction


@pytest.fixture
def env():
    return ContentGuardEnvironment()


# ── reset ──────────────────────────────────────────────────────────────────
class TestReset:
    def test_returns_observation(self, env: ContentGuardEnvironment):
        obs = env.reset(seed=42, difficulty="easy")
        assert obs is not None
        assert not obs.done
        assert obs.reward is None or obs.reward == 0.0

    def test_surface_metadata_visible(self, env: ContentGuardEnvironment):
        obs = env.reset(seed=42, difficulty="easy")
        assert obs.uploader_id != ""
        assert obs.content_duration_s > 0
        assert obs.claim_received is True
        assert obs.claimant_id != ""
        assert obs.content_type == "video"

    def test_investigation_fields_masked(self, env: ContentGuardEnvironment):
        obs = env.reset(seed=42, difficulty="easy")
        assert obs.rights_holder_count == -1
        assert obs.license_status == "unknown"
        assert obs.transformation_index == -1.0
        assert obs.fingerprint_match == -1
        assert obs.commercial_channel == -1
        assert obs.prior_disputes_same_uploader == -1

    def test_state_initialized(self, env: ContentGuardEnvironment):
        env.reset(seed=42, difficulty="medium")
        s = env.state
        assert s.step_count == 0
        assert s.budget_remaining == 1.0
        assert s.actions_taken == []
        assert s.difficulty == "medium"

    @pytest.mark.parametrize("difficulty", [
        "easy", "easy_medium", "medium", "medium_hard",
        "hard", "hard_expert", "expert",
    ])
    def test_all_difficulties_accepted(self, env: ContentGuardEnvironment, difficulty: str):
        obs = env.reset(seed=42, difficulty=difficulty)
        assert not obs.done

    def test_invalid_difficulty_raises(self, env: ContentGuardEnvironment):
        with pytest.raises(ValueError, match="Unknown difficulty"):
            env.reset(difficulty="impossible")


# ── state isolation ────────────────────────────────────────────────────────
class TestStateIsolation:
    def test_reset_clears_previous_episode(self, env: ContentGuardEnvironment):
        env.reset(seed=1, difficulty="easy")
        env.step(ContentGuardAction(operation="query_rights_db"))
        assert env.state.step_count == 1

        env.reset(seed=2, difficulty="medium")
        assert env.state.step_count == 0
        assert env.state.actions_taken == []
        assert env.state.budget_remaining == 1.0

    def test_state_returns_copy(self, env: ContentGuardEnvironment):
        env.reset(seed=42, difficulty="easy")
        s1 = env.state
        s1.actions_taken.append("fake_action")
        s2 = env.state
        assert "fake_action" not in s2.actions_taken


# ── step: investigation ────────────────────────────────────────────────────
class TestStepInvestigation:
    def test_unlocks_correct_fields(self, env: ContentGuardEnvironment):
        obs = env.reset(seed=42, difficulty="easy")
        assert obs.rights_holder_count == -1

        obs = env.step(ContentGuardAction(operation="query_rights_db"))
        assert obs.rights_holder_count != -1
        assert obs.db_confidence != -1.0
        assert not obs.done

    def test_step_count_increments(self, env: ContentGuardEnvironment):
        env.reset(seed=42, difficulty="easy")
        env.step(ContentGuardAction(operation="query_rights_db"))
        assert env.state.step_count == 1
        env.step(ContentGuardAction(operation="assess_transformation"))
        assert env.state.step_count == 2

    def test_budget_decreases(self, env: ContentGuardEnvironment):
        env.reset(seed=42, difficulty="easy")
        env.step(ContentGuardAction(operation="query_rights_db"))
        assert env.state.budget_remaining < 1.0

    def test_actions_recorded(self, env: ContentGuardEnvironment):
        env.reset(seed=42, difficulty="easy")
        env.step(ContentGuardAction(operation="query_rights_db"))
        env.step(ContentGuardAction(operation="assess_transformation"))
        assert env.state.actions_taken == ["query_rights_db", "assess_transformation"]

    @pytest.mark.parametrize("action,fields", list(ACTION_UNLOCKS.items()))
    def test_each_action_unlocks_its_fields(
        self, env: ContentGuardEnvironment, action: str, fields: list[str],
    ):
        env.reset(seed=42, difficulty="easy")
        obs = env.step(ContentGuardAction(operation=action))
        for field in fields:
            val = getattr(obs, field, None)
            # Field must be revealed — no longer the masked sentinel value
            assert val not in (-1, -1.0, "unknown"), (
                f"Field '{field}' was not revealed after action '{action}' "
                f"(still has masked default: {val!r})"
            )


# ── step: decide ───────────────────────────────────────────────────────────
class TestStepDecide:
    def test_decide_ends_episode(self, env: ContentGuardEnvironment):
        env.reset(seed=42, difficulty="easy")
        obs = env.step(ContentGuardAction(operation="decide", verdict="remove"))
        assert obs.done is True

    def test_decide_returns_reward(self, env: ContentGuardEnvironment):
        env.reset(seed=42, difficulty="easy")
        obs = env.step(ContentGuardAction(operation="decide", verdict="remove"))
        assert obs.reward != 0.0

    def test_correct_verdict_positive_reward(self, env: ContentGuardEnvironment):
        """Easy task (verbatim_commercial) → correct verdict is 'remove'."""
        env.reset(seed=42, difficulty="easy")
        obs = env.step(ContentGuardAction(operation="decide", verdict="remove"))
        assert obs.reward > 0.0

    def test_catastrophic_verdict_negative_reward(self, env: ContentGuardEnvironment):
        """Easy task (verbatim_commercial) → 'keep' on infringement is catastrophic."""
        env.reset(seed=42, difficulty="easy")
        obs = env.step(ContentGuardAction(operation="decide", verdict="keep"))
        assert obs.reward < -0.5


# ── budget timeout ─────────────────────────────────────────────────────────
class TestBudgetTimeout:
    def test_budget_exhaustion_ends_episode(self, env: ContentGuardEnvironment):
        env.reset(seed=42, difficulty="easy")
        # Exhaust budget: 1.0 / 0.02 = 50 steps
        for _ in range(51):
            obs = env.step(ContentGuardAction(operation="query_rights_db"))
            if obs.done:
                break
        assert obs.done is True
        assert obs.reward == -0.50
