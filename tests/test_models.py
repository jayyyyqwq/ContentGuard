"""Tests for models.py — Pydantic validation rules."""
import pytest
from pydantic import ValidationError
from models import ContentGuardAction, ContentGuardObservation, ContentGuardState


# ── ContentGuardAction ─────────────────────────────────────────────────────
class TestContentGuardAction:
    @pytest.mark.parametrize("op", [
        "query_rights_db", "assess_transformation", "check_fingerprint",
        "check_usage_context", "cross_ref_history",
    ])
    def test_investigation_ops_accepted(self, op: str):
        action = ContentGuardAction(operation=op)
        assert action.operation == op
        assert action.verdict is None

    def test_decide_requires_verdict(self):
        with pytest.raises(ValidationError, match="verdict is required"):
            ContentGuardAction(operation="decide")

    def test_decide_with_verdict_accepted(self):
        action = ContentGuardAction(operation="decide", verdict="remove")
        assert action.verdict == "remove"

    @pytest.mark.parametrize("verdict", ["remove", "monetize", "escalate", "keep"])
    def test_all_verdicts_accepted_on_decide(self, verdict: str):
        action = ContentGuardAction(operation="decide", verdict=verdict)
        assert action.verdict == verdict

    def test_verdict_on_non_decide_rejected(self):
        with pytest.raises(ValidationError, match="verdict must be None"):
            ContentGuardAction(operation="query_rights_db", verdict="remove")

    def test_invalid_operation_rejected(self):
        with pytest.raises(ValidationError):
            ContentGuardAction(operation="hack_the_system")

    def test_invalid_verdict_rejected(self):
        with pytest.raises(ValidationError):
            ContentGuardAction(operation="decide", verdict="destroy")


# ── ContentGuardObservation ────────────────────────────────────────────────
class TestContentGuardObservation:
    def test_defaults_are_masked(self):
        obs = ContentGuardObservation()
        assert obs.rights_holder_count == -1
        assert obs.license_status == "unknown"
        assert obs.license_age_days == -1
        assert obs.db_confidence == -1.0
        assert obs.conflict_flag == -1
        assert obs.transformation_index == -1.0
        assert obs.commentary_present == -1
        assert obs.overlap_duration_pct == -1.0
        assert obs.fingerprint_match == -1
        assert obs.composition_similarity_score == -1.0
        assert obs.commercial_channel == -1
        assert obs.sub_license_depth == -1
        assert obs.prior_disputes_same_uploader == -1

    def test_surface_metadata_defaults(self):
        obs = ContentGuardObservation()
        assert obs.uploader_id == ""
        assert obs.content_duration_s == 0
        assert obs.claim_received is False
        assert obs.claimant_id == ""
        assert obs.content_type == ""


# ── ContentGuardState ──────────────────────────────────────────────────────
class TestContentGuardState:
    def test_default_values(self):
        s = ContentGuardState()
        assert s.budget_remaining == 1.0
        assert s.actions_taken == []
        assert s.difficulty == "medium"
        assert s.case_id == ""

    def test_custom_values(self):
        s = ContentGuardState(
            budget_remaining=0.5,
            actions_taken=["query_rights_db"],
            difficulty="hard",
            case_id="test-123",
        )
        assert s.budget_remaining == 0.5
        assert s.actions_taken == ["query_rights_db"]
        assert s.difficulty == "hard"
        assert s.case_id == "test-123"
