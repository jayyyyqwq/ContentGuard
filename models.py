# models.py
from typing import Literal, Optional
from pydantic import model_validator
from openenv.core.env_server.types import Action, Observation, State



class ContentGuardObservation(Observation):
    """
    Surface metadata is always visible.
    Investigation fields start masked (-1 or "unknown").
    Each action unlocks specific fields — see ACTION_UNLOCKS in environment.py.
    """
    # Always visible
    uploader_id: str = ""
    content_duration_s: int = 0
    claim_received: bool = False
    claimant_id: str = ""
    content_type: str = ""

    # Unlocked by: query_rights_db
    rights_holder_count: int = -1
    license_status: str = "unknown"
    license_age_days: int = -1
    db_confidence: float = -1.0
    conflict_flag: int = -1

    # Unlocked by: assess_transformation
    transformation_index: float = -1.0
    commentary_present: int = -1
    overlap_duration_pct: float = -1.0

    # Unlocked by: check_fingerprint
    fingerprint_match: int = -1
    composition_similarity_score: float = -1.0

    # Unlocked by: check_usage_context
    commercial_channel: int = -1
    sub_license_depth: int = -1

    # Unlocked by: cross_ref_history
    prior_disputes_same_uploader: int = -1


class ContentGuardAction(Action):
    """
    operation: which investigative or terminal action to take.
    verdict: ONLY populated when operation == "decide". Enforced by validator.
    """
    operation: Literal[
        "query_rights_db",
        "assess_transformation",
        "check_fingerprint",
        "check_usage_context",
        "cross_ref_history",
        "decide",
    ]
    verdict: Optional[Literal["remove", "monetize", "escalate", "keep"]] = None

    @model_validator(mode="after")
    def verdict_only_on_decide(self) -> "ContentGuardAction":
        if self.operation == "decide" and self.verdict is None:
            raise ValueError("verdict is required when operation is 'decide'")
        if self.operation != "decide" and self.verdict is not None:
            raise ValueError("verdict must be None for non-decide operations")
        return self


class ContentGuardState(State):
    """
    Extends OpenEnv base State (which provides episode_id and step_count).
    Tracks budget and resolved field flags for conflict detection.
    """
    budget_remaining: float = 1.0
    actions_taken: list[str] = []
    resolved_fields: dict[str, bool] = {}
    difficulty: str = "medium"
    case_id: str = ""
