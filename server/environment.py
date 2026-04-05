# environment.py
from uuid import uuid4  # used only for episode_id fallback
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import ContentGuardAction, ContentGuardObservation, ContentGuardState
from .case_generator import generate_case
from .tasks import DIFFICULTY_ARCHETYPE_MAP
from .grader import terminal_reward, step_reward
from .skills import SkillRouter
from .skills.synthetic import SyntheticSkill
from .skills.audio import AudioSkill

ACTION_UNLOCKS = {
    "query_rights_db":    ["rights_holder_count", "license_status",
                           "license_age_days", "db_confidence", "conflict_flag"],
    "assess_transformation": ["transformation_index", "commentary_present",
                              "overlap_duration_pct"],
    "check_fingerprint":  ["fingerprint_match", "composition_similarity_score"],
    "check_usage_context": ["commercial_channel", "sub_license_depth"],
    "cross_ref_history":  ["prior_disputes_same_uploader"],
}

_router = SkillRouter()
_router.register("video", SyntheticSkill())
_router.register("audio", AudioSkill())
_router.set_fallback(SyntheticSkill())


class ContentGuardEnvironment(Environment):

    # Required for HF Spaces multi-worker deployment.
    # Without this flag, max_concurrent_envs > 1 raises ConcurrencyConfigurationError.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._case: dict = {}
        self._obs: ContentGuardObservation = ContentGuardObservation()
        self._state: ContentGuardState = ContentGuardState()

    def reset(self, seed=None, episode_id=None, **kwargs) -> ContentGuardObservation:
        """
        Difficulty is passed via kwargs: reset(difficulty="hard")
        Valid values: "easy", "medium", "hard". Defaults to "medium".
        """
        difficulty = kwargs.get("difficulty", "medium")
        if difficulty not in DIFFICULTY_ARCHETYPE_MAP:
            valid = ", ".join(sorted(DIFFICULTY_ARCHETYPE_MAP))
            raise ValueError(f"Unknown difficulty '{difficulty}'. Valid: {valid}")
        archetype_name = DIFFICULTY_ARCHETYPE_MAP[difficulty]
        self._case = generate_case(archetype_name, seed=seed)

        self._state = ContentGuardState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            budget_remaining=1.0,
            actions_taken=[],
            resolved_fields={"conflict_flag_value": self._case.get("conflict_flag", 0)},
            difficulty=difficulty,
            case_id=self._case.get("case_id", str(uuid4())),
        )

        # Return only surface metadata — all investigation fields masked
        self._obs = ContentGuardObservation(
            uploader_id=self._case.get("uploader_id", "uploader_1"),
            content_duration_s=self._case.get("content_duration_s", 60),
            claim_received=True,
            claimant_id=self._case.get("claimant_id", "claimant_1"),
            content_type=self._case.get("content_type", "video"),
        )
        return self._obs

    def step(self, action: ContentGuardAction) -> ContentGuardObservation:
        self._state.step_count += 1
        self._state.budget_remaining = round(self._state.budget_remaining - 0.02, 10)
        self._state.actions_taken.append(action.operation)

        # Budget exhausted without decide — timeout
        if self._state.budget_remaining <= 0 and action.operation != "decide":
            self._obs.done = True
            self._obs.reward = -0.50
            return self._obs

        # Terminal: agent called decide
        if action.operation == "decide":
            reward = terminal_reward(
                agent_verdict=action.verdict,
                ground_truth=self._case["ground_truth"],
                state=self._state,
                case=self._case,
            )
            self._obs.done = True
            self._obs.reward = reward
            return self._obs

        # Investigation action: unlock fields via skill router
        fields_to_reveal = ACTION_UNLOCKS.get(action.operation, [])
        resolved = _router.resolve(action.operation, self._case, fields_to_reveal)
        for field in fields_to_reveal:
            if field in resolved:
                setattr(self._obs, field, resolved[field])

        self._obs.reward = step_reward(action.operation, self._case, self._state)
        self._obs.done = False
        return self._obs

    @property
    def state(self) -> ContentGuardState:
        """
        Required by OpenEnv spec. openenv validate checks for this property.
        Returns a copy to prevent external mutation of episode state.
        """
        return self._state.model_copy(deep=True)

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="contentguard",
            description="Content rights adjudication RL environment",
            version="0.1.0",
        )

    def close(self) -> None:
        pass


