"""ContentGuard Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import ContentGuardAction, ContentGuardObservation, ContentGuardState


class ContentGuardEnv(
    EnvClient[ContentGuardAction, ContentGuardObservation, ContentGuardState]
):
    """
    Client for the ContentGuard Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each instance gets its own isolated session — state never leaks between clients.

    Example:
        >>> with ContentGuardEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(ContentGuardAction(operation="query_rights_db"))
        ...     result = client.step(ContentGuardAction(operation="decide", verdict="remove"))
    """

    def _step_payload(self, action: ContentGuardAction) -> Dict:
        payload: Dict = {"operation": action.operation}
        if action.verdict is not None:
            payload["verdict"] = action.verdict
        return {"action": payload}

    def _parse_result(self, payload: Dict) -> StepResult[ContentGuardObservation]:
        obs_data = payload.get("observation", {})
        observation = ContentGuardObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ContentGuardState:
        return ContentGuardState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            budget_remaining=payload.get("budget_remaining", 1.0),
            actions_taken=payload.get("actions_taken", []),
            resolved_fields=payload.get("resolved_fields", {}),
            difficulty=payload.get("difficulty", "medium"),
            case_id=payload.get("case_id", ""),
        )
