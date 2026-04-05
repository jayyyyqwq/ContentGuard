"""Skill-based media analysis backends for ContentGuard.

Skills provide modular analysis pipelines that source investigation field
values based on content type. The SkillRouter dispatches to the appropriate
skill when the environment processes an investigation action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    """Interface for media analysis backends."""

    @abstractmethod
    def resolve_fields(
        self, action: str, case: dict, fields_to_reveal: list[str]
    ) -> dict[str, Any]:
        """Return {field_name: value} for each field in fields_to_reveal.

        The returned dict MUST contain a key for every field in
        fields_to_reveal that the skill can resolve. Missing keys
        will remain masked on the observation.
        """
        ...


class SkillRouter:
    """Routes investigation actions to the appropriate skill backend."""

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}
        self._fallback: BaseSkill | None = None

    def register(self, content_type: str, skill: BaseSkill) -> None:
        self._skills[content_type] = skill

    def set_fallback(self, skill: BaseSkill) -> None:
        self._fallback = skill

    def resolve(
        self, action: str, case: dict, fields_to_reveal: list[str]
    ) -> dict[str, Any]:
        content_type = case.get("content_type", "video")
        skill = self._skills.get(content_type, self._fallback)
        if skill is None:
            return {}
        return skill.resolve_fields(action, case, fields_to_reveal)
