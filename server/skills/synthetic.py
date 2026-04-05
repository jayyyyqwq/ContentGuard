"""Synthetic skill: reads pre-generated values from the case dict.

This wraps the original ContentGuard behavior where case_generator.py
samples all field values upfront and step() reveals them. Zero behavior
change from the pre-skill architecture.
"""

from __future__ import annotations

from typing import Any

from . import BaseSkill


class SyntheticSkill(BaseSkill):
    """Default skill — reads values directly from the case dict."""

    def resolve_fields(
        self, action: str, case: dict, fields_to_reveal: list[str]
    ) -> dict[str, Any]:
        return {f: case[f] for f in fields_to_reveal if f in case}
