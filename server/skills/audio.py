"""Audio skill: sources observation fields from pre-computed audio analysis.

For audio-typed cases, investigation actions return values derived from
real audio analysis (librosa features pre-computed by precompute.py)
rather than purely synthetic random samples.

The .analysis.json sidecar files are loaded from data/samples/audio/.
If no analysis file exists for a clip, falls back to case dict values
(identical to SyntheticSkill behavior).
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import BaseSkill

_AUDIO_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "samples", "audio"
)

_ANALYSIS_CACHE: dict[str, dict] = {}


def _load_analysis(clip_id: str) -> dict | None:
    """Load pre-computed analysis for an audio clip. Cached in-memory."""
    if clip_id in _ANALYSIS_CACHE:
        return _ANALYSIS_CACHE[clip_id]

    path = os.path.join(_AUDIO_DATA_DIR, f"{clip_id}.analysis.json")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)
    _ANALYSIS_CACHE[clip_id] = data
    return data


class AudioSkill(BaseSkill):
    """Skill that sources field values from pre-computed audio analysis."""

    def resolve_fields(
        self, action: str, case: dict, fields_to_reveal: list[str]
    ) -> dict[str, Any]:
        clip_id = case.get("audio_clip_id")
        if clip_id is None:
            # No audio clip assigned — fall back to synthetic values
            return {f: case[f] for f in fields_to_reveal if f in case}

        analysis = _load_analysis(clip_id)
        if analysis is None:
            # Analysis file missing — fall back to synthetic values
            return {f: case[f] for f in fields_to_reveal if f in case}

        result: dict[str, Any] = {}
        action_data = analysis.get(action, {})

        for field in fields_to_reveal:
            if field in action_data:
                result[field] = action_data[field]
            elif field in case:
                # Fall back to case-level value for fields not covered
                # by audio analysis (e.g. rights_holder_count, license_status)
                result[field] = case[field]

        return result
