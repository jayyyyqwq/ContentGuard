"""Tests for the skill-based media analysis layer."""

import pytest
from models import ContentGuardAction
from server.environment import ContentGuardEnvironment
from server.skills import BaseSkill, SkillRouter
from server.skills.synthetic import SyntheticSkill
from server.skills.audio import AudioSkill
from server.case_generator import generate_case, AUDIO_ARCHETYPES


class TestSyntheticSkill:
    def test_returns_case_values(self):
        skill = SyntheticSkill()
        case = {"fingerprint_match": 1, "composition_similarity_score": 0.85}
        fields = ["fingerprint_match", "composition_similarity_score"]
        result = skill.resolve_fields("check_fingerprint", case, fields)
        assert result == {"fingerprint_match": 1, "composition_similarity_score": 0.85}

    def test_skips_missing_fields(self):
        skill = SyntheticSkill()
        case = {"fingerprint_match": 1}
        fields = ["fingerprint_match", "composition_similarity_score"]
        result = skill.resolve_fields("check_fingerprint", case, fields)
        assert result == {"fingerprint_match": 1}


class TestAudioSkill:
    def test_loads_analysis_for_known_clip(self):
        skill = AudioSkill()
        case = {"audio_clip_id": "ai_synth_melody_01", "content_type": "audio"}
        fields = ["fingerprint_match", "composition_similarity_score"]
        result = skill.resolve_fields("check_fingerprint", case, fields)
        assert result["fingerprint_match"] == 0
        assert result["composition_similarity_score"] == 0.82

    def test_falls_back_without_clip_id(self):
        skill = AudioSkill()
        case = {"content_type": "audio", "fingerprint_match": 1, "composition_similarity_score": 0.5}
        fields = ["fingerprint_match", "composition_similarity_score"]
        result = skill.resolve_fields("check_fingerprint", case, fields)
        assert result["fingerprint_match"] == 1  # from case dict

    def test_falls_back_for_non_audio_actions(self):
        skill = AudioSkill()
        case = {
            "audio_clip_id": "ai_synth_melody_01",
            "rights_holder_count": 2,
            "license_status": "disputed",
            "license_age_days": 100,
            "db_confidence": 0.7,
            "conflict_flag": 1,
        }
        fields = ["rights_holder_count", "license_status", "license_age_days",
                   "db_confidence", "conflict_flag"]
        result = skill.resolve_fields("query_rights_db", case, fields)
        # query_rights_db has no audio analysis — falls back to case values
        assert result["rights_holder_count"] == 2
        assert result["license_status"] == "disputed"

    def test_commentary_clip_analysis(self):
        skill = AudioSkill()
        case = {"audio_clip_id": "commentary_speech_01", "content_type": "audio"}
        fields = ["transformation_index", "commentary_present", "overlap_duration_pct"]
        result = skill.resolve_fields("assess_transformation", case, fields)
        assert result["transformation_index"] == 0.82
        assert result["commentary_present"] == 1
        assert result["overlap_duration_pct"] == 0.15


class TestSkillRouter:
    def test_routes_by_content_type(self):
        router = SkillRouter()
        router.register("video", SyntheticSkill())
        router.register("audio", AudioSkill())

        audio_case = {"content_type": "audio", "audio_clip_id": "ai_synth_melody_01"}
        video_case = {"content_type": "video", "fingerprint_match": 1, "composition_similarity_score": 0.9}

        audio_result = router.resolve("check_fingerprint", audio_case,
                                       ["fingerprint_match", "composition_similarity_score"])
        video_result = router.resolve("check_fingerprint", video_case,
                                       ["fingerprint_match", "composition_similarity_score"])

        assert audio_result["fingerprint_match"] == 0       # from audio analysis
        assert video_result["fingerprint_match"] == 1       # from case dict

    def test_fallback_for_unknown_type(self):
        router = SkillRouter()
        router.set_fallback(SyntheticSkill())
        case = {"content_type": "image", "fingerprint_match": 1}
        result = router.resolve("check_fingerprint", case, ["fingerprint_match"])
        assert result["fingerprint_match"] == 1  # fallback to synthetic


class TestAudioArchetypeIntegration:
    def test_audio_archetypes_get_content_type_audio(self):
        for archetype_name in AUDIO_ARCHETYPES:
            case = generate_case(archetype_name, seed=42)
            assert case["content_type"] == "audio", f"{archetype_name} should be audio"

    def test_audio_archetypes_get_clip_id(self):
        for archetype_name in AUDIO_ARCHETYPES:
            case = generate_case(archetype_name, seed=42)
            assert "audio_clip_id" in case, f"{archetype_name} should have audio_clip_id"

    def test_video_archetypes_stay_video(self):
        video_archetypes = ["verbatim_commercial", "parody_high_overlap",
                            "expired_license_disputed", "multi_claimant_non_overlapping"]
        for archetype_name in video_archetypes:
            case = generate_case(archetype_name, seed=42)
            assert case["content_type"] == "video", f"{archetype_name} should be video"
            assert "audio_clip_id" not in case

    def test_audio_clip_assignment_deterministic(self):
        for archetype_name in AUDIO_ARCHETYPES:
            case1 = generate_case(archetype_name, seed=99)
            case2 = generate_case(archetype_name, seed=99)
            assert case1["audio_clip_id"] == case2["audio_clip_id"]


class TestEnvironmentAudioRouting:
    @pytest.fixture
    def env(self):
        return ContentGuardEnvironment()

    def test_hard_difficulty_is_audio(self, env):
        obs = env.reset(seed=42, difficulty="hard")  # ai_audio_reconstruction
        assert obs.content_type == "audio"

    def test_easy_difficulty_stays_video(self, env):
        obs = env.reset(seed=42, difficulty="easy")  # verbatim_commercial
        assert obs.content_type == "video"

    def test_audio_fingerprint_returns_analysis_values(self, env):
        env.reset(seed=42, difficulty="hard")  # ai_audio_reconstruction
        obs = env.step(ContentGuardAction(operation="check_fingerprint"))
        # AI audio: fingerprint_match should be 0 (evades detection)
        assert obs.fingerprint_match == 0
        # But composition similarity should be high
        assert obs.composition_similarity_score > 0.7

    def test_audio_transformation_returns_analysis_values(self, env):
        env.reset(seed=42, difficulty="easy_medium")  # educational_excerpt → audio
        obs = env.step(ContentGuardAction(operation="assess_transformation"))
        assert obs.transformation_index != -1.0
        assert obs.commentary_present != -1
        assert obs.overlap_duration_pct != -1.0
