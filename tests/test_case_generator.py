"""Tests for server/case_generator.py — archetype generation, constraints, determinism."""
import pytest
from server.case_generator import (
    ARCHETYPES,
    HARD_CONSTRAINTS,
    generate_case,
    compute_ground_truth,
)
from server.grader import VERDICT_BINS, get_correct_bin


# ── All 14 archetypes generate valid cases ─────────────────────────────────
class TestArchetypeGeneration:
    @pytest.mark.parametrize("archetype", list(ARCHETYPES.keys()))
    def test_generates_without_error(self, archetype: str):
        case = generate_case(archetype, seed=42)
        assert case is not None
        assert case["archetype"] == archetype

    @pytest.mark.parametrize("archetype", list(ARCHETYPES.keys()))
    def test_has_required_fields(self, archetype: str):
        case = generate_case(archetype, seed=42)
        required = ["case_id", "uploader_id", "claimant_id", "content_duration_s",
                     "content_type", "ground_truth", "correct_verdict", "archetype"]
        for field in required:
            assert field in case, f"Missing field '{field}' in {archetype}"

    @pytest.mark.parametrize("archetype", list(ARCHETYPES.keys()))
    def test_ground_truth_in_range(self, archetype: str):
        case = generate_case(archetype, seed=42)
        gt_lo, gt_hi = ARCHETYPES[archetype]["ground_truth_range"]
        assert gt_lo <= case["ground_truth"] <= gt_hi, (
            f"{archetype}: ground_truth {case['ground_truth']} outside [{gt_lo}, {gt_hi}]"
        )

    @pytest.mark.parametrize("archetype", list(ARCHETYPES.keys()))
    def test_correct_verdict_matches_archetype(self, archetype: str):
        case = generate_case(archetype, seed=42)
        assert case["correct_verdict"] == ARCHETYPES[archetype]["correct_verdict"]


# ── Hard constraints ───────────────────────────────────────────────────────
class TestHardConstraints:
    @pytest.mark.parametrize("archetype", list(ARCHETYPES.keys()))
    def test_generated_case_passes_all_constraints(self, archetype: str):
        case = generate_case(archetype, seed=42)
        for i, constraint in enumerate(HARD_CONSTRAINTS):
            assert constraint(case), (
                f"Constraint {i} failed for {archetype}: {case}"
            )

    def test_valid_license_with_negative_age_rejected(self):
        bad_case = {"license_status": "valid", "license_age_days": -10}
        assert not HARD_CONSTRAINTS[0](bad_case)

    def test_single_holder_with_conflict_rejected(self):
        bad_case = {"rights_holder_count": 1, "conflict_flag": 1}
        assert not HARD_CONSTRAINTS[1](bad_case)

    def test_no_commentary_with_high_transform_rejected(self):
        bad_case = {"commentary_present": 0, "transformation_index": 0.80}
        assert not HARD_CONSTRAINTS[2](bad_case)

    def test_high_overlap_with_high_transform_rejected(self):
        bad_case = {"overlap_duration_pct": 0.95, "transformation_index": 0.65}
        assert not HARD_CONSTRAINTS[3](bad_case)

    def test_fingerprint_match_with_low_similarity_rejected(self):
        bad_case = {"fingerprint_match": 1, "composition_similarity_score": 0.20}
        assert not HARD_CONSTRAINTS[4](bad_case)


# ── Seed determinism ──────────────────────────────────────────────────────
class TestDeterminism:
    @pytest.mark.parametrize("archetype", list(ARCHETYPES.keys()))
    def test_same_seed_same_output(self, archetype: str):
        c1 = generate_case(archetype, seed=123)
        c2 = generate_case(archetype, seed=123)
        # Exclude rationale file path (contains UUID but seeded so should match)
        for key in ["ground_truth", "correct_verdict", "case_id",
                     "uploader_id", "claimant_id", "content_duration_s"]:
            assert c1[key] == c2[key], f"Non-deterministic field '{key}' for {archetype}"

    def test_different_seeds_different_output(self):
        c1 = generate_case("verbatim_commercial", seed=1)
        c2 = generate_case("verbatim_commercial", seed=2)
        assert c1["case_id"] != c2["case_id"]


# ── Ground truth formula ──────────────────────────────────────────────────
class TestGroundTruth:
    def test_high_transformation_low_overlap_noncommercial(self):
        """Should score high (fair use → keep)."""
        case = {
            "transformation_index": 0.85,
            "commercial_channel": 0,
            "overlap_duration_pct": 0.10,
        }
        gt = compute_ground_truth(case)
        assert gt > 0.65, f"Expected keep range, got {gt}"

    def test_low_transformation_high_overlap_commercial(self):
        """Should score low (infringement → remove)."""
        case = {
            "transformation_index": 0.10,
            "commercial_channel": 1,
            "overlap_duration_pct": 0.95,
        }
        gt = compute_ground_truth(case)
        assert gt < 0.30, f"Expected remove range, got {gt}"

    def test_score_bounded_zero_one(self):
        """Ground truth should always be in [0, 1]."""
        for arch in ARCHETYPES:
            case = generate_case(arch, seed=42)
            gt = compute_ground_truth(case)
            assert 0.0 <= gt <= 1.0, f"{arch}: ground_truth {gt} out of bounds"
