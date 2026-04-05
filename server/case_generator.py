#case_generator.py
import json
import random
import os
import uuid
from pathlib import Path

AUDIO_ARCHETYPES = {
    "ai_audio_reconstruction",
    "background_music_commercial",
    "commentary_clip_noncommercial",
    "educational_excerpt",
}

_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "audio" / "manifest.json"
_MANIFEST_CACHE: dict | None = None


def _load_audio_manifest() -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        if _MANIFEST_PATH.exists():
            with open(_MANIFEST_PATH) as f:
                _MANIFEST_CACHE = json.load(f)
        else:
            _MANIFEST_CACHE = {"clips": []}
    return _MANIFEST_CACHE


def _assign_audio_clip(archetype_name: str, rng: random.Random) -> str | None:
    """Deterministically pick an audio clip for an audio archetype."""
    manifest = _load_audio_manifest()
    candidates = [c for c in manifest["clips"] if c["archetype"] == archetype_name]
    if not candidates:
        return None
    return rng.choice(candidates)["clip_id"]


ARCHETYPES = {
    "verbatim_commercial": {
        "transformation_index":        (0.05, 0.15),
        "overlap_duration_pct":        (0.85, 1.00),
        "commercial_channel":          (1, 1),
        "rights_holder_count":         (1, 1),
        "conflict_flag":               (0, 0),
        "license_status":              "valid",
        "license_age_days":            (100, 1000),
        "fingerprint_match":           (1, 1),
        "composition_similarity_score": (0.85, 1.00),
        "commentary_present":          (0, 0),
        "prior_disputes_same_uploader": (0, 3),
        "ground_truth_range":          (0.02, 0.12),
        "correct_verdict":             "remove",
    },
    "commentary_clip_noncommercial": {
        "transformation_index":        (0.70, 0.90),
        "overlap_duration_pct":        (0.08, 0.25),
        "commercial_channel":          (0, 0),
        "rights_holder_count":         (1, 2),
        "commentary_present":          (1, 1),
        "prior_disputes_same_uploader": (0, 1),
        "ground_truth_range":          (0.72, 0.88),
        "correct_verdict":             "keep",
    },
    "parody_high_overlap": {
        "transformation_index":        (0.75, 0.90),
        "overlap_duration_pct":        (0.45, 0.70),
        "commercial_channel":          (0, 1),
        "commentary_present":          (1, 1),
        "prior_disputes_same_uploader": (0, 2),
        "ground_truth_range":          (0.47, 0.64),
        "correct_verdict":             "escalate",
    },
    "educational_excerpt": {
        "transformation_index":        (0.50, 0.70),
        "overlap_duration_pct":        (0.15, 0.35),
        "commercial_channel":          (0, 0),
        "commentary_present":          (1, 1),
        "prior_disputes_same_uploader": (0, 0),
        "ground_truth_range":          (0.68, 0.82),
        "correct_verdict":             "keep",
    },
    "background_music_commercial": {
        "transformation_index":        (0.05, 0.20),
        "overlap_duration_pct":        (0.35, 0.65),
        "commercial_channel":          (1, 1),
        "commentary_present":          (0, 0),
        "prior_disputes_same_uploader": (0, 2),
        "ground_truth_range":          (0.05, 0.29),
        "correct_verdict":             "remove",
    },
    "expired_license_disputed": {
        "license_status":              "expired",
        "license_age_days":            (-180, -1),
        "db_confidence":               (0.55, 0.75),
        "commentary_present":          (0, 0),
        "prior_disputes_same_uploader": (0, 1),
        "ground_truth_range":          (0.46, 0.64),
        "correct_verdict":             "escalate",
    },
    "multi_claimant_non_overlapping": {
        "rights_holder_count":         (2, 3),
        "conflict_flag":               (1, 1),
        "db_confidence":               (0.45, 0.64),
        "commentary_present":          (0, 0),
        "prior_disputes_same_uploader": (1, 4),
        "ground_truth_range":          (0.46, 0.64),
        "correct_verdict":             "escalate",
    },
    "orphaned_work": {
        "rights_holder_count":         (0, 0),
        "db_confidence":               (0.30, 0.50),
        "license_status":              "unknown",
        "commentary_present":          (0, 0),
        "prior_disputes_same_uploader": (0, 0),
        "ground_truth_range":          (0.48, 0.62),
        "correct_verdict":             "escalate",
    },
    "creative_commons_misapplication": {
        "license_status":              "disputed",
        "transformation_index":        (0.30, 0.55),
        "commentary_present":          (0, 1),
        "prior_disputes_same_uploader": (0, 2),
        "ground_truth_range":          (0.31, 0.44),
        "correct_verdict":             "monetize",
    },
    "transformative_large_amount": {
        "transformation_index":        (0.72, 0.88),
        "overlap_duration_pct":        (0.55, 0.80),
        "commentary_present":          (1, 1),
        "prior_disputes_same_uploader": (0, 1),
        "ground_truth_range":          (0.45, 0.62),
        "correct_verdict":             "escalate",
    },
    "noncommercial_direct_substitute": {
        "commercial_channel":          (0, 0),
        "overlap_duration_pct":        (0.80, 1.00),
        "transformation_index":        (0.02, 0.18),
        "commentary_present":          (0, 0),
        "prior_disputes_same_uploader": (0, 2),
        "ground_truth_range":          (0.08, 0.28),
        "correct_verdict":             "remove",
    },
    "educational_verbatim_complete": {
        "transformation_index":        (0.10, 0.30),
        "overlap_duration_pct":        (0.90, 1.00),
        "commercial_channel":          (0, 0),
        "commentary_present":          (0, 1),
        "prior_disputes_same_uploader": (0, 1),
        "ground_truth_range":          (0.31, 0.44),
        "correct_verdict":             "monetize",
    },
    # 2026 archetypes
    "live_sports_gameplay_disguise": {
        "transformation_index":        (0.05, 0.12),
        "overlap_duration_pct":        (0.82, 0.95),
        "rights_holder_count":         (2, 2),
        "conflict_flag":               (1, 1),
        "hud_overlay_present":         1,
        "db_confidence":               (0.85, 0.95),
        "commentary_present":          (0, 0),
        "prior_disputes_same_uploader": (1, 3),
        "ground_truth_range":          (0.05, 0.14),
        "correct_verdict":             "remove",
    },
    "ai_audio_reconstruction": {
        "fingerprint_match":           0,
        "composition_similarity_score": (0.72, 0.88),
        "transformation_index":        (0.15, 0.28),
        "copyright_years_remaining":   (8, 20),
        "ai_generated_flag":           1,
        "db_confidence":               (0.65, 0.80),
        "commentary_present":          (0, 0),
        "prior_disputes_same_uploader": (0, 2),
        "ground_truth_range":          (0.10, 0.22),
        "correct_verdict":             "remove",
    },
}

HARD_CONSTRAINTS = [
    lambda c: not (c.get("license_status") == "valid" and c.get("license_age_days", 0) < 0),
    lambda c: not (c.get("rights_holder_count") == 1 and c.get("conflict_flag") == 1),
    lambda c: not (c.get("commentary_present") == 0 and c.get("transformation_index", 0) > 0.75),
    lambda c: not (c.get("overlap_duration_pct", 0) > 0.90 and c.get("transformation_index", 0) > 0.60),
    lambda c: not (c.get("fingerprint_match") == 1 and c.get("composition_similarity_score", 0) < 0.30),
]

def derive_fields(case: dict) -> dict:
    rh = case.get("rights_holder_count", 1)
    ls = case.get("license_status", "valid")

    # db_confidence: only derive if not already sampled from the archetype spec
    if "db_confidence" not in case:
        base_conf = 0.90 if rh == 1 else 0.65
        if ls == "disputed": base_conf -= 0.20
        if ls == "expired":  base_conf -= 0.10
        if rh >= 3:          base_conf -= 0.15
        case["db_confidence"] = round(max(0.35, min(0.98, base_conf + random.uniform(-0.05, 0.05))), 2)

    # sub_license_depth derived from license complexity
    case["sub_license_depth"] = 0 if ls == "valid" else (2 if ls == "disputed" else 1)

    return case

def build_rationale(case: dict, archetype_name: str) -> str:
    content = (
        f"Archetype: {archetype_name}\n"
        f"Transformation factor: {case.get('transformation_index', 'N/A')} "
        f"(Campbell v. Acuff-Rose, 1994)\n"
        f"Overlap: {case.get('overlap_duration_pct', 'N/A')} "
        f"(Harper & Row v. Nation, 1985)\n"
        f"Commercial use: {case.get('commercial_channel', 'N/A')} "
        f"(Sony Corp. v. Universal, 1984)\n"
        f"Ground truth score: {case.get('ground_truth', 'N/A')}\n"
        f"Correct verdict: {case.get('correct_verdict', 'N/A')}"
    )
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "rationales"), exist_ok=True)
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "rationales", f"{case['case_id']}.txt")
    with open(filepath, "w") as f:
        f.write(content)
        
    return content

FACTOR_WEIGHTS = {
    "transformation": 0.35,   # Campbell v. Acuff-Rose (1994)
    "nature":         0.15,   # Sony Corp. v. Universal (1984)
    "amount":         0.20,   # Harper & Row v. Nation (1985)
    "market_effect":  0.30,   # Stewart v. Abend (1990)
}

def compute_ground_truth(case: dict) -> float:
    t = case.get("transformation_index", 0.5)
    n = 0.7 if case.get("commercial_channel", 0) == 0 else 0.3
    a = 1.0 - case.get("overlap_duration_pct", 0.5)
    m = 1.0 - (case.get("overlap_duration_pct", 0.5) * (0.8 if case.get("commercial_channel", 0) else 0.4))

    score = (t * FACTOR_WEIGHTS["transformation"] +
             n * FACTOR_WEIGHTS["nature"] +
             a * FACTOR_WEIGHTS["amount"] +
             m * FACTOR_WEIGHTS["market_effect"])
    return round(score, 3)

def generate_case(archetype_name: str, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    archetype = ARCHETYPES[archetype_name]

    for attempt in range(200):
        content_type = "audio" if archetype_name in AUDIO_ARCHETYPES else "video"
        case = {"archetype": archetype_name, "case_id": str(uuid.UUID(int=rng.getrandbits(128))), "uploader_id": f"uploader_{rng.randint(1000, 9999)}", "claimant_id": f"claimant_{rng.randint(1000, 9999)}", "content_duration_s": rng.randint(30, 3600), "content_type": content_type}

        for field, spec in archetype.items():
            if field in ("ground_truth_range", "correct_verdict"):
                continue
            if isinstance(spec, tuple) and len(spec) == 2:
                lo, hi = spec
                case[field] = round(rng.uniform(lo, hi), 3) if isinstance(lo, float) else rng.randint(lo, hi)
            else:
                case[field] = spec

        case = derive_fields(case)

        if all(constraint(case) for constraint in HARD_CONSTRAINTS):
            gt_lo, gt_hi = archetype["ground_truth_range"]
            case["ground_truth"] = round(rng.uniform(gt_lo, gt_hi), 3)
            case["correct_verdict"] = archetype["correct_verdict"]

            # Assign audio clip for audio archetypes (deterministic via seeded rng)
            if case["content_type"] == "audio":
                clip_id = _assign_audio_clip(archetype_name, rng)
                if clip_id:
                    case["audio_clip_id"] = clip_id

            case["rationale"] = build_rationale(case, archetype_name)
            return case

    raise ValueError(f"Archetype '{archetype_name}' failed constraints after 200 attempts - fix ranges")
