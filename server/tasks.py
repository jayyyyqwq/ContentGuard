# tasks.py
# ──────────────────────────────────────────────────────────
# 7 difficulty tiers mapping to 14 case archetypes.
# Covers all 4 verdict types and both 2026-era archetypes.
# ──────────────────────────────────────────────────────────

DIFFICULTY_ARCHETYPE_MAP: dict[str, str] = {
    "easy":         "verbatim_commercial",
    "easy_medium":  "educational_excerpt",
    "medium":       "parody_high_overlap",
    "medium_hard":  "creative_commons_misapplication",
    "hard":         "ai_audio_reconstruction",
    "hard_expert":  "multi_claimant_non_overlapping",
    "expert":       "live_sports_gameplay_disguise",
}

TASK_CONFIGS: dict[str, dict] = {
    "easy": {
        "archetype":        "verbatim_commercial",
        "expected_actions": 3,
        "expected_score":   0.94,
        "correct_verdict":  "remove",
        "description": (
            "Clear-cut verbatim commercial repost. All four fair-use factors "
            "point toward removal. Solvable with query_rights_db → "
            "assess_transformation → decide('remove')."
        ),
    },
    "easy_medium": {
        "archetype":        "educational_excerpt",
        "expected_actions": 3,
        "expected_score":   0.82,
        "correct_verdict":  "keep",
        "description": (
            "Short educational clip (15-35% overlap) from a non-commercial "
            "channel. Transformation is moderate (0.50-0.70) and purpose is "
            "clearly educational. Agent must verify non-commercial context "
            "and transformation level before deciding 'keep'."
        ),
    },
    "medium": {
        "archetype":        "parody_high_overlap",
        "expected_actions": 4,
        "expected_score":   0.60,
        "correct_verdict":  "escalate",
        "description": (
            "Parody content with high overlap (45-70%). Transformation index "
            "is high but so is the amount used — two fair-use factors pull in "
            "opposite directions. Correct verdict is 'escalate'. Agent must "
            "run assess_transformation AND check_fingerprint to distinguish "
            "this from a clear removal case."
        ),
    },
    "medium_hard": {
        "archetype":        "creative_commons_misapplication",
        "expected_actions": 3,
        "expected_score":   0.50,
        "correct_verdict":  "monetize",
        "description": (
            "Content using a Creative Commons license incorrectly — the "
            "license terms were violated (e.g. commercial use of NC-licensed "
            "work, or missing attribution). Not outright infringement, but "
            "the uploader cannot claim free use. Correct verdict is "
            "'monetize'. Agent must query rights DB to discover the license "
            "dispute and assess transformation to gauge severity."
        ),
    },
    "hard": {
        "archetype":        "ai_audio_reconstruction",
        "expected_actions": 4,
        "expected_score":   0.38,
        "correct_verdict":  "remove",
        "description": (
            "2026 archetype: AI-reconstructed melody. fingerprint_match=0 "
            "(evades Content ID) but composition_similarity_score ≈ 0.80 "
            "(composition copyright infringed). Agent that relies on "
            "fingerprint alone calls 'keep' and receives approximately -1.04."
        ),
    },
    "hard_expert": {
        "archetype":        "multi_claimant_non_overlapping",
        "expected_actions": 3,
        "expected_score":   0.30,
        "correct_verdict":  "escalate",
        "description": (
            "Multiple rights holders (2-3) with non-overlapping claims and "
            "conflict_flag=1. DB confidence is low (0.45-0.64). Agent must "
            "query_rights_db to surface the conflict, then cross_ref_history "
            "to check prior disputes. Deciding without resolving the conflict "
            "incurs -0.40 process penalty. Correct verdict is 'escalate'."
        ),
    },
    "expert": {
        "archetype":        "live_sports_gameplay_disguise",
        "expected_actions": 4,
        "expected_score":   0.20,
        "correct_verdict":  "remove",
        "description": (
            "2026 archetype: live sports broadcast disguised as gameplay via "
            "HUD overlay. Two rights holders with conflict_flag=1 and high "
            "DB confidence (0.85-0.95). Overlap is very high (82-95%) but "
            "the HUD overlay makes automated detection unreliable. Agent "
            "must check_fingerprint (reveals the overlay deception), "
            "query_rights_db (surfaces dual rights holders), AND "
            "check_usage_context before deciding 'remove'. This is the "
            "hardest task — it tests whether the agent can see through "
            "deliberate obfuscation while navigating multi-party claims."
        ),
    },
}
