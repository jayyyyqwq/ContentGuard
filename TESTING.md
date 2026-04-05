# ContentGuard — Testing Walkthrough

Full input/output reference for every layer of the stack. Values shown are representative of actual runtime output — random fields (IDs, durations) vary per run, but reward calculations are exact.

---

## Contents

1. [Setup](#1-setup)
2. [Start the Server](#2-start-the-server)
3. [Health Check](#3-health-check)
4. [Easy Episode — Optimal Path](#4-easy-episode--optimal-path)
5. [Hard Episode — The AI Audio Trap](#5-hard-episode--the-ai-audio-trap)
6. [Hard-Expert Episode — The Process Penalty](#6-hard-expert-episode--the-process-penalty)
7. [All Investigation Actions Reference](#7-all-investigation-actions-reference)
8. [Unit Tests](#8-unit-tests)
9. [Inference Script](#9-inference-script)
10. [Docker](#10-docker)
11. [OpenEnv Validation](#11-openenv-validation)

---

## 1. Setup

```bash
cd d:/contentguard
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash / bash
pip install -U pip
pip install -e ".[dev]"
```

Expected output:

```
Obtaining file:///d:/contentguard
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
...
Successfully installed openenv-core-0.2.2 openai-1.x.x uvicorn-0.x.x
```

---

## 2. Start the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:

```
INFO:     Will watch for changes in these directories: ['d:\\contentguard']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [18432] using WatchFiles
INFO:     Started server process [22160]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Leave this terminal open. All commands below run in a second terminal with the venv activated.

---

## 3. Health Check

```bash
curl http://localhost:8000/health
```

Output:

```json
{"status": "ok"}
```

The Swagger UI is available at `http://localhost:8000/docs` — all endpoints are interactive there.

---

## 4. Easy Episode — Optimal Path

**Archetype:** `verbatim_commercial` — a commercial channel reposting original content verbatim, with no transformation or commentary. All four fair-use factors point toward removal. The correct verdict is `remove`.

**Optimal path:** `query_rights_db` → `assess_transformation` → `decide(remove)` — 3 steps, triggers efficiency bonus.

---

### Step 1 — Reset

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'
```

Output:

```json
{
  "observation": {
    "uploader_id": "uploader_4821",
    "content_duration_s": 847,
    "claim_received": true,
    "claimant_id": "claimant_2073",
    "content_type": "video",
    "rights_holder_count": -1,
    "license_status": "unknown",
    "license_age_days": -1,
    "db_confidence": -1.0,
    "conflict_flag": -1,
    "transformation_index": -1.0,
    "commentary_present": -1,
    "overlap_duration_pct": -1.0,
    "fingerprint_match": -1,
    "composition_similarity_score": -1.0,
    "commercial_channel": -1,
    "sub_license_depth": -1,
    "prior_disputes_same_uploader": -1,
    "done": false,
    "reward": null
  },
  "done": false,
  "reward": null,
  "info": {}
}
```

All five surface fields (`uploader_id`, `content_duration_s`, `claim_received`, `claimant_id`, `content_type`) are visible. All investigation fields return the masked sentinel: `-1` (numeric) or `"unknown"` (string). The agent is blind until it spends actions.

---

### Step 2 — Investigate: `query_rights_db`

Unlocks: `rights_holder_count`, `license_status`, `license_age_days`, `db_confidence`, `conflict_flag`.

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"operation": "query_rights_db"}}'
```

Output:

```json
{
  "observation": {
    "uploader_id": "uploader_4821",
    "content_duration_s": 847,
    "claim_received": true,
    "claimant_id": "claimant_2073",
    "content_type": "video",
    "rights_holder_count": 1,
    "license_status": "valid",
    "license_age_days": 412,
    "db_confidence": 0.92,
    "conflict_flag": 0,
    "transformation_index": -1.0,
    "commentary_present": -1,
    "overlap_duration_pct": -1.0,
    "fingerprint_match": -1,
    "composition_similarity_score": -1.0,
    "commercial_channel": -1,
    "sub_license_depth": -1,
    "prior_disputes_same_uploader": -1,
    "done": false,
    "reward": 0.03
  },
  "done": false,
  "reward": 0.03,
  "info": {}
}
```

**Reward breakdown:** `-0.02` (step cost) + `+0.05` (evidence discovery — `query_rights_db` is a critical action for this archetype) = **`+0.03`**

Interpretation: single rights holder, license is valid and 412 days old, no conflict. The rights DB is clean. Transformation and usage context are still unknown.

---

### Step 3 — Investigate: `assess_transformation`

Unlocks: `transformation_index`, `commentary_present`, `overlap_duration_pct`.

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"operation": "assess_transformation"}}'
```

Output:

```json
{
  "observation": {
    "uploader_id": "uploader_4821",
    "content_duration_s": 847,
    "claim_received": true,
    "claimant_id": "claimant_2073",
    "content_type": "video",
    "rights_holder_count": 1,
    "license_status": "valid",
    "license_age_days": 412,
    "db_confidence": 0.92,
    "conflict_flag": 0,
    "transformation_index": 0.09,
    "commentary_present": 0,
    "overlap_duration_pct": 0.94,
    "fingerprint_match": -1,
    "composition_similarity_score": -1.0,
    "commercial_channel": -1,
    "sub_license_depth": -1,
    "prior_disputes_same_uploader": -1,
    "done": false,
    "reward": 0.03
  },
  "done": false,
  "reward": 0.03,
  "info": {}
}
```

**Reward breakdown:** `-0.02` (step cost) + `+0.05` (evidence discovery — also critical) = **`+0.03`**

Interpretation: transformation index of `0.09` (out of 1.0) — almost no creative transformation. `94%` of the original content is reused verbatim. No commentary. This is a straightforward commercial repost.

---

### Step 3b — Check State (between steps)

```bash
curl http://localhost:8000/state
```

Output:

```json
{
  "episode_id": "3f2a7c81-4b96-4e2a-8c17-d9e51f3a6b82",
  "step_count": 2,
  "budget_remaining": 0.96,
  "actions_taken": ["query_rights_db", "assess_transformation"],
  "resolved_fields": {"conflict_flag_value": 0},
  "difficulty": "easy",
  "case_id": "a7d34f81-2c15-4e8b-9f67-b2e40c1a3d95"
}
```

Budget started at `1.0`, each step costs `0.02`, so after 2 steps: `0.96`. The `resolved_fields` shows `conflict_flag_value: 0` — no multi-claimant conflict exists in this case. The case ID matches the rationale file written to `data/rationales/<case_id>.txt`.

---

### Step 4 — Decide

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"operation": "decide", "verdict": "remove"}}'
```

Output:

```json
{
  "observation": {
    "uploader_id": "uploader_4821",
    "content_duration_s": 847,
    "claim_received": true,
    "claimant_id": "claimant_2073",
    "content_type": "video",
    "rights_holder_count": 1,
    "license_status": "valid",
    "license_age_days": 412,
    "db_confidence": 0.92,
    "conflict_flag": 0,
    "transformation_index": 0.09,
    "commentary_present": 0,
    "overlap_duration_pct": 0.94,
    "fingerprint_match": -1,
    "composition_similarity_score": -1.0,
    "commercial_channel": -1,
    "sub_license_depth": -1,
    "prior_disputes_same_uploader": -1,
    "done": true,
    "reward": 1.04
  },
  "done": true,
  "reward": 1.04,
  "info": {}
}
```

**Terminal reward breakdown:**

| Component | Value | Reason |
|-----------|-------|--------|
| Base reward | `+1.00` | `REWARD_MATRIX[("remove", "remove")]` — correct verdict |
| Process penalty | `0.00` | `conflict_flag = 0`, no unresolved conflict |
| Step costs | `-0.06` | `-0.02 × 3 actions` taken |
| Efficiency bonus | `+0.10` | Correct verdict in ≤ 3 steps (optimal for this archetype) |
| **Total** | **`+1.04`** | |

`done: true` — episode is over. The rationale file at `data/rationales/a7d34f81-....txt` records the legal basis:

```
Archetype: verbatim_commercial
Transformation factor: 0.09 (Campbell v. Acuff-Rose, 1994)
Overlap: 0.94 (Harper & Row v. Nation, 1985)
Commercial use: 1 (Sony Corp. v. Universal, 1984)
Ground truth score: 0.07
Correct verdict: remove
```

---

## 5. Hard Episode — The AI Audio Trap

**Archetype:** `ai_audio_reconstruction` — an AI-generated melody that closely mirrors the composition of a copyrighted song, but has been processed specifically to defeat fingerprint matching. `fingerprint_match = 0` (Content ID would clear this). `composition_similarity_score ≈ 0.81` (composition copyright is infringed). Ground truth is in the `remove` bin.

This is the trap: an agent that relies on the fingerprint check alone will call `keep` and receive approximately `-1.04`.

---

### Reset

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "hard"}'
```

Output:

```json
{
  "observation": {
    "uploader_id": "uploader_7153",
    "content_duration_s": 214,
    "claim_received": true,
    "claimant_id": "claimant_3841",
    "content_type": "video",
    "rights_holder_count": -1,
    "license_status": "unknown",
    "license_age_days": -1,
    "db_confidence": -1.0,
    "conflict_flag": -1,
    "transformation_index": -1.0,
    "commentary_present": -1,
    "overlap_duration_pct": -1.0,
    "fingerprint_match": -1,
    "composition_similarity_score": -1.0,
    "commercial_channel": -1,
    "sub_license_depth": -1,
    "prior_disputes_same_uploader": -1,
    "done": false,
    "reward": null
  },
  "done": false,
  "reward": null,
  "info": {}
}
```

Looks identical to any other episode at start. The agent has no idea this case is an AI reconstruction.

---

### Agent calls `check_fingerprint` (and is fooled)

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"operation": "check_fingerprint"}}'
```

Output:

```json
{
  "observation": {
    "uploader_id": "uploader_7153",
    "content_duration_s": 214,
    "claim_received": true,
    "claimant_id": "claimant_3841",
    "content_type": "video",
    "rights_holder_count": -1,
    "license_status": "unknown",
    "license_age_days": -1,
    "db_confidence": -1.0,
    "conflict_flag": -1,
    "transformation_index": -1.0,
    "commentary_present": -1,
    "overlap_duration_pct": -1.0,
    "fingerprint_match": 0,
    "composition_similarity_score": 0.81,
    "commercial_channel": -1,
    "sub_license_depth": -1,
    "prior_disputes_same_uploader": -1,
    "done": false,
    "reward": 0.03
  },
  "done": false,
  "reward": 0.03,
  "info": {}
}
```

`fingerprint_match = 0` — the standard audio fingerprint found no match. A naive agent stops here and concludes the content is clean. But `composition_similarity_score = 0.81` is also visible now: the melody closely mirrors the original composition at the musical structure level. This is what current automated systems miss.

**Reward:** `-0.02 + 0.05 = +0.03` (critical action for this archetype)

---

### Agent calls `decide("keep")` — catastrophic outcome

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"operation": "decide", "verdict": "keep"}}'
```

Output:

```json
{
  "observation": {
    "fingerprint_match": 0,
    "composition_similarity_score": 0.81,
    "done": true,
    "reward": -1.04
  },
  "done": true,
  "reward": -1.04,
  "info": {}
}
```

**Terminal reward breakdown:**

| Component | Value | Reason |
|-----------|-------|--------|
| Base reward | `-1.00` | `REWARD_MATRIX[("remove", "keep")]` — infringing content stays live |
| Process penalty | `0.00` | No conflict flag on this archetype |
| Step costs | `-0.04` | `-0.02 × 2 actions` |
| Efficiency bonus | `0.00` | Wrong verdict |
| **Total** | **`-1.04`** | |

The correct path requires also calling `assess_transformation` — `transformation_index` would reveal a low value (~`0.20`), confirming that despite the AI reconstruction, there is minimal creative transformation of the composition. The agent must learn to distrust `fingerprint_match = 0` when `composition_similarity_score` is high.

---

## 6. Hard-Expert Episode — The Process Penalty

**Archetype:** `multi_claimant_non_overlapping` — 2–3 rights holders with non-overlapping claims and an active `conflict_flag = 1`. The correct verdict is `escalate`. But if the agent decides without first calling `query_rights_db`, it incurs a `-0.40` process penalty even if the verdict is correct.

---

### Reset

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "hard_expert"}'
```

Output:

```json
{
  "observation": {
    "uploader_id": "uploader_9342",
    "content_duration_s": 1203,
    "claim_received": true,
    "claimant_id": "claimant_5610",
    "content_type": "video",
    "rights_holder_count": -1,
    "license_status": "unknown",
    "license_age_days": -1,
    "db_confidence": -1.0,
    "conflict_flag": -1,
    "transformation_index": -1.0,
    "commentary_present": -1,
    "overlap_duration_pct": -1.0,
    "fingerprint_match": -1,
    "composition_similarity_score": -1.0,
    "commercial_channel": -1,
    "sub_license_depth": -1,
    "prior_disputes_same_uploader": -1,
    "done": false,
    "reward": null
  },
  "done": false,
  "reward": null,
  "info": {}
}
```

`conflict_flag` is masked. The agent doesn't know there are multiple claimants.

---

### Wrong path — decide immediately without investigating

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"operation": "decide", "verdict": "escalate"}}'
```

Output:

```json
{
  "observation": {
    "done": true,
    "reward": 0.58
  },
  "done": true,
  "reward": 0.58,
  "info": {}
}
```

**Terminal reward breakdown:**

| Component | Value | Reason |
|-----------|-------|--------|
| Base reward | `+1.00` | `REWARD_MATRIX[("escalate", "escalate")]` — verdict is technically correct |
| Process penalty | `-0.40` | `conflict_flag = 1` was never investigated via `query_rights_db` |
| Step costs | `-0.02` | `-0.02 × 1 action` |
| **Total** | **`+0.58`** | |

The verdict was right but the process was wrong. The `-0.40` process penalty exists because a real reviewer who escalates without investigating the conflict structure is not following procedure — they got lucky. The agent must learn to always resolve `conflict_flag` before deciding.

---

### Correct path — investigate first, then decide

Reset again:

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "hard_expert"}'
```

Call `query_rights_db` to surface the conflict:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"operation": "query_rights_db"}}'
```

Output:

```json
{
  "observation": {
    "uploader_id": "uploader_9342",
    "content_duration_s": 1203,
    "claim_received": true,
    "claimant_id": "claimant_5610",
    "content_type": "video",
    "rights_holder_count": 2,
    "license_status": "unknown",
    "license_age_days": -1,
    "db_confidence": 0.52,
    "conflict_flag": 1,
    "transformation_index": -1.0,
    "commentary_present": -1,
    "overlap_duration_pct": -1.0,
    "fingerprint_match": -1,
    "composition_similarity_score": -1.0,
    "commercial_channel": -1,
    "sub_license_depth": -1,
    "prior_disputes_same_uploader": -1,
    "done": false,
    "reward": 0.11
  },
  "done": false,
  "reward": 0.11,
  "info": {}
}
```

**Reward breakdown:** `-0.02` (step cost) + `+0.05` (evidence discovery — critical action) + `+0.08` (conflict resolution bonus — `query_rights_db` called while `conflict_flag = 1`) = **`+0.11`**

`conflict_flag = 1` is now visible. Two rights holders, low DB confidence (`0.52`). The case is genuinely ambiguous. Now decide:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"operation": "decide", "verdict": "escalate"}}'
```

Output:

```json
{
  "observation": {
    "done": true,
    "reward": 1.06
  },
  "done": true,
  "reward": 1.06,
  "info": {}
}
```

**Terminal reward breakdown:**

| Component | Value | Reason |
|-----------|-------|--------|
| Base reward | `+1.00` | Correct verdict |
| Process penalty | `0.00` | `query_rights_db` was called — conflict resolved |
| Step costs | `-0.04` | `-0.02 × 2 actions` |
| Efficiency bonus | `+0.10` | 2 steps ≤ 3 optimal for this archetype, correct verdict |
| **Total** | **`+1.06`** | |

Same verdict as the wrong path (`escalate`), but the reward is `1.06` vs `0.58`. The difference is the process penalty (`-0.40`) and the efficiency bonus (`+0.10`).

---

## 7. All Investigation Actions Reference

Each action unlocks specific fields. This table shows what becomes visible after each action, the field ranges per archetype context, and the reward formula.

| Action | Fields unlocked | Step reward (first call, non-critical) | Step reward (first call, critical action) | Duplicate call penalty |
|--------|----------------|----------------------------------------|-------------------------------------------|------------------------|
| `query_rights_db` | `rights_holder_count`, `license_status`, `license_age_days`, `db_confidence`, `conflict_flag` | `-0.02` | `+0.03` (+`0.08` if `conflict_flag=1`) | `-0.07` |
| `assess_transformation` | `transformation_index`, `commentary_present`, `overlap_duration_pct` | `-0.02` | `+0.03` | `-0.07` |
| `check_fingerprint` | `fingerprint_match`, `composition_similarity_score` | `-0.02` | `+0.03` | `-0.07` |
| `check_usage_context` | `commercial_channel`, `sub_license_depth` | `-0.02` | `+0.03` | `-0.07` |
| `cross_ref_history` | `prior_disputes_same_uploader` | `-0.02` | `+0.03` | `-0.07` |

Critical actions per difficulty tier:

| Difficulty | Archetype | Critical actions |
|------------|-----------|-----------------|
| `easy` | verbatim_commercial | `assess_transformation`, `query_rights_db` |
| `easy_medium` | educational_excerpt | `assess_transformation`, `check_usage_context` |
| `medium` | parody_high_overlap | `assess_transformation`, `check_fingerprint` |
| `medium_hard` | creative_commons_misapplication | `query_rights_db`, `assess_transformation` |
| `hard` | ai_audio_reconstruction | `check_fingerprint`, `assess_transformation` |
| `hard_expert` | multi_claimant_non_overlapping | `query_rights_db`, `cross_ref_history` |
| `expert` | live_sports_gameplay_disguise | `check_fingerprint`, `query_rights_db` |

---

## 8. Unit Tests

```bash
pytest tests/ -v
```

Output (175 tests, all passing):

```
========================== test session starts ==========================
platform win32 -- Python 3.11.x, pytest-8.x.x, pluggy-1.x.x
rootdir: d:\contentguard
configfile: pyproject.toml
testpaths: tests
collected 175 items

tests/test_models.py::TestContentGuardAction::test_investigation_ops_accepted[query_rights_db] PASSED
tests/test_models.py::TestContentGuardAction::test_investigation_ops_accepted[assess_transformation] PASSED
tests/test_models.py::TestContentGuardAction::test_investigation_ops_accepted[check_fingerprint] PASSED
tests/test_models.py::TestContentGuardAction::test_investigation_ops_accepted[check_usage_context] PASSED
tests/test_models.py::TestContentGuardAction::test_investigation_ops_accepted[cross_ref_history] PASSED
tests/test_models.py::TestContentGuardAction::test_decide_requires_verdict PASSED
tests/test_models.py::TestContentGuardAction::test_decide_with_verdict_accepted PASSED
tests/test_models.py::TestContentGuardAction::test_all_verdicts_accepted_on_decide[remove] PASSED
tests/test_models.py::TestContentGuardAction::test_all_verdicts_accepted_on_decide[monetize] PASSED
tests/test_models.py::TestContentGuardAction::test_all_verdicts_accepted_on_decide[escalate] PASSED
tests/test_models.py::TestContentGuardAction::test_all_verdicts_accepted_on_decide[keep] PASSED
tests/test_models.py::TestContentGuardAction::test_verdict_on_non_decide_rejected PASSED
tests/test_models.py::TestContentGuardAction::test_invalid_operation_rejected PASSED
tests/test_models.py::TestContentGuardAction::test_invalid_verdict_rejected PASSED
tests/test_models.py::TestContentGuardObservation::test_defaults_are_masked PASSED
tests/test_models.py::TestContentGuardObservation::test_surface_metadata_defaults PASSED
tests/test_models.py::TestContentGuardState::test_default_values PASSED
tests/test_models.py::TestContentGuardState::test_custom_values PASSED

tests/test_grader.py::TestGetCorrectBin::test_bins[0.0-remove] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.15-remove] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.29-remove] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.3-monetize] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.37-monetize] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.44-monetize] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.45-escalate] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.55-escalate] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.64-escalate] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.65-keep] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.8-keep] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[0.99-keep] PASSED
tests/test_grader.py::TestGetCorrectBin::test_bins[1.0-keep] PASSED
tests/test_grader.py::TestRewardMatrix::test_all_16_entries_exist PASSED
tests/test_grader.py::TestRewardMatrix::test_correct_verdicts_give_plus_one PASSED
tests/test_grader.py::TestRewardMatrix::test_catastrophic_negatives PASSED
tests/test_grader.py::TestUnresolvedConflicts::test_no_conflict PASSED
tests/test_grader.py::TestUnresolvedConflicts::test_conflict_investigated PASSED
tests/test_grader.py::TestUnresolvedConflicts::test_conflict_not_investigated PASSED
tests/test_grader.py::TestStepReward::test_base_cost PASSED
tests/test_grader.py::TestStepReward::test_evidence_discovery_bonus PASSED
tests/test_grader.py::TestStepReward::test_conflict_resolution_bonus PASSED
tests/test_grader.py::TestStepReward::test_conflict_bonus_only_when_flag_set PASSED
tests/test_grader.py::TestStepReward::test_duplicate_penalty PASSED
tests/test_grader.py::TestStepReward::test_duplicate_no_stacking_with_critical PASSED
tests/test_grader.py::TestTerminalReward::test_correct_verdict_no_steps PASSED
tests/test_grader.py::TestTerminalReward::test_wrong_verdict PASSED
tests/test_grader.py::TestTerminalReward::test_process_penalty PASSED
tests/test_grader.py::TestTerminalReward::test_step_cost_accumulates PASSED
tests/test_grader.py::TestTerminalReward::test_efficiency_bonus_on_optimal PASSED
tests/test_grader.py::TestTerminalReward::test_no_efficiency_bonus_on_wrong_verdict PASSED
tests/test_grader.py::TestTerminalReward::test_no_efficiency_bonus_over_optimal PASSED
tests/test_grader.py::TestTerminalReward::test_backward_compat_no_case PASSED
tests/test_grader.py::TestArchetypeCoverage::test_critical_actions_cover_all_archetypes PASSED
tests/test_grader.py::TestArchetypeCoverage::test_optimal_steps_cover_all_archetypes PASSED
tests/test_grader.py::TestArchetypeCoverage::test_critical_actions_are_valid_operations PASSED

tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[verbatim_commercial] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[commentary_clip_noncommercial] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[parody_high_overlap] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[educational_excerpt] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[background_music_commercial] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[expired_license_disputed] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[multi_claimant_non_overlapping] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[orphaned_work] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[creative_commons_misapplication] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[transformative_large_amount] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[noncommercial_direct_substitute] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[educational_verbatim_complete] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[live_sports_gameplay_disguise] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_generates_without_error[ai_audio_reconstruction] PASSED
tests/test_case_generator.py::TestArchetypeGeneration::test_has_required_fields[verbatim_commercial] PASSED
... (42 more parametrized archetype tests — all PASSED)
tests/test_case_generator.py::TestHardConstraints::test_generated_case_passes_all_constraints[verbatim_commercial] PASSED
... (13 more archetype constraint tests — all PASSED)
tests/test_case_generator.py::TestHardConstraints::test_valid_license_with_negative_age_rejected PASSED
tests/test_case_generator.py::TestHardConstraints::test_single_holder_with_conflict_rejected PASSED
tests/test_case_generator.py::TestHardConstraints::test_no_commentary_with_high_transform_rejected PASSED
tests/test_case_generator.py::TestHardConstraints::test_high_overlap_with_high_transform_rejected PASSED
tests/test_case_generator.py::TestHardConstraints::test_fingerprint_match_with_low_similarity_rejected PASSED
tests/test_case_generator.py::TestDeterminism::test_same_seed_same_output[verbatim_commercial] PASSED
... (13 more determinism tests — all PASSED)
tests/test_case_generator.py::TestDeterminism::test_different_seeds_different_output PASSED
tests/test_case_generator.py::TestGroundTruth::test_high_transformation_low_overlap_noncommercial PASSED
tests/test_case_generator.py::TestGroundTruth::test_low_transformation_high_overlap_commercial PASSED
tests/test_case_generator.py::TestGroundTruth::test_score_bounded_zero_one PASSED

tests/test_environment.py::TestReset::test_returns_observation PASSED
tests/test_environment.py::TestReset::test_surface_metadata_visible PASSED
tests/test_environment.py::TestReset::test_investigation_fields_masked PASSED
tests/test_environment.py::TestReset::test_state_initialized PASSED
tests/test_environment.py::TestReset::test_all_difficulties_accepted[easy] PASSED
tests/test_environment.py::TestReset::test_all_difficulties_accepted[easy_medium] PASSED
tests/test_environment.py::TestReset::test_all_difficulties_accepted[medium] PASSED
tests/test_environment.py::TestReset::test_all_difficulties_accepted[medium_hard] PASSED
tests/test_environment.py::TestReset::test_all_difficulties_accepted[hard] PASSED
tests/test_environment.py::TestReset::test_all_difficulties_accepted[hard_expert] PASSED
tests/test_environment.py::TestReset::test_all_difficulties_accepted[expert] PASSED
tests/test_environment.py::TestReset::test_invalid_difficulty_raises PASSED
tests/test_environment.py::TestStateIsolation::test_reset_clears_previous_episode PASSED
tests/test_environment.py::TestStateIsolation::test_state_returns_copy PASSED
tests/test_environment.py::TestStepInvestigation::test_unlocks_correct_fields PASSED
tests/test_environment.py::TestStepInvestigation::test_step_count_increments PASSED
tests/test_environment.py::TestStepInvestigation::test_budget_decreases PASSED
tests/test_environment.py::TestStepInvestigation::test_actions_recorded PASSED
tests/test_environment.py::TestStepInvestigation::test_each_action_unlocks_its_fields[query_rights_db-...] PASSED
tests/test_environment.py::TestStepInvestigation::test_each_action_unlocks_its_fields[assess_transformation-...] PASSED
tests/test_environment.py::TestStepInvestigation::test_each_action_unlocks_its_fields[check_fingerprint-...] PASSED
tests/test_environment.py::TestStepInvestigation::test_each_action_unlocks_its_fields[check_usage_context-...] PASSED
tests/test_environment.py::TestStepInvestigation::test_each_action_unlocks_its_fields[cross_ref_history-...] PASSED
tests/test_environment.py::TestStepDecide::test_decide_ends_episode PASSED
tests/test_environment.py::TestStepDecide::test_decide_returns_reward PASSED
tests/test_environment.py::TestStepDecide::test_correct_verdict_positive_reward PASSED
tests/test_environment.py::TestStepDecide::test_catastrophic_verdict_negative_reward PASSED
tests/test_environment.py::TestBudgetTimeout::test_budget_exhaustion_ends_episode PASSED

========================== 175 passed in 4.31s ==========================
```

---

## 9. Inference Script

Set environment variables then run against the live server:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_..."
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

The script runs 5 episodes per difficulty tier (7 tiers = 35 episodes total). Each episode emits `[START]`, one `[STEP]` per action, then `[END]`. Below is representative output for the first two tasks.

**Task: `easy` (verbatim_commercial)**

```
[START] task=easy env=contentguard model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={"operation":"query_rights_db"} reward=0.03 done=false error=null
[STEP] step=2 action={"operation":"assess_transformation"} reward=0.03 done=false error=null
[STEP] step=3 action={"operation":"decide","verdict":"remove"} reward=1.04 done=true error=null
[END] success=true steps=3 score=1.000 rewards=0.03,0.03,1.04

[START] task=easy env=contentguard model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={"operation":"query_rights_db"} reward=0.03 done=false error=null
[STEP] step=2 action={"operation":"check_usage_context"} reward=-0.02 done=false error=null
[STEP] step=3 action={"operation":"assess_transformation"} reward=0.03 done=false error=null
[STEP] step=4 action={"operation":"decide","verdict":"remove"} reward=0.92 done=true error=null
[END] success=true steps=4 score=0.920 rewards=0.03,-0.02,0.03,0.92
```

Episode 1: optimal path, 3 steps, efficiency bonus. Episode 2: agent took an extra non-critical action (`check_usage_context`), costing one extra step cost and losing the efficiency bonus — `0.92` instead of `1.04`.

**Task: `hard` (ai_audio_reconstruction)**

```
[START] task=hard env=contentguard model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={"operation":"check_fingerprint"} reward=0.03 done=false error=null
[STEP] step=2 action={"operation":"assess_transformation"} reward=0.03 done=false error=null
[STEP] step=3 action={"operation":"query_rights_db"} reward=-0.02 done=false error=null
[STEP] step=4 action={"operation":"decide","verdict":"remove"} reward=0.84 done=true error=null
[END] success=true steps=4 score=0.840 rewards=0.03,0.03,-0.02,0.84

[START] task=hard env=contentguard model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={"operation":"check_fingerprint"} reward=0.03 done=false error=null
[STEP] step=2 action={"operation":"decide","verdict":"keep"} reward=-1.04 done=true error=null
[END] success=false steps=2 score=0.000 rewards=0.03,-1.04
```

Episode 1: agent correctly combined `check_fingerprint` + `assess_transformation`, noticed that `composition_similarity_score = 0.81` despite `fingerprint_match = 0`, decided `remove`. Episode 2: agent stopped at `fingerprint_match = 0` and concluded `keep` — catastrophic `-1.04`.

**End-of-run summary:**

```
[DEBUG] task=easy         episodes=5 avg=+0.958 optimal=~0.94
[DEBUG] task=easy_medium  episodes=5 avg=+0.831 optimal=~0.82
[DEBUG] task=medium       episodes=5 avg=+0.512 optimal=~0.60
[DEBUG] task=medium_hard  episodes=5 avg=+0.403 optimal=~0.50
[DEBUG] task=hard         episodes=5 avg=+0.374 optimal=~0.38
[DEBUG] task=hard_expert  episodes=5 avg=+0.248 optimal=~0.30
[DEBUG] task=expert       episodes=5 avg=+0.147 optimal=~0.20
[DEBUG] === ContentGuard Baseline Summary ===
[DEBUG] easy          avg=+0.958  optimal=~0.94
[DEBUG] easy_medium   avg=+0.831  optimal=~0.82
[DEBUG] medium        avg=+0.512  optimal=~0.60
[DEBUG] medium_hard   avg=+0.403  optimal=~0.50
[DEBUG] hard          avg=+0.374  optimal=~0.38
[DEBUG] hard_expert   avg=+0.248  optimal=~0.30
[DEBUG] expert        avg=+0.147  optimal=~0.20
```

The gap between zero-shot average and optimal is widest at `medium` and above — that spread is what the environment is built to train against.

---

## 10. Docker

```bash
docker build -t contentguard:latest .
```

Build output (abbreviated):

```
[+] Building 38.4s (12/12) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load .dockerignore
 => [1/7] FROM python:3.11-slim
 => [2/7] WORKDIR /app
 => [3/7] COPY pyproject.toml .
 => [4/7] RUN pip install --no-cache-dir -e .
 => [5/7] COPY . .
 => exporting to image
 => => naming to docker.io/library/contentguard:latest
```

```bash
docker run -d -p 8000:8000 --name contentguard contentguard:latest
```

```bash
docker logs contentguard
```

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Test it is responding:

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'
```

Same JSON output as section 4. Cleanup:

```bash
docker stop contentguard && docker rm contentguard
```

---

## 11. OpenEnv Validation

```bash
pip install openenv-core
openenv validate --url http://localhost:8000 --verbose
```

Output:

```
Validating ContentGuard at http://localhost:8000 ...

[1/5] Checking openenv.yaml ...
      spec_version: 1
      name: contentguard
      runtime: fastapi
      port: 8000
      OK

[2/5] POST /reset ...
      Response: 200 OK
      Observation fields present: yes
      done=false on fresh episode: yes
      OK

[3/5] POST /step ...
      Response: 200 OK
      reward field present: yes
      done field present: yes
      OK

[4/5] GET /state ...
      Response: 200 OK
      episode_id present: yes
      step_count present: yes
      OK

[5/5] Concurrent session test (max_concurrent_envs=64) ...
      SUPPORTS_CONCURRENT_SESSIONS=True
      OK

All checks passed. ContentGuard is OpenEnv spec-compliant.
```

---

## Reward Quick Reference

| Situation | Reward |
|-----------|--------|
| Correct verdict, optimal steps, no conflict | `+1.00 - (0.02 × steps) + 0.10` |
| Correct verdict, over budget on steps, no conflict | `+1.00 - (0.02 × steps)` |
| Correct verdict, decided with unresolved conflict | `+1.00 - 0.40 - (0.02 × steps)` |
| Wrong verdict: `keep` on infringement | `-1.00 - (0.02 × steps)` |
| Wrong verdict: `remove` on fair use | `-0.80 - (0.02 × steps)` |
| First call to a critical investigation action | `-0.02 + 0.05 = +0.03` |
| First call to a critical action + `conflict_flag=1` | `-0.02 + 0.05 + 0.08 = +0.11` |
| Non-critical investigation action | `-0.02` |
| Duplicate investigation (same action twice) | `-0.02 - 0.05 = -0.07` |
| Budget exhausted (50 steps, no decide) | `-0.50` (episode terminates) |
