# ContentGuard — Full Product & Codebase Document

> **Audience:** Every team member, including those who have never seen the code.
> **Purpose:** Understand what ContentGuard is, how it works, and why every design decision was made.

---

## Table of Contents

1. [What is ContentGuard?](#1-what-is-contentguard)
2. [The Big Picture — How It All Fits Together](#2-the-big-picture)
3. [Directory Structure](#3-directory-structure)
4. [Core Concepts You Must Know](#4-core-concepts-you-must-know)
5. [The Data Models (models.py)](#5-the-data-models)
6. [How Cases Are Generated (case_generator.py)](#6-how-cases-are-generated)
7. [How the Environment Works (environment.py)](#7-how-the-environment-works)
8. [How Rewards Are Calculated (grader.py)](#8-how-rewards-are-calculated)
9. [Difficulty Tiers (tasks.py)](#9-difficulty-tiers)
10. [The Server (app.py)](#10-the-server)
11. [How Inference Works (inference.py)](#11-how-inference-works)
12. [End-to-End Walkthrough — One Full Episode](#12-end-to-end-walkthrough)
13. [The Test Suite](#13-the-test-suite)
14. [Configuration & Deployment](#14-configuration--deployment)
15. [Key Design Decisions & Why](#15-key-design-decisions--why)
16. [Quick Reference](#16-quick-reference)

---

## 1. What is ContentGuard?

ContentGuard is a **simulated copyright review environment** built for the Meta × Hugging Face OpenEnv Hackathon.

The real-world problem it simulates: every day, social platforms receive thousands of copyright claims. A rights holder (say, a music label) claims that a video uploaded by a creator infringes their copyright. A human reviewer — or an AI agent — must investigate the claim and decide one of four things:

| Verdict | Meaning |
|---------|---------|
| `remove` | The content infringes copyright. Take it down. |
| `keep` | The content is fair use. Leave it up. |
| `monetize` | The rights holder should get paid, but content stays up. |
| `escalate` | Too ambiguous. Send to a human expert. |

ContentGuard trains AI agents to do this job well — under time pressure (a budget), with incomplete information (they start blind), and with real consequences for wrong answers (asymmetric penalties).

**In short:** It is a reinforcement learning (RL) environment where an agent plays the role of a content policy adjudicator.

---

## 2. The Big Picture

Here is the full data flow in one picture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ContentGuard Server                       │
│                                                                  │
│  ┌─────────────┐    reset()    ┌──────────────────────────────┐ │
│  │             │ ──────────── ▶│   Case Generator             │ │
│  │   AI Agent  │               │   - Picks archetype          │ │
│  │ (inference) │               │   - Generates synthetic case │ │
│  │             │               │   - Returns masked obs.      │ │
│  │             │ ◀──────────── └──────────────────────────────┘ │
│  │             │   observation                                   │
│  │             │                                                  │
│  │             │    step()     ┌──────────────────────────────┐ │
│  │             │ ──────────── ▶│   Environment                │ │
│  │             │               │   - Unlocks fields           │ │
│  │             │               │   - Calls Grader             │ │
│  │             │               │   - Returns reward + obs.    │ │
│  │             │ ◀──────────── └──────────────────────────────┘ │
│  │             │   reward                                        │
│  └─────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Flow summary:**

1. Agent calls `reset()` → gets a copyright case with most fields hidden
2. Agent calls `step()` with an investigation action → gets more evidence revealed
3. Agent repeats step 2 until it has enough evidence
4. Agent calls `step()` with `decide` + a verdict → gets a final reward score
5. Episode ends. Start over for the next case.

---

## 3. Directory Structure

```
D:\contentguard\
│
├── inference.py            ← The AI agent that plays the game
├── models.py               ← Data types: what a case looks like, what actions look like
├── openenv.yaml            ← Tells HuggingFace how to run this server
├── Dockerfile              ← Container definition for deployment
├── pyproject.toml          ← Python package + dependency list
├── README.md               ← User-facing instructions
│
├── server/                 ← The game engine
│   ├── app.py              ← HTTP server setup (FastAPI)
│   ├── environment.py      ← Game loop: reset(), step()
│   ├── case_generator.py   ← Creates synthetic copyright cases
│   ├── grader.py           ← Calculates rewards
│   └── tasks.py            ← Maps difficulty levels to case types
│
├── tests/                  ← Automated tests
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_grader.py
│   ├── test_case_generator.py
│   └── test_environment.py
│
└── data/
    └── rationales/         ← Auto-generated legal summaries per case (gitignored)
```

---

## 4. Core Concepts You Must Know

### 4.1 What is a "Case"?

A **case** is a synthetic copyright dispute. It contains:

- Who uploaded the content (`uploader_id`)
- Who is claiming copyright (`claimant_id`)
- Properties of the content (duration, type)
- Evidence about the rights situation (license status, transformation level, overlap %)
- The **ground truth** — a score from 0.0 to 1.0 representing how much fair use applies
- The **correct verdict** — what a perfect reviewer would decide

### 4.2 What is an "Observation"?

An **observation** is what the agent *can see* at any moment. At the start of an episode, most fields are hidden (masked to `-1` or `"unknown"`). The agent must take investigation actions to reveal them.

### 4.3 What is an "Action"?

An **action** is what the agent does on each turn. There are two kinds:

- **Investigation actions** — reveal hidden evidence (cost: 0.02 budget each)
- **Terminal action** (`decide`) — issue a verdict and end the episode

### 4.4 What is "Budget"?

Each episode starts with a budget of `1.0`. Every action costs `0.02`. If the budget runs out before the agent decides, the episode ends with a penalty of `-0.50`. This simulates real-world time pressure.

### 4.5 What is a "Reward"?

A number the agent receives after each action. Positive = good. Negative = bad. The agent's goal is to maximize total reward over the episode.

- During investigation: small rewards (+0.03 for useful actions, −0.02 for wasted ones)
- At decision time: large reward based on how correct the verdict is (up to +1.04, down to −1.04)

### 4.6 What is "Reinforcement Learning" here?

The agent plays thousands of episodes. After each one, it learns from the reward signal — which actions led to good outcomes, which led to bad ones. Over time, it learns an investigation strategy: which evidence to gather first, when to escalate, and how to avoid catastrophic verdicts.

---

## 5. The Data Models

**File:** `models.py`

All data in ContentGuard is typed using **Pydantic** — a Python library that validates data structures. There are three main models.

---

### 5.1 ContentGuardAction — What the agent sends

```
operation: one of:
  "query_rights_db"       → Check the rights database
  "assess_transformation" → Measure how transformative the content is
  "check_fingerprint"     → Run audio/video fingerprint matching
  "check_usage_context"   → Check if it's commercial or non-commercial
  "cross_ref_history"     → Look up the uploader's dispute history
  "decide"                → Issue a final verdict (ends the episode)

verdict: required only when operation is "decide"
  one of: "remove", "keep", "monetize", "escalate"
```

**Validation rule:** If `operation` is `"decide"`, a `verdict` must be provided. Otherwise, it must not. Pydantic enforces this automatically.

---

### 5.2 ContentGuardObservation — What the agent sees

This is the agent's view of the case at any moment.

**Always visible (surface metadata):**

| Field | Type | What it means |
|-------|------|---------------|
| `uploader_id` | string | ID of the content creator |
| `content_duration_s` | int | Length of content in seconds |
| `claim_received` | bool | Whether a copyright claim was filed |
| `claimant_id` | string | ID of the rights holder filing the claim |
| `content_type` | string | "video", "audio", etc. |

**Hidden until `query_rights_db` is called:**

| Field | Default | What it reveals |
|-------|---------|-----------------|
| `rights_holder_count` | -1 | How many rights holders exist |
| `license_status` | "unknown" | "valid", "expired", "disputed", or "unknown" |
| `license_age_days` | -1 | How old the license is |
| `db_confidence` | -1.0 | How confident the database is (0–1) |
| `conflict_flag` | -1 | `1` if multiple claimants are in dispute |

**Hidden until `assess_transformation` is called:**

| Field | Default | What it reveals |
|-------|---------|-----------------|
| `transformation_index` | -1.0 | How transformative the content is (0–1) |
| `commentary_present` | -1 | `1` if commentary is present |
| `overlap_duration_pct` | -1.0 | What % of the original is used (0–1) |

**Hidden until `check_fingerprint` is called:**

| Field | Default | What it reveals |
|-------|---------|-----------------|
| `fingerprint_match` | -1 | `1` if content ID system matched |
| `composition_similarity_score` | -1.0 | Musical composition similarity (0–1) |

**Hidden until `check_usage_context` is called:**

| Field | Default | What it reveals |
|-------|---------|-----------------|
| `commercial_channel` | -1 | `1` if the uploader earns money from this channel |
| `sub_license_depth` | -1 | License complexity level (0–2) |

**Hidden until `cross_ref_history` is called:**

| Field | Default | What it reveals |
|-------|---------|-----------------|
| `prior_disputes_same_uploader` | -1 | How many past disputes this uploader has |

---

### 5.3 ContentGuardState — Episode metadata

This tracks the episode's progress. The agent can read it at any time via `GET /state`.

| Field | What it tracks |
|-------|----------------|
| `episode_id` | Unique ID for this episode |
| `step_count` | How many actions have been taken |
| `budget_remaining` | How much budget is left (starts at 1.0) |
| `actions_taken` | List of all actions taken so far, in order |
| `resolved_fields` | Whether conflict was investigated |
| `difficulty` | The difficulty tier of this episode |
| `case_id` | Unique ID for the generated case |

---

## 6. How Cases Are Generated

**File:** `server/case_generator.py`

This is the synthetic data engine. It creates copyright cases from scratch — no real data needed.

### 6.1 The 14 Archetypes

ContentGuard has 14 **archetypes** — templates that represent common real-world copyright scenarios.

| Archetype | Correct Verdict | What it represents |
|-----------|-----------------|-------------------|
| `verbatim_commercial` | remove | Someone re-uploaded someone else's content to make money |
| `commentary_clip_noncommercial` | keep | Non-commercial review/commentary using a small clip |
| `parody_high_overlap` | escalate | Parody with significant overlap — genuinely ambiguous |
| `educational_excerpt` | keep | Short clip in a non-commercial educational video |
| `background_music_commercial` | remove | Copyrighted music playing in a commercial video |
| `expired_license_disputed` | escalate | License expired + dispute = needs human review |
| `multi_claimant_non_overlapping` | escalate | Multiple rights holders with conflicting claims |
| `orphaned_work` | escalate | No known rights holder — who even owns this? |
| `creative_commons_misapplication` | monetize | CC license was misused — rights holder should get paid |
| `transformative_large_amount` | escalate | Very transformative but large portion used — borderline |
| `noncommercial_direct_substitute` | remove | Non-commercial but it fully replaces the original |
| `educational_verbatim_complete` | monetize | Verbatim use of full work in educational context |
| `live_sports_gameplay_disguise` *(2026)* | remove | HUD overlay trick to defeat automated detection |
| `ai_audio_reconstruction` *(2026)* | remove | AI rebuilt the audio to defeat fingerprinting — but composition matches |

Each archetype defines:
- The **range** for each evidence field (e.g., transformation index between 0.70 and 0.90)
- The **correct verdict**
- The **ground truth score range** (the fair-use score that maps to this verdict)

### 6.2 The Generation Process

When `generate_case(archetype_name, seed)` is called:

1. **Seed the random number generator.** Same seed = same case every time. This is critical for reproducibility in automated evaluation.

2. **Create random IDs** — uploader_id, claimant_id, case_id, content_duration_s.

3. **Sample each field from the archetype's defined range.** For example, for `verbatim_commercial`, the transformation_index is sampled between 0.00 and 0.15 (very low — verbatim means no transformation).

4. **Derive computed fields.** The `db_confidence` score is computed from how many rights holders exist, whether the license is valid/disputed/expired, etc.

5. **Check hard constraints.** Five logical rules prevent incoherent cases:
   - A valid license cannot have a negative age
   - Only 1 rights holder cannot be in conflict with itself
   - High transformation (>0.75) implies commentary must be present
   - Very high overlap (>0.90) cannot coexist with very high transformation (>0.60)
   - A fingerprint match cannot have a very low composition similarity score

   If any constraint fails, the generator tries again (up to 200 attempts).

6. **Compute the ground truth score.** This is a weighted formula based on U.S. fair-use law:

   ```
   score = transformation × 0.35
         + nature_of_use × 0.15
         + amount_used × 0.20
         + market_effect × 0.30
   ```

   Where:
   - **transformation** = transformation_index (higher = more transformative = more fair use)
   - **nature_of_use** = 0.7 if non-commercial, 0.3 if commercial
   - **amount_used** = 1.0 − overlap_duration_pct (lower overlap = less taken = more fair use)
   - **market_effect** = 1.0 − (overlap × commercial_factor) (commercial use of most content harms market most)

   The four weights come from landmark U.S. fair-use cases:
   - Transformation factor: *Campbell v. Acuff-Rose* (1994)
   - Nature of use: *Sony Corp. v. Universal* (1984)
   - Amount taken: *Harper & Row v. Nation* (1985)
   - Market effect: *Stewart v. Abend* (1990)

7. **Write a rationale file** to `data/rationales/<case_id>.txt` explaining the legal reasoning.

8. **Return the case dict** with all fields + ground_truth + correct_verdict.

### 6.3 Why Synthetic Generation?

- **Infinite supply** — no API rate limits, no data licensing issues
- **Deterministic** — same seed = same case, critical for automation pipelines
- **Fully controlled** — impossible incoherent combinations are rejected by constraints
- **Extensible** — swap in real data (MusicBrainz, AudD, etc.) without changing anything else

---

## 7. How the Environment Works

**File:** `server/environment.py`

This is the game engine. It manages the state of a single episode and handles the agent's actions.

### 7.1 reset(seed, episode_id, difficulty)

Called once at the start of every episode.

**What it does:**
1. Validates that `difficulty` is one of the 7 valid tiers
2. Maps difficulty → archetype name (via `tasks.py`)
3. Calls `generate_case(archetype, seed)` to create the case
4. Creates a fresh `ContentGuardState`:
   - budget_remaining = 1.0
   - step_count = 0
   - actions_taken = []
5. Creates a masked `ContentGuardObservation` — all investigation fields at their defaults (-1 or "unknown")
6. Returns the observation — **the agent starts blind**

### 7.2 step(action)

Called every time the agent takes an action.

**What it does:**

```
1. Increment step_count
2. Deduct 0.02 from budget_remaining

3. Did budget run out?
   ├── YES → done=True, reward=-0.50 (timeout penalty)
   └── NO  → continue

4. Is this a "decide" action?
   ├── YES → call terminal_reward(), done=True
   └── NO  → it's an investigation action → continue

5. Investigation action:
   ├── Look up which fields this action reveals (ACTION_UNLOCKS map)
   ├── Copy those field values from the case into the observation
   ├── Call step_reward() for mid-episode shaping
   └── Return updated observation + reward, done=False
```

**ACTION_UNLOCKS map:**

| Action | Fields it reveals |
|--------|-------------------|
| `query_rights_db` | rights_holder_count, license_status, license_age_days, db_confidence, conflict_flag |
| `assess_transformation` | transformation_index, commentary_present, overlap_duration_pct |
| `check_fingerprint` | fingerprint_match, composition_similarity_score |
| `check_usage_context` | commercial_channel, sub_license_depth |
| `cross_ref_history` | prior_disputes_same_uploader |

### 7.3 State Isolation

The `state` property returns a **deep copy** of the internal state. This means external code cannot accidentally modify the game's internal state by changing a reference. Required by the OpenEnv spec.

---

## 8. How Rewards Are Calculated

**File:** `server/grader.py`

The grader is the heart of the training signal. It answers: "how good was that action?"

### 8.1 Verdict Bins — Mapping Score to Verdict

The ground truth score (0.0–1.0) maps to a verdict like this:

| Score Range | Correct Verdict | Interpretation |
|-------------|-----------------|----------------|
| 0.00 – 0.30 | `remove` | Content is infringing |
| 0.30 – 0.45 | `monetize` | Rights holder should be paid |
| 0.45 – 0.65 | `escalate` | Too ambiguous for automated decision |
| 0.65 – 1.00 | `keep` | Fair use — content stays |

### 8.2 The Reward Matrix — Terminal Rewards

When the agent issues a verdict, it gets a reward based on (correct_verdict, agent_verdict):

|  | Agent: remove | Agent: keep | Agent: monetize | Agent: escalate |
|--|---------------|-------------|-----------------|-----------------|
| **True: remove** | **+1.00** | **−1.00** | −0.50 | +0.20 |
| **True: keep** | **−0.80** | **+1.00** | −0.30 | +0.60 |
| **True: monetize** | +0.30 | −0.40 | **+1.00** | +0.60 |
| **True: escalate** | +0.20 | +0.20 | +0.10 | **+1.00** |

**Key asymmetries (intentional):**

- `remove` → `keep` = **−1.00** (worst possible). Infringing content left live → platform legal risk.
- `keep` → `remove` = **−0.80** (very bad). Wrongful removal → creator harm, but appeal exists.
- `escalate` is always "safe" but never optimal. An agent that always escalates gets +0.20 per correct case but misses the +1.00 bonus.
- `escalate` from true `keep` = +0.60. Being cautious is better than being wrong.

**No clamping.** The agent must feel the full force of catastrophic mistakes.

### 8.3 Step Rewards — Mid-Episode Shaping

Every investigation action gets a small reward calculated as:

```
base_cost = −0.02  (budget pressure)

+ evidence_discovery_bonus = +0.05  (if this is a critical action AND first time calling it)
+ conflict_resolution_bonus = +0.08  (if query_rights_db AND case has conflict_flag=1)
− duplicate_action_penalty = −0.05  (if calling the same action more than once)
```

**Critical actions per archetype** — each archetype has 2 "most important" investigation actions. Calling them first gives a bonus.

Examples:
- `verbatim_commercial` → critical: `assess_transformation`, `query_rights_db`
- `ai_audio_reconstruction` → critical: `check_fingerprint`, `assess_transformation`
- `live_sports_gameplay_disguise` → critical: `check_fingerprint`, `query_rights_db`

**Step reward examples:**

| Scenario | Calculation | Result |
|----------|-------------|--------|
| First `query_rights_db` on verbatim_commercial | −0.02 + 0.05 | **+0.03** |
| `query_rights_db` on multi-claimant case | −0.02 + 0.05 + 0.08 | **+0.11** |
| Second `query_rights_db` (duplicate) | −0.02 − 0.05 | **−0.07** |
| Non-critical action | −0.02 | **−0.02** |

### 8.4 Terminal Reward — Full Calculation

When the agent calls `decide`:

```
terminal_reward = base
                + process_penalty
                + accumulated_step_costs
                + efficiency_bonus
```

Where:
- **base** = from the reward matrix above
- **process_penalty** = −0.40 if the case has `conflict_flag=1` AND `query_rights_db` was never called
- **accumulated_step_costs** = −0.02 × number_of_actions_taken
- **efficiency_bonus** = +0.10 if verdict is correct AND steps taken ≤ optimal steps for this archetype

**Example: Perfect easy episode**

```
Case: verbatim_commercial
Actions: query_rights_db, assess_transformation, decide("remove")
Ground truth: 0.10 → correct bin = "remove"

base           = +1.00  (correct)
process        = 0.00   (no conflict)
step costs     = −0.02 × 3 = −0.06
efficiency     = +0.10  (3 steps ≤ optimal 3, correct verdict)
─────────────────────────────────────
Total          = +1.04
```

**Example: Catastrophic mistake**

```
Case: verbatim_commercial
Actions: check_fingerprint, decide("keep")
Ground truth: 0.10 → correct bin = "remove"

base           = −1.00  (remove → keep, catastrophic)
process        = 0.00
step costs     = −0.02 × 2 = −0.04
efficiency     = 0.00   (wrong verdict)
─────────────────────────────────────
Total          = −1.04
```

**Example: Correct verdict but skipped conflict investigation**

```
Case: multi_claimant_non_overlapping (conflict_flag=1)
Actions: assess_transformation, decide("escalate")
Ground truth: 0.55 → correct bin = "escalate"

base           = +1.00  (correct verdict)
process        = −0.40  (conflict_flag=1, never called query_rights_db)
step costs     = −0.02 × 2 = −0.04
efficiency     = 0.00   (2 steps < optimal 3, but irrelevant — process penalty disqualifies)
─────────────────────────────────────
Total          = +0.56  (correct but penalized for skipping due process)
```

---

## 9. Difficulty Tiers

**File:** `server/tasks.py`

Seven difficulty levels, each mapped to an archetype:

| Difficulty | Archetype | Correct Verdict | Optimal Steps | Zero-Shot LLM Score | Why it's hard |
|------------|-----------|-----------------|---------------|---------------------|---------------|
| `easy` | verbatim_commercial | remove | 3 | ~0.72 | Clear-cut case |
| `easy_medium` | educational_excerpt | keep | 3 | ~0.65 | Context matters |
| `medium` | parody_high_overlap | escalate | 4 | ~0.51 | Two factors conflict |
| `medium_hard` | creative_commons_misapplication | monetize | 3 | ~0.40 | Not removal, just monetize |
| `hard` | ai_audio_reconstruction | remove | 4 | ~0.38 | Fingerprint=0 is a trap |
| `hard_expert` | multi_claimant_non_overlapping | escalate | 3 | ~0.25 | Conflict must be investigated |
| `expert` | live_sports_gameplay_disguise | remove | 4 | ~0.15 | Obfuscation defeats simple detection |

"Zero-shot LLM score" = how well a top LLM does with no training on this task. Expert is brutal.

**Random baseline** (always escalate): ~0.18 across all tiers.

---

## 10. The Server

**File:** `server/app.py`

ContentGuard runs as an HTTP server using **FastAPI**, created by the OpenEnv framework's `create_app()` factory.

### API Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| `POST` | `/reset` | Start a new episode. Body: `{"difficulty": "easy"}`. Returns masked observation. |
| `POST` | `/step` | Take an action. Body: `{"action": {"operation": "...", "verdict": "..."}}`. Returns updated observation + reward. |
| `GET` | `/state` | Get current episode state (step count, budget, actions taken, etc.) |
| `GET` | `/metadata` | Get environment name, description, version. |
| `GET` | `/health` | Health check. Returns `{"status": "ok"}`. |
| `WS` | `/ws` | WebSocket for persistent session (required by OpenEnv spec). |
| `GET` | `/docs` | Interactive Swagger UI for manual testing. |

### Concurrency

`SUPPORTS_CONCURRENT_SESSIONS = True` — the server can handle 64 simultaneous episodes. Required for HuggingFace Spaces where multiple workers run in parallel.

---

## 11. How Inference Works

**File:** `inference.py`

This is the **baseline AI agent** — it plays ContentGuard using a large language model (LLM). This is what gets submitted to the hackathon evaluator.

### 11.1 What It Uses

- **LLM:** Llama-3.3-70B-Instruct (or any model via environment variable)
- **API:** OpenAI-compatible client pointing to HuggingFace's inference router
- **Mode:** Zero-shot — the LLM is not fine-tuned; it reads the prompt and decides

### 11.2 Environment Variables Required

```bash
API_BASE_URL   # The LLM API endpoint
MODEL_NAME     # Model identifier (e.g., meta-llama/Llama-3.3-70B-Instruct)
HF_TOKEN       # HuggingFace API key for authentication
ENV_BASE_URL   # ContentGuard server URL (default: http://localhost:8000)
```

### 11.3 The Prompt

For every step, the agent receives a prompt like this:

```
You are a content policy adjudicator for a major social platform.
Investigate this rights dispute and make a defensible policy decision.

Current case state:
  uploader_id: uploader_4821
  content_duration_s: 847
  claim_received: True
  claimant_id: claimant_2073
  content_type: video
  rights_holder_count: NOT YET INVESTIGATED
  license_status: NOT YET INVESTIGATED
  transformation_index: NOT YET INVESTIGATED
  fingerprint_match: NOT YET INVESTIGATED
  ...

Remaining action budget: 0.94

Available actions:
  query_rights_db, assess_transformation, check_fingerprint,
  check_usage_context, cross_ref_history, decide (requires verdict)

Rules:
- Investigate fields showing "NOT YET INVESTIGATED" before deciding on ambiguous cases.
- Never call decide() while conflict_flag is unknown or unresolved.
- When calling decide, specify verdict: remove | monetize | escalate | keep.
- Respond with a single JSON object only.

Your response:
```

The LLM responds with JSON:
```json
{"operation": "query_rights_db"}
```
or
```json
{"operation": "decide", "verdict": "remove"}
```

### 11.4 Fallback Behavior

- If the LLM response is unparseable JSON → fallback to `query_rights_db`
- If the LLM API call fails → fallback to `{"operation": "decide", "verdict": "escalate"}`

This ensures the inference loop never crashes.

### 11.5 Episode Loop

```python
for difficulty in ["easy", "easy_medium", "medium", "medium_hard", "hard", "hard_expert", "expert"]:
    for episode in range(5):
        obs = env.reset(difficulty=difficulty)
        done = False
        while not done:
            prompt = build_prompt(obs, budget)
            response = llm.complete(prompt)
            action = parse_action(response)
            obs, reward, done = env.step(action)
        print(f"[END] difficulty={difficulty} score={total_reward}")
```

7 difficulties × 5 episodes each = **35 episodes per run.**

### 11.6 Log Format

Every step is logged in a machine-parseable format:

```
[START] task=easy env=contentguard model=meta-llama/Llama-3.3-70B-Instruct
[STEP]  step=1 action={"operation":"query_rights_db"} reward=0.03 done=false error=null
[STEP]  step=2 action={"operation":"assess_transformation"} reward=0.03 done=false error=null
[STEP]  step=3 action={"operation":"decide","verdict":"remove"} reward=1.04 done=true error=null
[END]   success=true steps=3 score=1.04 rewards=0.03,0.03,1.04
```

---

## 12. End-to-End Walkthrough

Let's trace a full easy episode manually.

### Setup

The server is running. A client (the agent) calls `POST /reset` with `{"difficulty": "easy"}`.

**Internal flow:**

1. `tasks.py` maps `"easy"` → `"verbatim_commercial"` archetype
2. `case_generator.py` generates a case:
   - transformation_index = 0.08 (very low — almost no transformation)
   - overlap_duration_pct = 0.94 (94% of original used)
   - license_status = "valid"
   - conflict_flag = 0
   - ground_truth = 0.10 (falls in "remove" bin: 0.00–0.30)
   - correct_verdict = "remove"
3. `environment.py` creates masked observation — all investigation fields = -1 or "unknown"
4. Server returns the masked observation

---

### Step 1: Agent calls `query_rights_db`

**Request:** `POST /step` `{"action": {"operation": "query_rights_db"}}`

**Internal flow:**

1. step_count → 1, budget_remaining → 0.98
2. ACTION_UNLOCKS reveals: rights_holder_count=1, license_status="valid", license_age_days=523, db_confidence=0.92, conflict_flag=0
3. `grader.step_reward("query_rights_db", case, state)`:
   - base = −0.02
   - critical action for `verbatim_commercial`? YES → +0.05
   - conflict flag? 0 → no bonus
   - Result: +0.03
4. Return updated observation (rights DB fields now visible), reward=0.03, done=false

**Agent now knows:** License is valid, 1 rights holder, no conflict.

---

### Step 2: Agent calls `assess_transformation`

**Request:** `POST /step` `{"action": {"operation": "assess_transformation"}}`

**Internal flow:**

1. step_count → 2, budget_remaining → 0.96
2. Reveals: transformation_index=0.08, commentary_present=0, overlap_duration_pct=0.94
3. `grader.step_reward(...)`:
   - base = −0.02
   - critical action? YES → +0.05
   - Result: +0.03
4. Return updated observation, reward=0.03, done=false

**Agent now knows:** 8% transformation, no commentary, 94% overlap. This is clearly infringing.

---

### Step 3: Agent calls `decide` with verdict `"remove"`

**Request:** `POST /step` `{"action": {"operation": "decide", "verdict": "remove"}}`

**Internal flow:**

1. step_count → 3, budget_remaining → 0.94
2. Terminal action detected
3. `grader.terminal_reward("remove", ground_truth=0.10, state, case)`:
   - correct_bin = "remove" (0.10 is in 0.00–0.30 range)
   - base = REWARD_MATRIX[("remove", "remove")] = +1.00
   - process penalty = 0.0 (no conflict)
   - step costs = −0.02 × 3 = −0.06
   - efficiency bonus = +0.10 (verdict correct AND 3 steps ≤ optimal 3)
   - Total = 1.00 + 0.0 − 0.06 + 0.10 = **+1.04**
4. Return final observation, reward=1.04, done=true

**Episode summary:** 3 steps, total reward = 0.03 + 0.03 + 1.04 = **+1.10**

---

## 13. The Test Suite

**Folder:** `tests/`

ContentGuard has four test files with 40+ tests.

### test_models.py

Tests Pydantic validation:
- `decide` without a verdict → error
- `decide` with a valid verdict → passes
- Investigation action with a verdict → error
- Invalid operation names → error
- Observation defaults are correct (all masked)

### test_grader.py

Tests every reward calculation:
- All 4 verdict bins map correctly from ground truth scores
- All 16 reward matrix entries are correct values
- Evidence discovery bonus only triggers on first call
- Duplicate penalty short-circuits other bonuses
- Conflict resolution bonus only triggers when conflict_flag=1
- Process penalty triggers when conflict unresolved at terminal
- Efficiency bonus only triggers when verdict is correct + steps ≤ optimal

### test_case_generator.py

Tests synthetic case generation:
- All 14 archetypes generate without errors
- All 5 hard constraints are enforced (bad cases get rejected)
- Same seed → identical output (determinism)
- Different seeds → different output (randomness)
- Ground truth formula: high transformation + low overlap + non-commercial → keep range

### test_environment.py

Tests the game loop:
- `reset()` returns masked observation
- Each of the 5 investigation actions reveals exactly the right fields
- step_count and budget_remaining update correctly
- Actions are recorded in order
- `decide` ends the episode (done=True)
- Correct verdict → positive reward, catastrophic verdict → negative reward
- Budget timeout (51 steps) → done=True, reward=−0.50
- `state` property returns a deep copy (mutations don't leak)
- Invalid difficulty → ValueError

### Running the Tests

```bash
cd D:\contentguard
pytest tests/ -v
```

---

## 14. Configuration & Deployment

### Local Development

```bash
cd D:\contentguard

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash

# Install with dev dependencies
pip install -e ".[dev]"

# Start the server
python -m server.app
# Server is now at http://localhost:8000

# In another terminal, run inference
API_BASE_URL=https://... MODEL_NAME=meta-llama/... HF_TOKEN=hf_... python inference.py
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `openenv-core[core]>=0.2.2` | OpenEnv framework (FastAPI wrapper, base classes) |
| `openai>=1.0.0` | OpenAI-compatible client for calling the LLM |
| `uvicorn[standard]>=0.24.0` | ASGI server to run FastAPI |
| `pytest>=7.0` *(dev only)* | Test runner |

### Docker

```bash
# Build
docker build -t contentguard:latest .

# Run
docker run -d -p 8000:8000 --name contentguard contentguard:latest

# Server is now at http://localhost:8000
```

The Dockerfile uses `python:3.11-slim`, installs dependencies via pip, and runs `uvicorn server.app:app --host 0.0.0.0 --port 8000`.

### HuggingFace Spaces

1. Create a new Space (Docker runtime)
2. Push this repo to the Space
3. HuggingFace reads `openenv.yaml` and starts the server on port 8000
4. Set Secrets: `HF_TOKEN`, `MODEL_NAME`, `API_BASE_URL`
5. Validate: `openenv validate --url https://<space>.hf.space --verbose`

### OpenEnv Validation Checklist

The hackathon evaluator checks:
- ✅ `POST /reset` → typed observation
- ✅ `POST /step` → typed observation with reward
- ✅ `GET /state` → typed state
- ✅ Pydantic models for Action, Observation, State
- ✅ `openenv.yaml` present and correct
- ✅ Concurrent sessions supported (up to 64)
- ✅ Deterministic grader (no LLM-as-judge)
- ✅ `Dockerfile` present
- ✅ `inference.py` at repo root

---

## 15. Key Design Decisions & Why

### Decision 1: Masked Observations

**What:** The agent starts each episode with most fields hidden. It must spend budget to reveal them.

**Why:** Real copyright review is an *investigation problem*, not a classification problem. The challenge is deciding *what to investigate* under time pressure. If the agent saw everything at once, the problem would be trivially easy.

---

### Decision 2: Synthetic Case Generation (No Real Data)

**What:** All cases are generated algorithmically from archetypes. No real copyright data is used.

**Why:**
- Deterministic: same seed = same case, needed for reproducible evaluation
- Infinite scale: no API calls, no rate limits, no cost
- Fully controlled: constraints prevent logically impossible cases

**Extensibility:** Swap `case_generator.py` with a real database lookup. Everything else stays the same.

---

### Decision 3: Asymmetric Reward Matrix

**What:** Not all wrong answers are equally bad. Keeping infringing content live (−1.00) is worse than wrongfully removing content (−0.80).

**Why:** This mirrors real platform policy. Infringing content creates legal liability for the platform. A wrongful removal is bad, but the creator can appeal and get reinstated. The agent must learn that some mistakes are catastrophic.

---

### Decision 4: Process Penalty for Unresolved Conflicts

**What:** If a case has `conflict_flag=1` (multiple claimants) and the agent never calls `query_rights_db`, it gets −0.40 even if the verdict is correct.

**Why:** Multi-claimant cases require due process. An agent that guesses correctly but skips the investigation is not trustworthy — it got lucky. This teaches agents to investigate conflicts properly, not just guess.

---

### Decision 5: No Reward Clamping

**What:** Rewards can go below −1.0 (e.g., −1.04 with step costs) or above +1.0 (e.g., +1.04 with efficiency bonus).

**Why:** Clamping hides the magnitude of mistakes. An agent should feel the full force of a catastrophic error. Unclamped rewards force the agent to learn which mistakes are fatal.

---

### Decision 6: Two 2026 Archetypes

**What:** `ai_audio_reconstruction` and `live_sports_gameplay_disguise` represent modern evasion techniques that defeat standard automated systems.

`ai_audio_reconstruction`:
- `fingerprint_match = 0` (evades Content ID)
- `composition_similarity_score = 0.82` (underlying composition is still infringing)
- An agent that relies only on fingerprinting decides "keep" → catastrophic

`live_sports_gameplay_disguise`:
- HUD overlay defeats standard detection
- Two rights holders + conflict_flag=1
- Requires both fingerprint check AND rights DB check

**Why:** Forces agents to reason beyond simple pattern matching. The "easy shortcut" (check fingerprint = done) fails on these cases.

---

### Decision 7: Mid-Episode Shaping

**What:** Investigation actions give small immediate rewards, not just the terminal reward.

**Why:** Without mid-episode shaping, the agent only learns from the final verdict. It has no signal about *which* investigations were useful vs. wasteful. Shaping teaches strategy step by step.

---

## 16. Quick Reference

### All 5 Investigation Actions

| Action | What it reveals | Optimal for |
|--------|-----------------|-------------|
| `query_rights_db` | License status, rights holders, conflict flag | Most cases |
| `assess_transformation` | How transformative + how much was taken | Parody, commentary, education cases |
| `check_fingerprint` | Audio/video fingerprint match + composition similarity | AI audio, sports disguise cases |
| `check_usage_context` | Commercial or not, license complexity | Commercial infringement cases |
| `cross_ref_history` | Past disputes by this uploader | Hard/expert cases |

### Reward Quick Reference

| Situation | Reward |
|-----------|--------|
| Correct verdict, optimal steps | up to +1.10 |
| Correct verdict, extra steps | +1.00 − (0.02 × extra_steps) |
| `escalate` when correct verdict exists | +0.20 to +0.60 |
| Wrong verdict | −0.30 to −0.80 |
| Infringing content left up (`remove` → `keep`) | −1.00 |
| Budget timeout | −0.50 |
| First critical investigation action | +0.03 |
| Conflict resolution investigation | +0.11 |
| Duplicate action | −0.07 |

### Verdict Decision Guide

| Ground Truth Score | Correct Verdict |
|--------------------|-----------------|
| 0.00 – 0.30 | `remove` |
| 0.30 – 0.45 | `monetize` |
| 0.45 – 0.65 | `escalate` |
| 0.65 – 1.00 | `keep` |

### File Map

| You want to change... | Edit this file |
|----------------------|----------------|
| What fields exist on a case / action / state | `models.py` |
| Add a new archetype | `server/case_generator.py` |
| Change the reward matrix | `server/grader.py` |
| Add a new difficulty tier | `server/tasks.py` |
| Change what each action reveals | `server/environment.py` |
| Change the agent's prompt | `inference.py` |
| Change the server port or concurrency | `server/app.py` |
| Change deployment config | `Dockerfile`, `openenv.yaml` |

---
