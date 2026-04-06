---
title: ContentGuard
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
pinned: false
license: mit
---

<div align="center">

# ⚖️ ContentGuard

**Built by Team Gan j Ai - Jay Choukikar & Gautam Soni**
*Meta × Hugging Face OpenEnv Hackathon*

[![Hackathon](https://img.shields.io/badge/Hackathon-Meta%20%C3%97%20Hugging%20Face-blue?style=for-the-badge)](https://github.com/jayyyyqwq/ContentGuard)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1%20Spec-green?style=for-the-badge)](https://github.com/jayyyyqwq/ContentGuard/blob/master/openenv.yaml)
[![FastAPI](https://img.shields.io/badge/Runtime-FastAPI-009688?style=for-the-badge)](https://github.com/jayyyyqwq/ContentGuard/blob/master/server/app.py)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?style=for-the-badge)](https://github.com/jayyyyqwq/ContentGuard/blob/master/Dockerfile)
[![Tests](https://img.shields.io/badge/Tests-191%20passing-brightgreen?style=for-the-badge)](https://github.com/jayyyyqwq/ContentGuard/tree/master/tests)

*An RL environment for content rights adjudication. Agents start blind, spend budget to investigate evidence channels, and issue defensible platform policy decisions the way a real Trust & Safety reviewer would.*

**[🚀 Live HF Space](https://huggingface.co/spaces/Jayyyy234/contentguard)** · **[📂 GitHub](https://github.com/jayyyyqwq/ContentGuard)**

</div>

---

![Episode Flow](assets/episode_flow.png)

---

## 🚫 The Problem → ✅ The Solution

Current automated systems are good at **one thing**: exact fingerprint matching. They cannot handle:

- 🎵 **AI-reconstructed audio** that evades Content ID but still infringes composition copyright
- 🎮 **Live sports disguised as gameplay** via HUD overlay
- ⚖️ **Multi-claimant disputes** where rights are split across jurisdictions

**ContentGuard** is the training ground that doesn't exist yet. A standardised RL environment where agents learn to *investigate*, *gather evidence*, and make **defensible policy calls** informed by the four-factor fair use rubric (17 U.S.C. § 107).

### The Four Verdicts

| Verdict | Meaning | When |
|---------|---------|------|
| `remove` | Infringing, take it down | Verbatim repost, commercial, no transformation |
| `monetize` | License violation, not piracy | CC misapplication, expired license |
| `escalate` | Ambiguous, route to human review | Competing claimants, contradictory evidence |
| `keep` | Fair use, no action | Commentary, education, high transformation |

---

## 🏗️ How the Agent Sees a Case

Each episode the agent starts with **only surface metadata**. Investigation fields are masked until explicitly unlocked:

```
ContentGuardObservation
│
├── Always visible
│   └── uploader_id · content_duration_s · content_type · claimant_id
│
├── query_rights_db      → license_status · conflict_flag · rights_holder_count · db_confidence
├── assess_transformation → transformation_index · commentary_present · overlap_duration_pct
├── check_fingerprint    → fingerprint_match · composition_similarity_score  ← key for AI audio
├── check_usage_context  → commercial_channel · sub_license_depth
└── cross_ref_history    → prior_disputes_same_uploader
```

**Budget pressure:** every step costs `-0.02`. The agent must decide *which* channels are worth querying, not just *what* to conclude.

---

## ⚖️ Asymmetric Reward Function

![Reward Matrix Heatmap](assets/reward_heatmap.png)

```
Total Reward = Base Verdict Reward  +  Process Penalty  +  Step Costs
```

| Component | Value | Trigger |
|-----------|-------|---------|
| Correct verdict | `+1.00` | Agent matches ground truth bin |
| Catastrophic miss | `-1.00` | Keep on infringing content |
| Wrongful takedown | `-0.80` | Remove on fair use |
| Process penalty | `-0.40` | Deciding while `conflict_flag` unresolved |
| Step cost | `-0.02` | Per action taken |
| Efficiency bonus | `+0.10` | Correct verdict within optimal step budget |
| Critical action | `+0.05` | First call to an archetype-relevant channel |

> Rewards are **not clamped**. The agent must learn that `-1.00` is catastrophic, not just "bad".

---

## 🔬 Case Generation: 14 Archetypes × Four-Factor Rubric

Cases are generated from **14 archetypes** using rejection sampling with hard logical constraints. Archetype first, derived fields second, constraints enforced. The design is intentionally synthetic: it runs with no external credentials, scales to any episode count at zero cost, and is fully seeded. Same seed always produces the same case, which is what automated evaluation pipelines need.

Hard constraints prevent logically impossible cases (e.g. `license_status == "valid"` with `license_age_days < 0`). Each archetype defines a `ground_truth_range` calibrated against the four-factor fair use rubric:

| Factor | Weight | Legal Basis |
| ------ | :----: | ----------- |
| Transformation | 35% | *Campbell v. Acuff-Rose* (1994) |
| Nature of use | 15% | *Sony Corp. v. Universal* (1984) |
| Amount used | 20% | *Harper & Row v. Nation* (1985) |
| Market effect | 30% | *Stewart v. Abend* (1990) |

The ground truth score is a weighted sum of these four factors. The score falls into a verdict bin (`remove < 0.30`, `monetize < 0.45`, `escalate < 0.65`, `keep ≥ 0.65`), and every generated case produces a rationale file in `data/rationales/` recording the legal basis for its score.

---

## 🎯 7 Difficulty Tiers

| Tier | Archetype | Correct Verdict | Optimal Steps | Target Score |
| ---- | --------- | --------------- | :-----------: | :----------: |
| `easy` | Verbatim commercial repost | `remove` | 3 | ~0.94 |
| `easy_medium` | Educational excerpt | `keep` | 3 | ~0.82 |
| `medium` | Parody with high overlap | `escalate` | 4 | ~0.60 |
| `medium_hard` | Creative Commons misapplication | `monetize` | 3 | ~0.50 |
| `hard` ⚠️ | **AI audio reconstruction** *(2026)* | `remove` | 4 | ~0.38 |
| `hard_expert` ⚠️ | **Multi-claimant, non-overlapping** | `escalate` | 3 | ~0.30 |
| `expert` ⚠️ | **Live sports disguised as gameplay** *(2026)* | `remove` | 4 | ~0.20 |

**⚠️ Two 2026 archetypes** designed specifically because current automated systems fail them:

- **AI audio** : `fingerprint_match = 0` (evades Content ID) but `composition_similarity_score ≈ 0.80`. An agent relying on fingerprint alone calls `keep` and receives **-1.04**.
- **Sports/gameplay disguise** : HUD overlay defeats detection. Dual rights holders with `conflict_flag = 1`. Deciding without `query_rights_db` incurs the `-0.40` process penalty.

---

## 📊 Baseline Scores

![Baseline Scores Chart](assets/baseline_scores.png)

| Tier | Zero-Shot LLM | Optimal | Gap (= training signal) |
| ---- | :-----------: | :-----: | :---------------------: |
| `easy` | 0.72 | 0.94 | 0.22 |
| `easy_medium` | 0.65 | 0.82 | 0.17 |
| `medium` | 0.51 | 0.60 | 0.09 |
| `medium_hard` | 0.40 | 0.50 | 0.10 |
| `hard` | 0.38 | 0.38 | 0.00 |
| `hard_expert` | 0.25 | 0.30 | 0.05 |
| `expert` | 0.15 | 0.20 | 0.05 |

Random baseline (always `escalate`): **~0.18** across all tiers. The zero-shot gap is the training signal ContentGuard is built to provide.

---

## ⚙️ Setup

### Local

```bash
git clone https://github.com/jayyyyqwq/ContentGuard && cd ContentGuard
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Baseline Inference

```bash
export HF_TOKEN="your_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
python inference.py
```

### Docker

```bash
docker build -t contentguard:latest .
docker run -d -p 8000:8000 contentguard:latest
openenv validate --url http://localhost:8000
```

---

## ✅ OpenEnv Spec Compliance

| Requirement | Status |
| ----------- | :----: |
| `POST /reset` → typed `ContentGuardObservation` | ✅ |
| `POST /step` → typed `ContentGuardObservation` | ✅ |
| `GET /state` → typed `ContentGuardState` | ✅ |
| `openenv validate` passes | ✅ |
| Concurrent WebSocket sessions | ✅ |
| Deterministic graders, no LLM-as-judge | ✅ |
| Dockerfile, no GPU, runs on 2 vCPU / 8 GB | ✅ |
| `inference.py` at project root | ✅ |
| 191 unit tests passing | ✅ |

---

## 🧠 Why This Design

**Why masked observations?** The hard part of real copyright review isn't knowing *what* to decide, it's knowing *what to investigate first* under budget constraints. Masking forces genuine sequential reasoning.

**Why no reward clamping?** A wrongful takedown and a missed infringement are both catastrophic but differently. Clamping to `[0, 1]` erases that signal.

**Why 14 archetypes instead of random?** Fair use cases cluster around recognizable patterns. Pure random sampling produces legally incoherent cases. Archetype-first + rejection sampling gives variance *within* valid structure.

**Why synthetic?** Full determinism, infinite scale, zero external dependencies. Same seed always produces the same case. Automated validators need reproducible scores.

---

<div align="center">
<i>"Don't just detect the fingerprint. Understand the context."</i><br><br>

**Team Gan j Ai**

| Role | Name |
| ---- | ---- |
| Team Lead | Jay Choukikar |
| Member | Gautam Soni |

</div>
