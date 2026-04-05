# inference.py
"""
ContentGuard baseline inference script.

MANDATORY environment variables:
  API_BASE_URL  — LLM API endpoint  (e.g. https://api.groq.com/openai/v1)
  MODEL_NAME    — Model identifier  (e.g. llama-3.3-70b-versatile)
  HF_TOKEN      — API key (Groq, HuggingFace, or OpenAI-compatible)

Optional env vars:
  ENV_BASE_URL  — ContentGuard env server (default: http://localhost:8000)

STDOUT FORMAT (machine-parsed by evaluator):
  [START] task=<task> env=contentguard model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Usage:
  python inference.py
"""

import os
import sys
import json
import statistics

from openai import OpenAI
from openenv.core import GenericEnvClient

# ── Mandatory competition variables ───────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is required. Set it to your API key.", flush=True)
    sys.exit(1)

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK        = "contentguard"
EPISODES_PER_TASK = 5
SUCCESS_THRESHOLD = 0.0   # terminal_reward > 0 = successful decision
MAX_EPISODE_STEPS = 10

llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── All 7 difficulty tiers ─────────────────────────────────────────────────────
TASKS = [
    ("easy",         "~0.94"),
    ("easy_medium",  "~0.82"),
    ("medium",       "~0.60"),
    ("medium_hard",  "~0.50"),
    ("hard",         "~0.38"),
    ("hard_expert",  "~0.30"),
    ("expert",       "~0.20"),
]

VALID_OPERATIONS = [
    "query_rights_db",
    "assess_transformation",
    "check_fingerprint",
    "check_usage_context",
    "cross_ref_history",
    "decide",
]
VALID_VERDICTS = ["remove", "monetize", "escalate", "keep"]

INVESTIGATION_OPS = [op for op in VALID_OPERATIONS if op != "decide"]

# ── System prompt with adjudication rubric ────────────────────────────────────

SYSTEM_PROMPT = """You are a content policy adjudicator for a major social platform.
You review copyright disputes by investigating evidence channels, then issuing a verdict.

## How the environment works
- You start each case seeing only surface metadata (uploader, claimant, duration, content type).
- All investigation fields begin MASKED (shown as -1 or "unknown").
- Each investigation action reveals specific fields. You have 5 investigation actions available:
  1. query_rights_db      -> rights_holder_count, license_status, license_age_days, db_confidence, conflict_flag
  2. assess_transformation -> transformation_index (0-1), commentary_present (0/1), overlap_duration_pct (0-1)
  3. check_fingerprint    -> fingerprint_match (0/1), composition_similarity_score (0-1)
  4. check_usage_context  -> commercial_channel (0/1), sub_license_depth
  5. cross_ref_history    -> prior_disputes_same_uploader

- Each action costs budget. Repeating an action wastes budget and is penalized.
- When you have enough evidence, call "decide" with a verdict.

## Verdict decision rules (apply these strictly based on revealed evidence)

remove (clear infringement):
  - fingerprint_match=1 with low transformation_index (<0.4) and commercial use
  - OR composition_similarity_score > 0.7 even if fingerprint_match=0 (AI-reconstructed audio)
  - OR verbatim repost: high overlap (>0.7), no commentary, commercial channel

monetize (license violation, not outright piracy):
  - license_status is "disputed" or "expired" with moderate transformation
  - Creative Commons misapplication (license exists but terms violated)
  - overlap is significant but content has some transformative value

escalate (ambiguous, needs human review):
  - conflict_flag=1 (multiple competing rights holders)
  - Evidence contradicts itself (high transformation BUT high overlap)
  - Orphaned work (rights_holder_count=0, unknown license)
  - When in genuine doubt after investigation

keep (fair use / non-infringing):
  - transformation_index > 0.6 AND commentary_present=1 AND commercial_channel=0
  - Educational use with moderate overlap (<0.35) and clear transformation
  - Low overlap, non-commercial, clear commentary/criticism purpose

## Investigation strategy
- Start with query_rights_db (reveals license status and conflict_flag — most universally useful).
- Then assess_transformation (reveals transformation level and overlap — the two biggest fair-use factors).
- Use check_fingerprint if you suspect audio copying or need to confirm/deny a match.
- Use check_usage_context if commercial vs non-commercial status matters for your decision.
- Use cross_ref_history only if you need prior dispute context.
- Decide as soon as you have enough evidence. Do NOT investigate all 5 channels every time.
- NEVER repeat an action you already called.
- If conflict_flag=1, you MUST have called query_rights_db before deciding (or you get a -0.40 penalty).

## Response format
Respond with a single JSON object. No explanation, no markdown, no extra text.

Examples:
  {"operation": "query_rights_db"}
  {"operation": "decide", "verdict": "remove"}
"""


# ── Mandatory log helpers (format must match exactly) ─────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation formatter ─────────────────────────────────────────────────────

def format_observation(obs: dict, actions_taken: list[str]) -> str:
    """Format the current observation into a readable message for the LLM."""
    lines = []
    for field, value in obs.items():
        if field in ("done", "reward", "metadata"):
            continue
        if value in (-1, -1.0, "unknown"):
            lines.append(f"  {field}: [MASKED]")
        else:
            lines.append(f"  {field}: {value}")

    obs_block = "\n".join(lines)

    already_used = [op for op in actions_taken if op in INVESTIGATION_OPS]
    remaining = [op for op in INVESTIGATION_OPS if op not in already_used]
    remaining_str = ", ".join(remaining) if remaining else "none — you MUST call decide now"

    return f"""Current case evidence:
{obs_block}

Actions already taken: {", ".join(actions_taken) if actions_taken else "none (first step)"}
Remaining investigation actions: {remaining_str}

Choose your next action. Respond with JSON only."""


# ── Action parser ──────────────────────────────────────────────────────────────

def parse_action(response_text: str) -> dict:
    """Parse LLM response into a ContentGuardAction dict. Falls back to safe default."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        op = data.get("operation", "query_rights_db")
        if op not in VALID_OPERATIONS:
            op = "query_rights_db"
        verdict = data.get("verdict") if op == "decide" else None
        if op == "decide" and verdict not in VALID_VERDICTS:
            verdict = "escalate"
        action: dict[str, str] = {"operation": op}
        if verdict is not None:
            action["verdict"] = verdict
        return action
    except Exception:
        return {"operation": "query_rights_db"}


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(difficulty: str, episode_num: int) -> float:
    """
    Run one full episode, emitting [START] / [STEP]* / [END] to stdout.
    Returns terminal reward.
    """
    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    terminal_reward = 0.0
    actions_taken: list[str] = []

    # Multi-turn conversation history
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        with GenericEnvClient(base_url=ENV_BASE_URL).sync() as env:
            result = env.reset(difficulty=difficulty, seed=episode_num)
            step = 0

            while not result.done:
                step += 1
                obs = result.observation
                error: str | None = None

                if step > MAX_EPISODE_STEPS:
                    action = {"operation": "decide", "verdict": "escalate"}
                else:
                    # Add current observation as user message
                    obs_message = format_observation(obs, actions_taken)
                    messages.append({"role": "user", "content": obs_message})

                    try:
                        llm_response = llm_client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            max_tokens=64,
                            temperature=0.0,
                        )
                        response_text = llm_response.choices[0].message.content or ""
                        action = parse_action(response_text)

                        # Add LLM's response to conversation history
                        messages.append({"role": "assistant", "content": response_text.strip()})
                    except Exception as exc:
                        action = {"operation": "decide", "verdict": "escalate"}
                        error = str(exc)[:120]

                    # Hard-enforce: never repeat an investigation action
                    if action["operation"] in actions_taken and action["operation"] in INVESTIGATION_OPS:
                        remaining = [op for op in INVESTIGATION_OPS if op not in actions_taken]
                        if remaining:
                            action = {"operation": remaining[0]}
                        else:
                            action = {"operation": "decide", "verdict": "escalate"}

                actions_taken.append(action["operation"])
                result = env.step(action)

                reward = result.reward or 0.0
                done = result.done
                action_str = json.dumps(action, separators=(",", ":"))

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        terminal_reward = rewards[-1] if rewards else 0.0

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        score = min(1.0, max(0.0, terminal_reward))
        success = terminal_reward > SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return terminal_reward


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    task_averages: dict[str, float] = {}

    for difficulty, expected_optimal in TASKS:
        episode_scores: list[float] = []

        for ep in range(EPISODES_PER_TASK):
            score = run_episode(difficulty, episode_num=ep)
            episode_scores.append(score)

        avg = round(statistics.mean(episode_scores), 3)
        task_averages[difficulty] = avg
        print(
            f"[DEBUG] task={difficulty} episodes={EPISODES_PER_TASK} avg={avg:+.3f} optimal={expected_optimal}",
            flush=True,
        )

    print("[DEBUG] === ContentGuard Baseline Summary ===", flush=True)
    for difficulty, expected_optimal in TASKS:
        avg = task_averages[difficulty]
        print(f"[DEBUG] {difficulty:<12}  avg={avg:+.3f}  optimal={expected_optimal}", flush=True)


if __name__ == "__main__":
    main()
