# inference.py
"""
ContentGuard baseline inference script.

MANDATORY environment variables:
  API_BASE_URL  — LLM API endpoint  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — Model identifier  (e.g. meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN      — Hugging Face / API key

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
import json
import statistics

from openai import OpenAI
from openenv.core import GenericEnvClient

# ── Mandatory competition variables ───────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")   # no default — must be set in environment

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK        = "contentguard"
EPISODES_PER_TASK = 5
SUCCESS_THRESHOLD = 0.0   # terminal_reward > 0 = successful decision

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


# ── Prompt builder ─────────────────────────────────────────────────────────────

def obs_to_prompt(obs: dict, budget: float) -> str:
    lines = []
    for field, value in obs.items():
        if field in ("done", "reward", "metadata"):
            continue
        if value in (-1, -1.0, "unknown"):
            lines.append(f"  {field}: NOT YET INVESTIGATED")
        else:
            lines.append(f"  {field}: {value}")

    obs_block = "\n".join(lines)
    actions_str = ", ".join(VALID_OPERATIONS)

    return f"""You are a content policy adjudicator for a major social platform.
Investigate this rights dispute and make a defensible policy decision.

Current case state:
{obs_block}

Remaining action budget: {budget:.2f}
Available actions: {actions_str}

Rules:
- Investigate fields showing "NOT YET INVESTIGATED" before deciding on ambiguous cases.
- Never call decide() while conflict_flag is unknown or unresolved.
- When calling decide, you must also specify verdict: remove | monetize | escalate | keep.
- Respond with a single JSON object only — no explanation, no markdown.

Examples:
  {{"operation": "query_rights_db"}}
  {{"operation": "decide", "verdict": "remove"}}

Your response:"""


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

def run_episode(difficulty: str) -> float:
    """
    Run one full episode, emitting [START] / [STEP]* / [END] to stdout.
    Returns terminal reward.
    """
    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    terminal_reward = 0.0
    success = False
    budget = 1.0

    try:
        with GenericEnvClient(base_url=ENV_BASE_URL).sync() as env:
            result = env.reset(difficulty=difficulty)
            step = 0

            while not result.done:
                step += 1
                obs = result.observation
                error: str | None = None

                prompt = obs_to_prompt(obs, budget)
                try:
                    llm_response = llm_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=64,
                        temperature=0.0,
                    )
                    action = parse_action(llm_response.choices[0].message.content)
                except Exception as exc:
                    action = {"operation": "decide", "verdict": "escalate"}
                    error = str(exc)[:120]

                result = env.step(action)
                budget = max(0.0, round(budget - 0.02, 10))

                reward = result.reward or 0.0
                done = result.done
                action_str = json.dumps(action, separators=(",", ":"))

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        terminal_reward = rewards[-1] if rewards else 0.0
        score = min(1.0, max(0.0, terminal_reward))
        success = terminal_reward > SUCCESS_THRESHOLD

    except Exception as exc:
        score = 0.0
        success = False
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

        for _ in range(EPISODES_PER_TASK):
            score = run_episode(difficulty)
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
