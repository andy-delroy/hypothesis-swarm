"""
agents.py — The four-role hypothesis swarm and its orchestration.

Roles:
    PROPOSER — invents candidate hypotheses (natural-language claim + predict code)
    CRITIC   — attacks a hypothesis for over-fitting / triviality / code-claim mismatch
    REFINER  — rewrites the hypothesis to address the critique
    VERIFIER — checks the code faithfully operationalizes the claim, and NARRATES the
               ground-truth result. Crucially, the VERIFIER does NOT decide the score.
               reward.score_hypothesis() does, by executing code on held-out data.
               This is the whole "graded by reality, not by an AI judge" thesis — keep
               the authority in reward.py, not here.

Days 1-2 (this file): all four roles are separate calls to FROZEN Fireworks-hosted
models. No GPU needed. You get a working product demo + a baseline number.
Later: the PROPOSER becomes the policy you GRPO-train on the MI300X; reward.py is the
grader. The other three roles can stay frozen (they shape the trajectory) or also be
folded into the trained policy via self-play.

Set FIREWORKS_API_KEY in your env. Model IDs change — verify current ones at
https://fireworks.ai/models (the defaults below are illustrative).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from reward import Dataset, score_hypothesis, RewardBreakdown

# ----------------------------------------------------------------------------
# Fireworks client (OpenAI-compatible endpoint)
# ----------------------------------------------------------------------------
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
# Model IDs confirmed against this account's live /v1/models list on 2026-07-07.
# Proposer/Critic/Refiner share deepseek-v4-pro (strongest reasoning, most critical
# roles). Verifier deliberately uses a different model family (kimi-k2p6) so a
# malformed-JSON failure in the deepseek family doesn't take down every role at once.
MODEL_PROPOSER = os.environ.get("MODEL_PROPOSER", "accounts/fireworks/models/deepseek-v4-pro")
MODEL_CRITIC   = os.environ.get("MODEL_CRITIC",   "accounts/fireworks/models/deepseek-v4-pro")
MODEL_REFINER  = os.environ.get("MODEL_REFINER",  "accounts/fireworks/models/deepseek-v4-pro")
MODEL_VERIFIER = os.environ.get("MODEL_VERIFIER", "accounts/fireworks/models/kimi-k2p6")

# Per-role endpoint + credential overrides.
# All default to the shared Fireworks URL/key, so existing behaviour is fully
# unchanged unless you explicitly set these env vars.
#
# Hybrid setup (locally-trained Proposer + frozen Critic/Refiner/Verifier):
#   After GRPO training, merge and serve the checkpoint (see serve_trained_proposer.sh),
#   then set these three vars in your run_demo.py shell before running --label AFTER:
#
#       export PROPOSER_BASE_URL=http://localhost:8001/v1
#       export PROPOSER_API_KEY=local     # vLLM ignores the key unless --api-key was
#                                         # explicitly passed to `vllm serve`
#       export MODEL_PROPOSER=concrete-swarm-proposer   # must match --served-model-name
#
#   Critic, Refiner, and Verifier are unaffected — their *_BASE_URL vars remain
#   unset, so they continue routing through Fireworks as before.
_FW_KEY = os.environ.get("FIREWORKS_API_KEY", "")   # shared fallback for all roles

BASE_URL_PROPOSER = os.environ.get("PROPOSER_BASE_URL", FIREWORKS_BASE_URL)
BASE_URL_CRITIC   = os.environ.get("CRITIC_BASE_URL",   FIREWORKS_BASE_URL)
BASE_URL_REFINER  = os.environ.get("REFINER_BASE_URL",  FIREWORKS_BASE_URL)
BASE_URL_VERIFIER = os.environ.get("VERIFIER_BASE_URL", FIREWORKS_BASE_URL)

API_KEY_PROPOSER = os.environ.get("PROPOSER_API_KEY", _FW_KEY)
API_KEY_CRITIC   = os.environ.get("CRITIC_API_KEY",   _FW_KEY)
API_KEY_REFINER  = os.environ.get("REFINER_API_KEY",  _FW_KEY)
API_KEY_VERIFIER = os.environ.get("VERIFIER_API_KEY", _FW_KEY)

_ROLE_BASE_URLS: dict[str, str] = {
    "proposer": BASE_URL_PROPOSER,
    "critic":   BASE_URL_CRITIC,
    "refiner":  BASE_URL_REFINER,
    "verifier": BASE_URL_VERIFIER,
}
_ROLE_API_KEYS: dict[str, str] = {
    "proposer": API_KEY_PROPOSER,
    "critic":   API_KEY_CRITIC,
    "refiner":  API_KEY_REFINER,
    "verifier": API_KEY_VERIFIER,
}


def _client(role: str):
    from openai import OpenAI  # imported lazily so reward.py has no hard dep
    return OpenAI(
        base_url=_ROLE_BASE_URLS.get(role, FIREWORKS_BASE_URL),
        # Fall back to "local" if no key is configured — vLLM's OpenAI-compatible
        # server accepts any non-empty string; Fireworks will 401 on a bad key,
        # which is the right failure mode rather than an OpenAI SDK init error.
        api_key=_ROLE_API_KEYS.get(role, _FW_KEY) or "local",
    )


def _call_once(model: str, system: str, user: str, temperature: float,
               role: str = "unknown") -> str:
    # DeepSeek-V4-Pro defaults to thinking mode ON, which either leaks its
    # chain-of-thought reasoning into the content field or consumes the token
    # budget before the JSON answer is emitted — both break _extract_json().
    # Disable explicitly for deepseek models only; other models 500 on this param.
    # Ref: https://api-docs.deepseek.com/guides/thinking_mode
    extra_kwargs = (
        {"extra_body": {"thinking": {"type": "disabled"}}} if "deepseek" in model else {}
    )
    resp = _client(role).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=2000,
        **extra_kwargs,
    )
    return resp.choices[0].message.content or ""


def call_llm(model: str, system: str, user: str, temperature: float = 0.7,
             role: str = "unknown") -> str:
    content = _call_once(model, system, user, temperature, role)
    if not content.strip():
        retry_temp = min(temperature + 0.2, 1.0)
        print(f"[warn] {model} (role={role}): empty response, retrying at "
              f"temperature={retry_temp:.1f}", file=sys.stderr)
        content = _call_once(model, system, user, retry_temp, role)
    if not content.strip():
        raise RuntimeError(
            f"Model {model!r} (role={role!r}) returned empty content on both the "
            "original call and one retry. Check model availability and whether the "
            "prompt is triggering a safety refusal."
        )
    return content


def _extract_json(raw: str) -> dict:
    """Best-effort JSON extraction from a model reply.

    Try a direct parse first so that well-formed JSON whose string values happen
    to contain backticks (e.g. predict_code wrapped in ```python fences) isn't
    mis-detected as an outer markdown fence. Fall back to fence-stripping only
    when the raw string isn't valid JSON as-is.
    """
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if m:
            raw = m.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


# ----------------------------------------------------------------------------
# Shared context you feed every agent
# ----------------------------------------------------------------------------
def schema_str(data: Dataset, n_sample: int = 12) -> str:
    """Feature schema + a SMALL sample of TRAIN rows. Never leak the test set."""
    feats = [k for k in data.train[0].keys() if k != data.target]
    lines = [f"Task type: {data.task_type}",
             f"Target column to predict: '{data.target}'",
             f"Feature columns: {feats}",
             f"Sample of {min(n_sample, len(data.train))} training rows:"]
    for r in data.train[:n_sample]:
        lines.append("  " + json.dumps(r))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# ROLE PROMPTS
# ----------------------------------------------------------------------------
PROPOSER_SYSTEM = """You are a scientist proposing testable, general hypotheses about a dataset.

A hypothesis is a claim about the UNDERLYING STRUCTURE relating the features to the target — not a description of these particular rows. It must generalize to data you have never seen.

For each hypothesis you output BOTH:
  1. "claim": one plain-English sentence a domain expert could read and judge.
  2. "predict_code": Python defining `features(row: dict) -> dict[str, float]` that
     operationalizes the claim as one or more NAMED engineered features (e.g.
     {"water_binder_ratio": ..., "log_age": ...}). It receives a dict of feature values
     (NO target) and returns a dict mapping feature names to numeric values. Do NOT
     include literal numeric coefficients or weights — a real least-squares fit against
     training data determines those automatically. Your job is choosing WHICH transforms
     and interactions matter, a real scientific judgment call, not guessing their weights.
     You may use `math` and `statistics`. No imports, no I/O, no data lookups.

Hard rules:
  - Do NOT return a constant feature that ignores the row's inputs — it carries no signal.
  - Do NOT hard-code these specific rows; encode a generalizable transform.
  - The code must faithfully implement the claim. If they disagree, you fail verification.

Return ONLY JSON: {"hypotheses": [{"claim": "...", "predict_code": "..."}, ...]}"""

PROPOSER_USER = """{schema}

Propose {k} DIVERSE, competing hypotheses about what predicts '{target}'.
Favor simple, mechanistic rules over complex ones. Make them genuinely different from each other so we can test which reflects reality."""

CRITIC_SYSTEM = """You are a skeptical peer reviewer. Your job is to find the reason a hypothesis will FAIL on held-out data before we waste a real experiment on it.

Attack it on:
  - Overfitting: does it lean on noise or on quirks of the sample rather than real structure?
  - Triviality/degeneracy: is the feature transform close to a trivial pass-through of one
    raw column (no real transform, easily collapses to the majority class / the mean), or
    does it capture real structure?
  - Faithfulness: does the features() code actually implement the stated claim's transform?
  - Missed structure: is there an obvious feature or interaction it ignores?

Be specific and concrete — name the weakness and the exact change that would test or fix it.
Do NOT rewrite the hypothesis yourself. One tight paragraph.
Return ONLY JSON: {"critique": "...", "most_likely_failure": "..."}"""

CRITIC_USER = """{schema}

Hypothesis under review:
  claim: {claim}
  predict_code:
{predict_code}

Its measured accuracy ON TRAINING DATA was {train_metric}. (Held-out is hidden from you.)
Critique it."""

REFINER_SYSTEM = """You are a scientist revising a hypothesis in response to peer critique.

Produce an IMPROVED hypothesis that directly addresses the critique's most_likely_failure,
while staying simple and general. Same output contract as the proposer: a plain-English
claim plus faithful `features(row: dict) -> dict[str, float]` code defining named engineered
features (no literal coefficients — those are fit automatically). No constants, no imports,
no lookups.

Return ONLY JSON: {"claim": "...", "predict_code": "...", "what_changed": "..."}"""

REFINER_USER = """{schema}

Original hypothesis:
  claim: {claim}
  predict_code:
{predict_code}

Peer critique:
  {critique}
  Most likely failure: {most_likely_failure}

Return the revised hypothesis."""

VERIFIER_SYSTEM = """You are the verification officer. You are given a hypothesis AND the
GROUND-TRUTH result of running its code against held-out data (computed by an executor,
not by you — you cannot change it).

Your only jobs:
  1. Confirm the predict_code faithfully implements the stated claim (flag if it doesn't).
  2. Write a two-sentence plain-English verdict for a human scientist: what the hypothesis
     asserts, and how well it actually predicted unseen data.

Never inflate or override the measured numbers. Report reality.
Return ONLY JSON: {"faithful": true/false, "verdict": "..."}"""

VERIFIER_USER = """Hypothesis:
  claim: {claim}
  predict_code:
{predict_code}

GROUND-TRUTH held-out result (authoritative, from code execution):
  held-out score: {raw_metric}
  trivial-baseline score: {baseline_metric}
  normalized skill (0-1): {skill}
  reward: {total}
  note: {note}

Write the verdict."""


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
@dataclass
class Candidate:
    claim: str
    predict_code: str
    reward: RewardBreakdown | None = None
    critique: str = ""
    verdict: str = ""
    what_changed: str = ""
    debate: dict | None = None

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "predict_code": self.predict_code,
            "critique": self.critique,
            "verdict": self.verdict,
            "reward": self.reward.as_dict() if self.reward is not None else None,
        }


def propose(data: Dataset, k: int = 4) -> list[Candidate]:
    reply = call_llm(MODEL_PROPOSER, PROPOSER_SYSTEM,
                     PROPOSER_USER.format(schema=schema_str(data), k=k, target=data.target),
                     role="proposer")
    hyps = _extract_json(reply)["hypotheses"]
    return [Candidate(h["claim"], h["predict_code"]) for h in hyps]


def _train_metric(cand: Candidate, data: Dataset) -> float:
    # Cheap signal for the critic, without ever touching the real held-out test
    # set. Split train itself into a fit slice + a held-out-within-train
    # validation slice, fit on the former, score against the latter — in-sample
    # scoring (fit and score on the same rows) would be structurally
    # near-guaranteed to look good regardless of hypothesis quality, since OLS
    # minimizes error on its own fit data by construction. That would make the
    # critic's overfitting-detection signal meaningless.
    #
    # Fixed slice (not shuffled here) — data.train's own order is already a
    # one-time deterministic shuffle from load_concrete(seed=...), so this is
    # stable across every call against the same Dataset object within a run.
    n = len(data.train)
    split = int(n * 0.8)
    fit_slice = data.train[:split]
    val_slice = data.train[split:]
    if len(val_slice) < 2:   # too few rows to split meaningfully — fall back to in-sample
        fit_slice = data.train
        val_slice = data.train
    internal_split = Dataset(data.task_type, data.target, train=fit_slice, test=val_slice)
    return score_hypothesis(cand.predict_code, internal_split).raw_metric


def critique(cand: Candidate, data: Dataset) -> tuple[str, str]:
    reply = call_llm(MODEL_CRITIC, CRITIC_SYSTEM, CRITIC_USER.format(
        schema=schema_str(data), claim=cand.claim, predict_code=cand.predict_code,
        train_metric=round(_train_metric(cand, data), 3)), role="critic")
    j = _extract_json(reply)
    return j["critique"], j.get("most_likely_failure", "")


def refine(cand: Candidate, critique_text: str, failure: str, data: Dataset) -> Candidate:
    reply = call_llm(MODEL_REFINER, REFINER_SYSTEM, REFINER_USER.format(
        schema=schema_str(data), claim=cand.claim, predict_code=cand.predict_code,
        critique=critique_text, most_likely_failure=failure), role="refiner")
    j = _extract_json(reply)
    cand = Candidate(j["claim"], j["predict_code"])
    cand.what_changed = j.get("what_changed", "")
    return cand


def verify(cand: Candidate, data: Dataset) -> str:
    """Score with reality FIRST, then let the LLM narrate — never the other way around."""
    cand.reward = score_hypothesis(cand.predict_code, data)   # authoritative
    reply = call_llm(MODEL_VERIFIER, VERIFIER_SYSTEM, VERIFIER_USER.format(
        claim=cand.claim, predict_code=cand.predict_code, **cand.reward.as_dict()),
        temperature=0.2, role="verifier")
    try:
        return _extract_json(reply).get("verdict", "")
    except json.JSONDecodeError:
        print(f"[warn] verifier: JSON parse failed; raw reply:\n{reply}", file=sys.stderr)
        return "(verifier response could not be parsed — see stderr)"


def run_swarm(data: Dataset, k: int = 4, verbose: bool = True) -> Candidate:
    """One full propose -> critique -> refine -> verify cycle. Returns the winner.

    This is what you run for the frozen-baseline demo. The winner's reward.total is
    your 'before' number; after GRPO-training the proposer, rerun and compare.
    """
    cands = propose(data, k=k)
    # rank raw proposals by held-out reward, take the best to refine
    for c in cands:
        c.reward = score_hypothesis(c.predict_code, data)
    best = max(cands, key=lambda c: c.reward.total)
    if verbose:
        print(f"[propose] {len(cands)} hypotheses; best raw reward={best.reward.total:.2f}")

    crit, fail = critique(best, data)
    best.critique = crit
    refined = refine(best, crit, fail, data)
    refined.reward = score_hypothesis(refined.predict_code, data)
    if verbose:
        print(f"[refine ] reward {best.reward.total:.2f} -> {refined.reward.total:.2f}")

    winner = max([best, refined], key=lambda c: c.reward.total)
    winner.verdict = verify(winner, data)
    if verbose:
        print(f"[verify ] {winner.verdict}")
    winner.debate = {
        "original_proposal": {
            "claim": best.claim,
            "predict_code": best.predict_code,
            "reward": best.reward.total,
        },
        "critique": crit,
        "refinement": {
            "claim": refined.claim,
            "predict_code": refined.predict_code,
            "what_changed": refined.what_changed,
            "reward": refined.reward.total,
        },
        "final_verdict": winner.verdict,
    }
    return winner
