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
from typing import Any, NamedTuple

from reward import Dataset, score_hypothesis, RewardBreakdown

# ----------------------------------------------------------------------------
# Fireworks client (OpenAI-compatible endpoint)
# ----------------------------------------------------------------------------
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
# Model IDs confirmed against this account's live /v1/models list on 2026-07-07.
# Proposer/Critic/Refiner share deepseek-v4-pro (strongest reasoning, most critical
# roles). Verifier deliberately uses a different model family (kimi-k2p5) so a
# malformed-JSON failure in the deepseek family doesn't take down every role at once.
MODEL_PROPOSER = os.environ.get("MODEL_PROPOSER", "accounts/fireworks/models/deepseek-v4-pro")
MODEL_CRITIC   = os.environ.get("MODEL_CRITIC",   "accounts/fireworks/models/deepseek-v4-pro")
MODEL_REFINER  = os.environ.get("MODEL_REFINER",  "accounts/fireworks/models/deepseek-v4-pro")
MODEL_VERIFIER = os.environ.get("MODEL_VERIFIER", "accounts/fireworks/models/kimi-k2p6")


def _client():
    from openai import OpenAI  # imported lazily so reward.py has no hard dep
    return OpenAI(base_url=FIREWORKS_BASE_URL, api_key=os.environ["FIREWORKS_API_KEY"])


def _call_once(model: str, system: str, user: str, temperature: float) -> str:
    # DeepSeek-V4-Pro defaults to thinking mode ON, which either leaks its
    # chain-of-thought reasoning into the content field or consumes the token
    # budget before the JSON answer is emitted — both break _extract_json().
    # Disable explicitly for deepseek models only; other models 500 on this param.
    # Ref: https://api-docs.deepseek.com/guides/thinking_mode
    extra_kwargs = (
        {"extra_body": {"thinking": {"type": "disabled"}}} if "deepseek" in model else {}
    )
    resp = _client().chat.completions.create(
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
    content = _call_once(model, system, user, temperature)
    if not content.strip():
        retry_temp = min(temperature + 0.2, 1.0)
        print(f"[warn] {model} (role={role}): empty response, retrying at "
              f"temperature={retry_temp:.1f}", file=sys.stderr)
        content = _call_once(model, system, user, retry_temp)
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
  2. "predict_code": Python defining `predict(row: dict) -> value` that operationalizes
     the claim. It receives a dict of feature values (NO target) and returns the predicted
     target. You may use `math` and `statistics`. No imports, no I/O, no data lookups.

Hard rules:
  - Do NOT return a constant. A hypothesis that ignores the features is worthless.
  - Do NOT hard-code these specific rows; encode a RULE.
  - The code must faithfully implement the claim. If they disagree, you fail verification.

Return ONLY JSON: {"hypotheses": [{"claim": "...", "predict_code": "..."}, ...]}"""

PROPOSER_USER = """{schema}

Propose {k} DIVERSE, competing hypotheses about what predicts '{target}'.
Favor simple, mechanistic rules over complex ones. Make them genuinely different from each other so we can test which reflects reality."""

CRITIC_SYSTEM = """You are a skeptical peer reviewer. Your job is to find the reason a hypothesis will FAIL on held-out data before we waste a real experiment on it.

Attack it on:
  - Overfitting: does it lean on noise or on quirks of the sample rather than real structure?
  - Triviality/degeneracy: is it close to just predicting the majority class / the mean?
  - Faithfulness: does predict_code actually implement the stated claim?
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
claim plus faithful `predict(row: dict) -> value` code. No constants, no imports, no lookups.

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

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "predict_code": self.predict_code,
            "critique": self.critique,
            "verdict": self.verdict,
            "reward": self.reward.as_dict() if self.reward is not None else None,
        }


class SwarmResult(NamedTuple):
    winner: Candidate
    best: Candidate    # best raw proposal before refinement
    refined: Candidate  # refiner's output (may have lower reward than best)


def propose(data: Dataset, k: int = 4) -> list[Candidate]:
    reply = call_llm(MODEL_PROPOSER, PROPOSER_SYSTEM,
                     PROPOSER_USER.format(schema=schema_str(data), k=k, target=data.target),
                     role="proposer")
    hyps = _extract_json(reply)["hypotheses"]
    return [Candidate(h["claim"], h["predict_code"]) for h in hyps]


def _train_metric(cand: Candidate, data: Dataset) -> float:
    # cheap train-only score to give the critic a signal without touching the test set
    train_as_test = Dataset(data.task_type, data.target, train=data.train, test=data.train)
    return score_hypothesis(cand.predict_code, train_as_test).raw_metric


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
    return Candidate(j["claim"], j["predict_code"])


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


def run_swarm(data: Dataset, k: int = 4, verbose: bool = True) -> SwarmResult:
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
    return SwarmResult(winner=winner, best=best, refined=refined)
