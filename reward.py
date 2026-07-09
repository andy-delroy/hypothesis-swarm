"""
reward.py — The verifiable reward function for the Scientific Hypothesis Swarm.

This is the heart of the whole project and the thing that makes the pitch honest:
a hypothesis is scored by EXECUTING it against held-out data the agent never saw,
not by asking another LLM whether it "sounds right".

A hypothesis is a pair:
    - natural_language : a human-readable claim (this is what you show in the demo)
    - predict_code     : Python source defining `predict(row: dict) -> value`
                         that operationalizes the claim so it can be checked.

The reward is baseline-relative: you only earn credit for beating the trivial
predictor (majority class / mean). That keeps the "it discovered something real"
story honest — a hypothesis that just predicts the majority class scores ~0.

The same `score_hypothesis` function is used in three places:
    1. Day 1-2 frozen-swarm baseline (CPU + Fireworks).
    2. As the TRL GRPOTrainer reward function on the MI300X (see grpo_reward).
    3. As the Fireworks RFT grader if you fall back to managed training.

SECURITY NOTE: this executes model-generated code. The executor below runs in a
restricted namespace with no imports, no file/network builtins, and a wall-clock
timeout. That is *reasonable* for a hackathon on an isolated box, NOT hardened for
untrusted production input. If you ever expose this publicly, move execution into a
container / gVisor / firejail sandbox. Do not skip that step.
"""

from __future__ import annotations

import math
import re
import signal
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

# ----------------------------------------------------------------------------
# Reward weights — tune these. skill (held-out performance vs baseline) dominates.
# ----------------------------------------------------------------------------
W_FORMAT = 0.10      # did it parse + define predict() + run at all
W_COVERAGE = 0.10    # fraction of held-out rows it predicted without erroring
W_SKILL = 0.80       # normalized improvement over the trivial baseline
MIN_COVERAGE_FOR_SKILL = 0.80   # must predict >=80% of rows before skill counts
PREDICT_TIMEOUT_SEC = 5         # wall-clock budget for predicting the whole set

TaskType = Literal["classification", "regression"]


# ----------------------------------------------------------------------------
# Safe execution of model-generated predict() code
# ----------------------------------------------------------------------------
_SAFE_BUILTIN_NAMES = [
    "abs", "min", "max", "round", "len", "sum", "pow", "float", "int", "bool",
    "str", "sorted", "range", "enumerate", "zip", "map", "filter", "all", "any",
    "list", "dict", "tuple", "set", "isinstance",
]


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


def _build_safe_globals() -> dict:
    import builtins as _b
    safe_builtins = {n: getattr(_b, n) for n in _SAFE_BUILTIN_NAMES}
    # No __import__, open, eval, exec, compile, getattr, setattr, globals, etc.
    g: dict[str, Any] = {"__builtins__": safe_builtins}
    # Curated safe math surface, injected by name (import statements will fail,
    # because __import__ is absent from __builtins__ above).
    g["math"] = math
    g["statistics"] = statistics
    return g


def _extract_code(raw: str) -> str:
    """Strip ```python ... ``` fences if the model wrapped the code."""
    if "```" in raw:
        m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        if m:
            return m.group(1).strip()
    return raw.strip()


def compile_predict(predict_code: str) -> Callable[[dict], Any] | None:
    """Compile predict_code into a callable, or return None if it's not usable."""
    code = _extract_code(predict_code)
    g = _build_safe_globals()
    try:
        exec(code, g)  # noqa: S102 — restricted namespace, see SECURITY NOTE
    except Exception:
        return None
    fn = g.get("predict")
    return fn if callable(fn) else None


def _run_predictions(fn: Callable[[dict], Any], rows: list[dict]) -> list[Any]:
    """Run fn over rows with a single wall-clock timeout. None marks a failed row."""
    preds: list[Any] = []
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, PREDICT_TIMEOUT_SEC)
    try:
        for row in rows:
            try:
                preds.append(fn(dict(row)))
            except _Timeout:
                raise
            except Exception:
                preds.append(None)
    except _Timeout:
        # ran out of time — pad the rest as failures
        preds.extend([None] * (len(rows) - len(preds)))
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return preds


# ----------------------------------------------------------------------------
# Metrics + baselines
# ----------------------------------------------------------------------------
def _accuracy(preds, truth) -> float:
    ok = [1.0 for p, t in zip(preds, truth) if p is not None and p == t]
    return sum(ok) / len(truth) if truth else 0.0


def _majority_baseline_acc(train_y, test_y) -> float:
    if not train_y:
        return 0.0
    majority = max(set(train_y), key=train_y.count)
    return sum(1 for t in test_y if t == majority) / len(test_y)


def _regression_skill(preds, truth, train_y) -> float:
    """Skill score vs the mean-predictor baseline: 1 - SSE_model / SSE_baseline.
    Clipped to [0, 1]. 0 means 'no better than predicting the training mean'."""
    paired = [(float(p), t) for p, t in zip(preds, truth) if p is not None]
    if not paired or not train_y:
        return 0.0
    mean_y = statistics.fmean(train_y)
    sse_model = sum((p - t) ** 2 for p, t in paired)
    sse_base = sum((mean_y - t) ** 2 for _, t in paired)
    if sse_base <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1.0 - sse_model / sse_base))


def _is_degenerate(preds) -> bool:
    """A hypothesis that outputs one constant value learned nothing structural."""
    seen = {p for p in preds if p is not None}
    return len(seen) <= 1


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
@dataclass
class RewardBreakdown:
    total: float
    format_ok: float
    coverage: float
    skill: float
    raw_metric: float          # accuracy or regression-skill on held-out
    baseline_metric: float     # trivial-predictor score on held-out
    degenerate: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "format_ok": self.format_ok,
            "coverage": round(self.coverage, 3),
            "skill": round(self.skill, 3),
            "raw_metric": round(self.raw_metric, 3),
            "baseline_metric": round(self.baseline_metric, 3),
            "degenerate": self.degenerate,
            "note": self.note,
        }


@dataclass
class Dataset:
    task_type: TaskType
    target: str
    train: list[dict] = field(default_factory=list)   # agent may see a SAMPLE of this
    test: list[dict] = field(default_factory=list)     # held-out; agent NEVER sees this

    @property
    def train_y(self) -> list:
        return [r[self.target] for r in self.train]

    @property
    def test_y(self) -> list:
        return [r[self.target] for r in self.test]

    def test_features(self) -> list[dict]:
        # rows WITHOUT the target, so predict() can't cheat by reading the answer
        return [{k: v for k, v in r.items() if k != self.target} for r in self.test]


def score_hypothesis(predict_code: str, data: Dataset) -> RewardBreakdown:
    """Execute a hypothesis against held-out data and return a graded reward.

    This is the authoritative verifier. The LLM 'Verifier' agent never sets this
    number — reality does.
    """
    fn = compile_predict(predict_code)
    if fn is None:
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False,
                               note="did not compile / no predict()")

    preds = _run_predictions(fn, data.test_features())
    truth = data.test_y
    covered = sum(1 for p in preds if p is not None)
    coverage = covered / len(truth) if truth else 0.0
    format_ok = 1.0 if covered > 0 else 0.0

    degenerate = _is_degenerate(preds)

    if data.task_type == "classification":
        raw = _accuracy(preds, truth)
        base = _majority_baseline_acc(data.train_y, truth)
        # normalized improvement over baseline, in [0, 1]
        skill = max(0.0, (raw - base) / (1.0 - base + 1e-9)) if raw >= base else 0.0
    else:
        raw = _regression_skill(preds, truth, data.train_y)   # already vs mean baseline
        base = 0.0
        skill = raw

    if coverage < MIN_COVERAGE_FOR_SKILL or degenerate:
        skill = 0.0

    total = W_FORMAT * format_ok + W_COVERAGE * coverage + W_SKILL * skill
    note = ""
    if degenerate:
        note = "constant predictor — no credit"
    elif coverage < MIN_COVERAGE_FOR_SKILL:
        note = f"coverage {coverage:.0%} below {MIN_COVERAGE_FOR_SKILL:.0%} — no skill credit"

    return RewardBreakdown(total, format_ok, coverage, skill, raw, base, degenerate, note)


# ----------------------------------------------------------------------------
# Adapters for the two RL backends you'll plug into later
# ----------------------------------------------------------------------------
def grpo_reward(completions, dataset: Dataset, extract_code: Callable[[Any], str], **kwargs):
    """TRL GRPOTrainer-compatible reward: maps a batch of completions -> [floats].

    `extract_code` pulls predict_code out of one completion (depends on how you
    format the policy's output). Kept injectable so you can change the output
    contract without touching the scorer.
    """
    out = []
    for c in completions:
        try:
            code = extract_code(c)
            out.append(score_hypothesis(code, dataset).total)
        except Exception:
            out.append(0.0)
    return out


def fireworks_grader(predict_code: str, data: Dataset) -> float:
    """Shape expected by a Fireworks RFT evaluator: return a single scalar reward."""
    return score_hypothesis(predict_code, data).total
