"""
reward.py — The verifiable reward function for the Scientific Hypothesis Swarm.

This is the heart of the whole project and the thing that makes the pitch honest:
a hypothesis is scored by EXECUTING it against held-out data the agent never saw,
not by asking another LLM whether it "sounds right".

A hypothesis is a pair:
    - natural_language : a human-readable claim (this is what you show in the demo)
    - predict_code     : Python source defining `features(row: dict) -> dict[str, float]`
                         that operationalizes the claim as one or more NAMED engineered
                         features. The LLM proposes the functional form (which transforms
                         matter); it does NOT guess numeric coefficients — those are fit
                         by ordinary least squares against real training data below. LLMs
                         are reasonably good at picking a plausible transform from a small
                         sample of rows in a prompt; they are not reliable at hand-deriving
                         multi-variable regression weights from that same sample.

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

import concurrent.futures
import math
import re
import signal
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

# ----------------------------------------------------------------------------
# Reward weights — tune these. skill (held-out performance vs baseline) dominates.
# ----------------------------------------------------------------------------
# format_ok/coverage are gates, not additive credit (see score_hypothesis): a
# hypothesis must pass both, and not be degenerate, before it earns ANY reward —
# but passing them earns nothing on its own. Only skill above baseline pays out.
# W_SKILL=1.0 so total == skill exactly when the gate passes — the old 0.10/0.10
# format/coverage weights that used to make the three sum to 1.0 are gone, so
# nothing should hold skill's own weight below the full 0-1 scale anymore.
W_SKILL = 1.0        # normalized improvement over the trivial baseline
MIN_COVERAGE_FOR_SKILL = 0.80   # must predict >=80% of held-out rows before skill counts
PREDICT_TIMEOUT_SEC = 5         # wall-clock budget for one features()-over-all-rows pass

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


_SELF_IMPORT_RE = re.compile(
    r'^[ \t]*import\s+(math|statistics)[ \t]*(#.*)?$', re.MULTILINE
)


def _strip_self_imports(code: str) -> str:
    """Remove bare `import math` / `import statistics` lines, at ANY
    indentation (including inside a function body) and with an optional
    trailing comment.

    math/statistics are already provided as globals inside the sandbox (see
    _build_safe_globals) — __import__ is deliberately absent from the
    sandbox's __builtins__, so any self-import of either name raises. Because
    `import math` inside a function body only executes when that function is
    CALLED (not when the module is exec'd to define it), a self-import at the
    top of features()/predict() raises on every single row, not at compile
    time — score_hypothesis then just sees an empty/near-empty design matrix
    or all-None predictions and reports a plain 0.0, with nothing that looks
    like an exception. Only these two names are stripped; any other import
    (numpy, os, ...) is intentionally left to keep failing — see SECURITY NOTE.
    """
    return _SELF_IMPORT_RE.sub('', code)


def compile_features(predict_code: str) -> Callable[[dict], dict] | None:
    """Compile predict_code into a `features(row) -> dict[str, float]` callable,
    or return None if it's not usable.

    The field/parameter is still named predict_code (JSON contract + Candidate
    dataclass field name are unchanged); the function IT DEFINES is `features`,
    not `predict` — see the module docstring for why.
    """
    code = _extract_code(predict_code)
    code = _strip_self_imports(code)
    g = _build_safe_globals()
    try:
        exec(code, g)  # noqa: S102 — restricted namespace, see SECURITY NOTE
    except Exception:
        return None
    fn = g.get("features")
    return fn if callable(fn) else None


def _predict_loop(fn: Callable[[dict], Any], rows: list[dict], preds: list[Any]) -> None:
    """Append one prediction per row into preds, in place. None marks a failed row."""
    for row in rows:
        try:
            preds.append(fn(dict(row)))
        except _Timeout:
            raise
        except Exception:
            preds.append(None)


def _run_predictions(fn: Callable[[dict], Any], rows: list[dict]) -> list[Any]:
    """Run fn over rows with a single wall-clock timeout. None marks a failed row."""
    preds: list[Any] = []
    # signal.alarm()/SIGALRM only works in the main thread of the main
    # interpreter — it raises ValueError otherwise (e.g. inside Ray worker
    # threads). Fall back to a ThreadPoolExecutor-based timeout there, same
    # as the existing non-POSIX (no SIGALRM) fallback below.
    use_alarm = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if use_alarm:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, PREDICT_TIMEOUT_SEC)
        try:
            _predict_loop(fn, rows, preds)
        except _Timeout:
            # ran out of time — pad the rest as failures
            preds.extend([None] * (len(rows) - len(preds)))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    else:
        # Unlike the signal path (which aborts _predict_loop immediately via an
        # injected exception), Python cannot force-stop a running thread: on
        # timeout below, _predict_loop keeps calling fn() for the remaining
        # rows to its own natural completion in the background, even though
        # we've already stopped waiting and moved on.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_predict_loop, fn, rows, preds)
        try:
            future.result(timeout=PREDICT_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            # Ran out of time. The worker thread can't be killed and may still
            # be appending to `preds` in the background, so pad off a snapshot
            # rather than mutating the shared list — avoids a race that could
            # leave preds longer than rows.
            snapshot = list(preds)
            snapshot.extend([None] * (len(rows) - len(snapshot)))
            preds = snapshot
        finally:
            executor.shutdown(wait=False)
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
        # rows WITHOUT the target, so features() can't cheat by reading the answer
        return [{k: v for k, v in r.items() if k != self.target} for r in self.test]

    def train_features(self) -> list[dict]:
        # same, over train — used to build the design matrix for the OLS fit
        return [{k: v for k, v in r.items() if k != self.target} for r in self.train]


# ----------------------------------------------------------------------------
# OLS fitting: turn named features into a linear model via least squares
# ----------------------------------------------------------------------------
def _fit_ols(train_feature_rows: list[Any], train_y: list[float]
             ) -> tuple[list[str], np.ndarray] | None:
    """Fit intercept + linear coefficients via ordinary least squares, mapping
    named features (from features()) to the real training targets.

    Returns (feature_key_order, coef_vector) — coef_vector[0] is the intercept,
    followed by one coefficient per key in feature_key_order — or None if no
    usable fit exists: too few valid rows, inconsistent feature keys across
    rows, non-numeric feature/target values, or NaN/inf anywhere in the design
    matrix, targets, or fitted coefficients.
    """
    paired = [(f, y) for f, y in zip(train_feature_rows, train_y)
              if isinstance(f, dict) and f]
    if len(paired) < 2:
        return None

    keys = sorted(paired[0][0].keys())
    try:
        X_rows = []
        y_rows = []
        for f, y in paired:
            if sorted(f.keys()) != keys:
                return None   # inconsistent feature keys across rows
            X_rows.append([float(f[k]) for k in keys])
            y_rows.append(float(y))
        X = np.asarray(X_rows, dtype=float)
        y_arr = np.asarray(y_rows, dtype=float)
    except (TypeError, ValueError):
        return None            # non-numeric feature or target value

    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y_arr))):
        return None            # NaN/inf in the design matrix or targets

    X_design = np.column_stack([np.ones(len(X)), X])
    try:
        coefs, *_ = np.linalg.lstsq(X_design, y_arr, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(coefs)):
        return None            # NaN/inf in the fit result

    return keys, coefs


def _apply_fit(feature_rows: list[Any], keys: list[str], coefs: np.ndarray) -> list[Any]:
    """Apply a fitted intercept+coefficients to each row's feature dict.

    None marks a row that can't be scored: features() failed/timed out on it
    (feature_rows entry is None), it's missing one of `keys`, a value isn't
    numeric, or the resulting prediction isn't finite.
    """
    preds: list[Any] = []
    for f in feature_rows:
        if not isinstance(f, dict):
            preds.append(None)
            continue
        try:
            values = [1.0] + [float(f[k]) for k in keys]
        except (KeyError, TypeError, ValueError):
            preds.append(None)
            continue
        pred = float(np.dot(coefs, values))
        preds.append(pred if math.isfinite(pred) else None)
    return preds


def score_hypothesis(predict_code: str, data: Dataset) -> RewardBreakdown:
    """Execute a hypothesis's features() against held-out data and return a
    graded reward.

    The LLM proposes which named features to compute; ordinary least squares
    (see _fit_ols) fits the actual coefficients against data.train, and the
    fitted linear model is applied to data.test to score skill. This is the
    authoritative verifier. The LLM 'Verifier' agent never sets this number —
    reality does.
    """
    fn = compile_features(predict_code)
    if fn is None:
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False,
                               note="did not compile / no features()")

    train_feats = _run_predictions(fn, data.train_features())
    fit = _fit_ols(train_feats, data.train_y)
    if fit is None:
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False,
                               note="feature fit failed (too few usable rows, "
                                    "inconsistent/non-numeric feature keys, or "
                                    "NaN/inf in the design matrix or fit result)")
    keys, coefs = fit

    test_feats = _run_predictions(fn, data.test_features())
    preds = _apply_fit(test_feats, keys, coefs)
    truth = data.test_y
    covered = sum(1 for p in preds if p is not None)
    coverage = covered / len(truth) if truth else 0.0
    format_ok = 1.0 if covered > 0 else 0.0

    degenerate = _is_degenerate(preds)

    if data.task_type == "classification":
        # OLS gives a continuous score; threshold at 0.5 for the class decision
        # (a linear probability model) rather than adding a second fitting method.
        class_preds = [None if p is None else (1 if p >= 0.5 else 0) for p in preds]
        raw = _accuracy(class_preds, truth)
        base = _majority_baseline_acc(data.train_y, truth)
        # normalized improvement over baseline, in [0, 1]
        skill = max(0.0, (raw - base) / (1.0 - base + 1e-9)) if raw >= base else 0.0
    else:
        raw = _regression_skill(preds, truth, data.train_y)   # already vs mean baseline
        base = 0.0
        skill = raw

    if coverage < MIN_COVERAGE_FOR_SKILL or degenerate:
        skill = 0.0

    # format_ok/coverage are gates, not additive credit (see W_SKILL comment
    # above): reward is earned only for skill above baseline, and only once the
    # gate is passed — no partial credit just for compiling and running.
    gate_ok = format_ok == 1.0 and coverage >= MIN_COVERAGE_FOR_SKILL and not degenerate
    total = W_SKILL * skill if gate_ok else 0.0

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
