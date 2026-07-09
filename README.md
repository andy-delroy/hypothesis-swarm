# Hypothesis Swarm — day 1–2 scaffold

A four-agent scientific hypothesis swarm whose reward comes from **executing hypotheses
against held-out data**, not from an LLM judge. This is the "graded by reality, not by
vibes" core, ready to build on the moment you start.

## Files
- `reward.py` — the verifiable reward. Compiles model-generated `predict(row)` code in a
  restricted namespace, runs it on held-out rows, scores it *relative to the trivial
  baseline* (majority class / mean). This is the authoritative grader — used for the
  baseline demo, as the TRL GRPO reward, and as the Fireworks RFT grader.
- `agents.py` — the four roles (Proposer, Critic, Refiner, Verifier), a Fireworks client,
  and the propose→critique→refine→verify loop. The Verifier **narrates** the ground-truth
  number; it never sets it.
- `run_demo.py` — end-to-end harness. Offline mock swarm with no API key; real swarm when
  `FIREWORKS_API_KEY` is set. Ships with a synthetic planted-rule dataset so you can prove
  "discovery" works before touching real data.
- `test_reward.py` — unit check that the reward rewards the true rule, gives partial credit
  to near-misses, and zeroes out constant-predictor cheese and blocked imports.

## Run it now (offline, free)
```bash
pip install numpy
python test_reward.py     # confirm reward behaves
python run_demo.py        # confirm the full swarm graph works (mock LLM)
```

## Go live on Fireworks
```bash
pip install openai
export FIREWORKS_API_KEY=...        # verify model IDs at https://fireworks.ai/models
python run_demo.py                  # real four-agent swarm; records your BEFORE number
```

## Day 1–2 checklist (CPU + Fireworks, no GPU)
1. Swap `make_synthetic()` for a real loader (a Matbench-style materials task or a
   drug-interaction set). Keep the train/test split strict — the agent sees only a
   sample of train.
2. Run the frozen swarm, record `winner.reward.total` as your **BEFORE** baseline.
3. Build the leaderboard / before-after UI on top of `run_swarm` output. With frozen
   models you already have a demoable product — this is your insurance if GPU credit slips.

## Handoff to GRPO (MI300X) or Fireworks RFT
- The thing you train is the **Proposer**. Its output contract is the JSON with
  `predict_code`. `reward.grpo_reward(completions, dataset, extract_code=...)` is your
  TRL `reward_funcs` entry; `reward.fireworks_grader(predict_code, data)` is the RFT
  evaluator. Same scorer, both backends.
- Keep Critic/Refiner/Verifier frozen (they shape the trajectory) or fold them into the
  policy via self-play later.
- The **AFTER** number is the same swarm, same held-out set, post-training. That before/after
  jump, plus the plain-English discovered hypothesis, is the 3-minute demo.

## Before you rely on it
- Model IDs in `agents.py` are illustrative — confirm current Fireworks strings.
- `reward.py` executes model-generated code in a restricted namespace with a timeout.
  Fine for an isolated hackathon box; sandbox it (container/gVisor/firejail) before any
  public exposure.
