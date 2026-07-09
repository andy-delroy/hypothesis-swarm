#!/usr/bin/env python3
"""
dashboard_gen.py — generate a self-contained offline HTML dashboard from swarm results.

Usage:
    python dashboard_gen.py                        # defaults
    python dashboard_gen.py --out /path/to.html
    python dashboard_gen.py --before results/latest_BEFORE.json \\
                            --after  results/latest_AFTER.json
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def esc(s: object) -> str:
    return html_lib.escape(str(s))


def fmt_float(v, spec: str = ".1f", suffix: str = "") -> str:
    try:
        return f"{float(v):{spec}}{suffix}"
    except (TypeError, ValueError):
        return "—"


def fmt_ts(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts


# ---------------------------------------------------------------------------
# SVG trajectory bar chart
# ---------------------------------------------------------------------------

def _stage_label(stage: str) -> str:
    return {"propose_best": "Propose best", "refine": "Refine", "winner": "Winner"}.get(
        stage, stage
    )


def generate_chart_svg(before_traj: list[dict], after_traj: list[dict] | None) -> str:
    W, H = 700, 260
    ml, mr, mt, mb = 52, 16, 28, 58
    pw = W - ml - mr   # 632
    ph = H - mt - mb   # 174

    x0 = ml
    y1 = mt + ph       # bottom of plot area

    has_after = after_traj is not None
    b_vals = [t["reward"] for t in before_traj]
    a_vals = [t["reward"] for t in after_traj] if has_after else []
    n = len(b_vals)

    group_w = pw / n
    if has_after:
        bar_w = group_w * 0.28
        gap   = group_w * 0.06
    else:
        bar_w = group_w * 0.42

    parts: list[str] = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;display:block;max-width:{W}px;margin:0 auto">'
    ]

    # Grid lines + y-axis ticks
    for tick in [0.2, 0.4, 0.6, 0.8, 1.0]:
        gy = y1 - tick * ph
        parts.append(
            f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0+pw}" y2="{gy:.1f}"'
            f' class="chart-grid" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0-5}" y="{gy+4:.1f}" text-anchor="end"'
            f' class="chart-tick">{tick:.1f}</text>'
        )

    # Baseline
    parts.append(
        f'<line x1="{x0}" y1="{y1}" x2="{x0+pw}" y2="{y1}"'
        f' class="chart-grid" stroke-width="1.5"/>'
    )

    # Bars
    for i, bv in enumerate(b_vals):
        gx     = x0 + i * group_w
        center = gx + group_w / 2

        bx = center - bar_w - gap / 2 if has_after else center - bar_w / 2
        bh = bv * ph
        by = y1 - bh

        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}"'
            f' class="bar-before" rx="2"/>'
        )
        parts.append(
            f'<text x="{bx+bar_w/2:.1f}" y="{max(by-4, mt+10):.1f}"'
            f' text-anchor="middle" class="chart-val-before">{bv:.3f}</text>'
        )

        if has_after and i < len(a_vals):
            av = a_vals[i]
            ax = center + gap / 2
            ah = av * ph
            ay = y1 - ah
            parts.append(
                f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{bar_w:.1f}" height="{ah:.1f}"'
                f' class="bar-after" rx="2"/>'
            )
            parts.append(
                f'<text x="{ax+bar_w/2:.1f}" y="{max(ay-4, mt+10):.1f}"'
                f' text-anchor="middle" class="chart-val-after">{av:.3f}</text>'
            )

        stage_lbl = _stage_label(before_traj[i]["stage"])
        parts.append(
            f'<text x="{center:.1f}" y="{y1+18:.1f}" text-anchor="middle"'
            f' class="chart-label">{esc(stage_lbl)}</text>'
        )

    # Y-axis label
    cy = mt + ph / 2
    parts.append(
        f'<text x="{x0-36}" y="{cy:.1f}" text-anchor="middle"'
        f' class="chart-axis-label" transform="rotate(-90 {x0-36:.1f} {cy:.1f})">'
        f'reward</text>'
    )

    # Legend
    lx = x0 + pw - (138 if has_after else 72)
    ly = mt + 6
    parts.append(
        f'<rect x="{lx}" y="{ly}" width="10" height="10" class="bar-before" rx="2"/>'
    )
    parts.append(
        f'<text x="{lx+14}" y="{ly+9}" class="chart-legend">BEFORE</text>'
    )
    if has_after:
        parts.append(
            f'<rect x="{lx+72}" y="{ly}" width="10" height="10" class="bar-after" rx="2"/>'
        )
        parts.append(
            f'<text x="{lx+86}" y="{ly+9}" class="chart-legend">AFTER</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML components
# ---------------------------------------------------------------------------

def render_selection_note(sel: dict) -> str:
    n       = sel.get("n_runs", 1)
    all_r   = sel.get("all_run_rewards", [])
    idx     = sel.get("chosen_run_index", 0)
    rewards = ", ".join(f"{r:.4f}" for r in all_r)
    return (
        f'<p class="selection-note">Best of {n} runs '
        f'(run {idx + 1} chosen) &middot; all: [{esc(rewards)}]</p>'
    )


def render_result_card(result: dict, label: str) -> str:
    w   = result["winner"]
    r   = w["reward"]
    sel = result.get("selection", {})
    ts  = fmt_ts(result.get("timestamp", ""))

    sel_html = render_selection_note(sel) if sel else ""
    ts_html  = f'<p class="card-ts">{esc(ts)}</p>' if ts else ""

    return f"""\
<div class="result-card">
  <div class="card-eyebrow">{esc(label)}</div>
  <div class="card-reward">{fmt_float(r.get('total'), '.4f')}</div>
  <div class="stats-row">
    <div class="stat">
      <span class="stat-label">skill</span>
      <span class="stat-value">{fmt_float(r.get('skill'), '.3f')}</span>
    </div>
    <div class="stat">
      <span class="stat-label">coverage</span>
      <span class="stat-value">{fmt_float(r.get('coverage'), '.3f')}</span>
    </div>
    <div class="stat">
      <span class="stat-label">raw metric</span>
      <span class="stat-value">{fmt_float(r.get('raw_metric'), '.3f')}</span>
    </div>
    <div class="stat">
      <span class="stat-label">format ok</span>
      <span class="stat-value">{fmt_float(r.get('format_ok'), '.0f')}</span>
    </div>
  </div>
  <div class="card-divider"></div>
  <p class="card-claim">&ldquo;{esc(w.get('claim', ''))}&rdquo;</p>
  <p class="card-verdict">{esc(w.get('verdict', ''))}</p>
  {sel_html}
  {ts_html}
</div>"""


def render_placeholder_card(label: str) -> str:
    return f"""\
<div class="result-card result-card--pending">
  <div class="card-eyebrow">{esc(label)}</div>
  <div class="card-reward card-reward--pending">&mdash;</div>
  <p class="pending-msg">Pending GRPO training</p>
  <ol class="pending-steps">
    <li>Run the swarm:<br><code>python run_demo.py --label AFTER</code></li>
    <li>Regenerate this dashboard:<br><code>python dashboard_gen.py</code></li>
  </ol>
  <p class="pending-note">This file is a static snapshot &mdash; it won&rsquo;t update automatically.</p>
</div>"""


def render_delta(before: dict, after: dict) -> str:
    b_total = before["winner"]["reward"]["total"]
    a_total = after["winner"]["reward"]["total"]
    delta   = a_total - b_total
    pct     = (delta / b_total * 100) if b_total else 0.0
    b_skill = before["winner"]["reward"]["skill"]
    a_skill = after["winner"]["reward"]["skill"]
    d_skill = a_skill - b_skill

    sign   = "+" if delta  >= 0 else ""
    s_sign = "+" if d_skill >= 0 else ""
    cls    = "delta--pos" if delta >= 0 else "delta--neg"

    return f"""\
<div class="delta-callout {cls}">
  <span class="delta-value">{sign}{delta:.4f} reward</span>
  <span class="delta-sep">&middot;</span>
  <span class="delta-skill">{s_sign}{d_skill:.3f} skill</span>
  <span class="delta-pct">({sign}{pct:.1f}%)</span>
</div>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """\
/* ── tokens ── */
:root {
  --bg:           #0d0f12;
  --surface:      #161a20;
  --surface2:     #1d2230;
  --border:       #262e3d;
  --text:         #e4eaf5;
  --muted:        #64748b;
  --accent:       #f0a500;
  --before-bar:   #4d7cfe;
  --after-bar:    #34d399;
  --pending-bg:   #0f1218;
  --pending-fg:   #2e3a4e;
  --code-bg:      #090b0e;
  --delta-pos:    #34d399;
  --delta-neg:    #f87171;
}
@media (prefers-color-scheme: light) { :root {
  --bg:           #edf0f5;
  --surface:      #ffffff;
  --surface2:     #f4f6fb;
  --border:       #cdd4e2;
  --text:         #0f1523;
  --muted:        #64748b;
  --accent:       #b56f00;
  --before-bar:   #2455d4;
  --after-bar:    #0f9963;
  --pending-bg:   #f4f6fb;
  --pending-fg:   #aab4c8;
  --code-bg:      #e5e9f2;
  --delta-pos:    #0f9963;
  --delta-neg:    #dc2626;
}}
:root[data-theme="dark"] {
  --bg:           #0d0f12;
  --surface:      #161a20;
  --surface2:     #1d2230;
  --border:       #262e3d;
  --text:         #e4eaf5;
  --muted:        #64748b;
  --accent:       #f0a500;
  --before-bar:   #4d7cfe;
  --after-bar:    #34d399;
  --pending-bg:   #0f1218;
  --pending-fg:   #2e3a4e;
  --code-bg:      #090b0e;
  --delta-pos:    #34d399;
  --delta-neg:    #f87171;
}
:root[data-theme="light"] {
  --bg:           #edf0f5;
  --surface:      #ffffff;
  --surface2:     #f4f6fb;
  --border:       #cdd4e2;
  --text:         #0f1523;
  --muted:        #64748b;
  --accent:       #b56f00;
  --before-bar:   #2455d4;
  --after-bar:    #0f9963;
  --pending-bg:   #f4f6fb;
  --pending-fg:   #aab4c8;
  --code-bg:      #e5e9f2;
  --delta-pos:    #0f9963;
  --delta-neg:    #dc2626;
}

/* ── reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.6;
  min-height: 100vh;
  padding: 2.5rem 1.5rem 5rem;
}

/* ── header ── */
.site-header {
  max-width: 1080px;
  margin: 0 auto 2.25rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 1.5rem;
}
.site-eyebrow {
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 700;
  margin-bottom: 0.4rem;
}
.site-heading {
  font-size: 1.6rem;
  font-weight: 700;
  line-height: 1.2;
  text-wrap: balance;
  margin-bottom: 1rem;
}
.dataset-row {
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
}
.dataset-item { display: flex; flex-direction: column; gap: 0.15rem; }
.dataset-label {
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
}
.dataset-value {
  font-size: 1rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

/* ── main wrapper ── */
.main {
  max-width: 1080px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* ── delta callout ── */
.delta-callout {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  padding: 0.9rem 1.25rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--delta-pos);
}
.delta-callout.delta--neg { border-left-color: var(--delta-neg); }
.delta-value {
  font-size: 1.5rem;
  font-weight: 800;
  font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace;
  font-variant-numeric: tabular-nums;
  color: var(--delta-pos);
}
.delta--neg .delta-value { color: var(--delta-neg); }
.delta-sep  { color: var(--muted); }
.delta-skill {
  font-size: 1rem;
  font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace;
  color: var(--text);
}
.delta-pct { font-size: 0.85rem; color: var(--muted); }

/* ── cards grid ── */
.cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
}
@media (max-width: 680px) { .cards { grid-template-columns: 1fr; } }

/* ── result card ── */
.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 1.75rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.result-card--pending {
  background: var(--pending-bg);
  border-color: var(--border);
  border-style: dashed;
}

.card-eyebrow {
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 700;
}
.result-card--pending .card-eyebrow { color: var(--pending-fg); }

.card-reward {
  font-size: 4rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.02em;
  font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace;
  font-variant-numeric: tabular-nums;
}
.card-reward--pending { color: var(--pending-fg); }

/* ── stats ── */
.stats-row {
  display: flex;
  gap: 1.75rem;
  flex-wrap: wrap;
}
.stat { display: flex; flex-direction: column; gap: 0.2rem; }
.stat-label {
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
}
.stat-value {
  font-size: 1.1rem;
  font-weight: 600;
  font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace;
  font-variant-numeric: tabular-nums;
}

.card-divider { height: 1px; background: var(--border); }

.card-claim {
  font-size: 1.05rem;
  font-style: italic;
  line-height: 1.55;
  text-wrap: balance;
}
.card-verdict {
  font-size: 0.875rem;
  color: var(--muted);
  line-height: 1.65;
}
.selection-note {
  font-size: 0.72rem;
  color: var(--muted);
  font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace;
  font-variant-numeric: tabular-nums;
}
.card-ts {
  font-size: 0.68rem;
  color: var(--pending-fg);
  font-family: ui-monospace, monospace;
}

/* ── pending ── */
.pending-msg {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--pending-fg);
}
.pending-sub {
  font-size: 0.875rem;
  color: var(--pending-fg);
  line-height: 1.6;
}
.pending-steps {
  list-style: decimal;
  padding-left: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  color: var(--pending-fg);
  font-size: 0.875rem;
  line-height: 1.6;
}
.pending-steps code {
  display: inline-block;
  margin-top: 0.15rem;
  background: var(--code-bg);
  padding: 0.15em 0.5em;
  font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace;
  font-size: 0.8rem;
}
.pending-note {
  font-size: 0.75rem;
  color: var(--pending-fg);
  font-style: italic;
  line-height: 1.5;
}

/* ── chart section ── */
.chart-section {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 1.75rem;
}
.chart-heading {
  font-size: 0.7rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
  margin-bottom: 1.25rem;
}

/* ── SVG chart classes (use CSS vars so both themes work) ── */
.chart-grid        { stroke: var(--border); fill: none; }
.chart-tick        { fill: var(--muted); font-size: 11px; font-family: ui-monospace, monospace; }
.chart-label       { fill: var(--muted); font-size: 12px; font-family: system-ui, sans-serif; }
.chart-axis-label  { fill: var(--muted); font-size: 11px; font-family: system-ui, sans-serif; }
.chart-val-before  { fill: var(--before-bar); font-size: 11px; font-family: ui-monospace, monospace; font-weight: 600; }
.chart-val-after   { fill: var(--after-bar);  font-size: 11px; font-family: ui-monospace, monospace; font-weight: 600; }
.chart-legend      { fill: var(--text); font-size: 12px; font-family: system-ui, sans-serif; }
.bar-before        { fill: var(--before-bar); }
.bar-after         { fill: var(--after-bar); }

/* ── theme button ── */
.theme-btn {
  position: fixed;
  top: 1rem;
  right: 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.72rem;
  letter-spacing: 0.07em;
  padding: 0.4rem 0.85rem;
  cursor: pointer;
  font-family: system-ui, sans-serif;
  text-transform: uppercase;
}
.theme-btn:hover { color: var(--text); border-color: var(--muted); }
.theme-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

JS = """\
(function () {
  var root = document.documentElement;
  var btn  = document.getElementById('theme-btn');
  function applyTheme(t) {
    root.setAttribute('data-theme', t);
    btn.textContent = t === 'dark' ? 'Light mode' : 'Dark mode';
  }
  var stored = localStorage.getItem('theme');
  if (stored) { applyTheme(stored); }
  btn.addEventListener('click', function () {
    var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    applyTheme(next);
    localStorage.setItem('theme', next);
  });
})();
"""

# ---------------------------------------------------------------------------
# Full HTML assembly
# ---------------------------------------------------------------------------

def render_html(before: dict, after: dict | None, out_path: str) -> None:
    ds      = before["dataset"]
    ds_name = ds.get("name", "dataset").replace("_", " ").title()

    # Pre-format dataset values defensively
    train_sz  = f"{ds.get('train_size', '—'):,}"  if isinstance(ds.get('train_size'), int)  else "—"
    test_sz   = f"{ds.get('test_size',  '—'):,}"  if isinstance(ds.get('test_size'),  int)  else "—"
    task      = esc(ds.get("task_type", "—"))
    mean_s    = fmt_float(ds.get("mean_strength"), ".1f", " MPa")
    rng       = (
        f"{fmt_float(ds.get('min_strength'), '.1f')} – "
        f"{fmt_float(ds.get('max_strength'), '.1f')} MPa"
    )

    header = f"""\
<header class="site-header">
  <div class="site-eyebrow">Hypothesis Swarm &middot; Scientific Discovery</div>
  <h1 class="site-heading">{esc(ds_name)}</h1>
  <div class="dataset-row">
    <div class="dataset-item">
      <span class="dataset-label">Task</span>
      <span class="dataset-value">{task}</span>
    </div>
    <div class="dataset-item">
      <span class="dataset-label">Train rows</span>
      <span class="dataset-value">{train_sz}</span>
    </div>
    <div class="dataset-item">
      <span class="dataset-label">Test rows</span>
      <span class="dataset-value">{test_sz}</span>
    </div>
    <div class="dataset-item">
      <span class="dataset-label">Mean strength</span>
      <span class="dataset-value">{mean_s}</span>
    </div>
    <div class="dataset-item">
      <span class="dataset-label">Strength range</span>
      <span class="dataset-value">{rng}</span>
    </div>
  </div>
</header>"""

    delta_html       = render_delta(before, after) if after else ""
    before_card_html = render_result_card(before, "BEFORE")
    after_card_html  = render_result_card(after, "AFTER") if after else render_placeholder_card("AFTER")

    chart_svg = generate_chart_svg(
        before["trajectory"],
        after["trajectory"] if after else None,
    )
    chart_section = f"""\
<section class="chart-section">
  <div class="chart-heading">Reward trajectory</div>
  {chart_svg}
</section>"""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hypothesis Swarm &mdash; {esc(ds_name)}</title>
<style>
{CSS}
</style>
</head>
<body>
<button class="theme-btn" id="theme-btn">Light mode</button>
{header}
<div class="main">
  {delta_html}
  <div class="cards">
    {before_card_html}
    {after_card_html}
  </div>
  {chart_section}
</div>
<script>
{JS}
</script>
</body>
</html>"""

    dest_dir = os.path.dirname(out_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Dashboard written → {os.path.abspath(out_path)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML dashboard from swarm result JSON files."
    )
    parser.add_argument("--before", default="results/latest_BEFORE.json",
                        help="Path to BEFORE result JSON (required)")
    parser.add_argument("--after", default="results/latest_AFTER.json",
                        help="Path to AFTER result JSON (optional; placeholder shown if missing)")
    parser.add_argument("--out", default="results/dashboard.html",
                        help="Output path for the HTML file")
    args = parser.parse_args()

    before = load_json(args.before)
    if before is None:
        print(f"ERROR: BEFORE file not found: {args.before}", file=sys.stderr)
        sys.exit(1)

    after = load_json(args.after)
    if after is None:
        print(f"Note: {args.after} not found — AFTER card will show pending placeholder.")

    render_html(before, after, args.out)


if __name__ == "__main__":
    main()
