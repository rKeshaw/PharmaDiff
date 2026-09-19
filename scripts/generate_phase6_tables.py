#!/usr/bin/env python3
"""
Phase 6: Publication Table Generation for P²Diff paper.

Generates:
  Table 1 — Main PLINDER Benchmark (112 test complexes, 8 baselines)
  Table 2 — Post-hoc Energy Relaxation Impact
  Table 3 — Multi-target SBDD Case Study (from Phase 3)
  Table 4 — Component Ablation Results (from Phase 4)

Each table is saved as:
  - LaTeX (.tex)    → ready for paper
  - Markdown (.md)  → for README / review
  - JSON (.json)    → for downstream scripting

All output to: paper_records/phase6_publication/tables/
"""

import os
import json
import math
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
WORKSPACE  = "/mnt/nas/keshaw/PharmaDiff"
ERRBAR_DIR = os.path.join(WORKSPACE, "paper_records/phase1_benchmark/error_bars")
PHASE3_DIR = os.path.join(WORKSPACE, "paper_records/phase3_case_studies")
PHASE4_DIR = os.path.join(WORKSPACE, "paper_records/phase4_ablations")
OUT_DIR    = os.path.join(WORKSPACE, "paper_records/phase6_publication/tables")
os.makedirs(OUT_DIR, exist_ok=True)

# ── model ordering & files ─────────────────────────────────────────────────────
MODEL_ORDER = [
    "Native GT",
    "MiDi",
    "Pocket-Only SBDD",
    "PharmaDiff",
    "P²Diff (Ours)",
    "P²Diff + Relax",
]

MODEL_FILE_MAP = {
    "Native GT":        "native_gt_distribution.json",
    "MiDi":             "unconditional_distribution.json",
    "Pocket-Only SBDD": "pocket_only_distribution.json",
    "PharmaDiff":       "p2diff_no_pocket_distribution.json",
    "P²Diff (Ours)":    "p2diff_full_distribution.json",
    "P²Diff + Relax":   "p2diff_full_relaxed_distribution.json",
}

RELAXED_FILE_MAP = {
    "MiDi":             "unconditional_relaxed_distribution.json",
    "Pocket-Only SBDD": "pocket_only_relaxed_distribution.json",
}


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def agg(entries, key, pct=False, negate=False):
    """Return (mean, se) for key from list of per-complex dicts."""
    vals = [e.get(key) for e in entries if e.get(key) is not None]
    if not vals:
        return (float('nan'), float('nan'))
    m = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
    if pct:
        m *= 100; se *= 100
    if negate:
        m = -m
    return (m, se)


def fmt(mean, se, ndigits=2, is_pct=False, better=None):
    """Format mean ± SE for table cells."""
    if math.isnan(mean):
        return "—"
    suffix = "%" if is_pct else ""
    return f"{mean:.{ndigits}f}{suffix} ±{se:.{ndigits}f}"


def bold_best(rows, col_idx, higher_is_better=True):
    """Return row-index of the best value for a given column (ignoring Native GT = row 0)."""
    vals = []
    for i, row in enumerate(rows):
        v = row[col_idx]
        try:
            num = float(str(v).replace("%", "").split("±")[0].strip())
        except Exception:
            num = float('nan')
        vals.append((num, i))
    valid = [(v, i) for v, i in vals if not math.isnan(v) and i > 0]
    if not valid:
        return -1
    if higher_is_better:
        return max(valid, key=lambda x: x[0])[1]
    else:
        return min(valid, key=lambda x: x[0])[1]


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Main PLINDER Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def table1_main_benchmark():
    print("[Table 1] Main PLINDER Benchmark …")
    COLS = ["Model", "Validity (%)", "Match Score", "Vina (kcal/mol)",
            "Severe Clash (%)", "QED", "SAS", "PAINS Pass (%)"]

    rows = []
    for name in MODEL_ORDER:
        fname = MODEL_FILE_MAP.get(name)
        if fname is None:
            continue
        entries = load_json(os.path.join(ERRBAR_DIR, fname))
        if entries is None:
            continue

        valid_mu, valid_se = agg(entries, 'is_valid', pct=True)
        ms_mu, ms_se       = agg(entries, 'match_score')
        vina_mu, vina_se   = agg(entries, 'vina_score')
        clash_mu, clash_se = agg(entries, 'severe_clash', pct=True)
        qed_mu, qed_se     = agg(entries, 'qed')
        sas_mu, sas_se     = agg(entries, 'sas')
        pains_mu, pains_se = agg(entries, 'pains_pass', pct=True)

        rows.append([
            name,
            fmt(valid_mu, valid_se, is_pct=True),
            fmt(ms_mu, ms_se, ndigits=3),
            fmt(vina_mu, vina_se),
            fmt(clash_mu, clash_se, is_pct=True),
            fmt(qed_mu, qed_se, ndigits=3),
            fmt(sas_mu, sas_se),
            fmt(pains_mu, pains_se, is_pct=True),
        ])

    # higher_better flags for cols 1-7
    higher_better = [True, True, False, False, True, False, True]
    best_rows = [bold_best(rows, ci+1, hb)
                 for ci, hb in enumerate(higher_better)]

    # ── Markdown ──────────────────────────────────────────────────────────────
    md_lines = ["# Table 1: Main PLINDER Benchmark", "",
                f"N = 112 test complexes  |  K = 1 pose per complex  |  "
                f"Mean ± SE across all complexes", ""]
    header = "| " + " | ".join(COLS) + " |"
    sep    = "| " + " | ".join(["---"] * len(COLS)) + " |"
    md_lines += [header, sep]
    for i, row in enumerate(rows):
        cells = list(row)
        for ci, br in enumerate(best_rows):
            if br == i:
                cells[ci+1] = f"**{cells[ci+1]}**"
        md_lines.append("| " + " | ".join(cells) + " |")
    md_out = os.path.join(OUT_DIR, "table1_main_benchmark.md")
    with open(md_out, 'w') as f:
        f.write("\n".join(md_lines) + "\n")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    def tex_esc(s):
        return s.replace("²", "$^2$").replace("±", r"$\pm$").replace("%", r"\%")

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main benchmark on PLINDER test set (112 complexes). "
        r"$\uparrow$ higher is better, $\downarrow$ lower is better. "
        r"Best baseline (excluding GT) in \textbf{bold}.}",
        r"\label{tab:main_benchmark}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        " & ".join([r"\textbf{Model}",
                    r"\textbf{Validity $\uparrow$}",
                    r"\textbf{Match Score $\uparrow$}",
                    r"\textbf{Vina $\downarrow$}",
                    r"\textbf{Severe Clash $\downarrow$}",
                    r"\textbf{QED $\uparrow$}",
                    r"\textbf{SAS $\downarrow$}",
                    r"\textbf{PAINS $\uparrow$}"]) + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        cells = [tex_esc(c) for c in row]
        for ci, br in enumerate(best_rows):
            if br == i:
                cells[ci+1] = r"\textbf{" + cells[ci+1] + "}"
        if row[0] == "Native GT":
            tex_lines.append(r"\rowcolor{gray!15}")
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    tex_out = os.path.join(OUT_DIR, "table1_main_benchmark.tex")
    with open(tex_out, 'w') as f:
        f.write("\n".join(tex_lines) + "\n")

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_data = [dict(zip(COLS, row)) for row in rows]
    json_out = os.path.join(OUT_DIR, "table1_main_benchmark.json")
    with open(json_out, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"  [✓] Table 1 saved (md, tex, json)")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Post-hoc Energy Relaxation Impact
# ══════════════════════════════════════════════════════════════════════════════

def table2_relaxation_impact():
    print("[Table 2] Energy Relaxation Impact …")

    pairs = [
        ("P²Diff (Ours)",    "p2diff_full_distribution.json",         "p2diff_full_relaxed_distribution.json"),
        ("MiDi",             "unconditional_distribution.json",       "unconditional_relaxed_distribution.json"),
        ("Pocket-Only SBDD", "pocket_only_distribution.json",         "pocket_only_relaxed_distribution.json"),
        ("PharmaDiff",       "p2diff_no_pocket_distribution.json",    "pharmadiff_relaxed_distribution.json"),
    ]

    COLS = ["Model", "Vina Pre", "Vina Post", "ΔVina",
            "Clash Pre (%)", "Clash Post (%)", "ΔClash (pp)"]
    rows = []
    for name, pre_f, post_f in pairs:
        pre  = load_json(os.path.join(ERRBAR_DIR, pre_f))
        post = load_json(os.path.join(ERRBAR_DIR, post_f))
        if pre is None:
            continue
        vpre_mu, vpre_se   = agg(pre,  'vina_score')
        cpre_mu, cpre_se   = agg(pre,  'severe_clash', pct=True)
        if post is not None:
            vpost_mu, vpost_se = agg(post, 'vina_score')
            cpost_mu, cpost_se = agg(post, 'severe_clash', pct=True)
            delta_v = vpost_mu - vpre_mu
            delta_c = cpost_mu - cpre_mu
            rows.append([
                name,
                fmt(vpre_mu, vpre_se),
                fmt(vpost_mu, vpost_se),
                f"{delta_v:+.2f}",
                fmt(cpre_mu, cpre_se, is_pct=True),
                fmt(cpost_mu, cpost_se, is_pct=True),
                f"{delta_c:+.1f} pp",
            ])
        else:
            rows.append([name,
                         fmt(vpre_mu, vpre_se), "—", "—",
                         fmt(cpre_mu, cpre_se, is_pct=True), "—", "—"])

    # Markdown
    md_lines = ["# Table 2: Post-hoc Energy Relaxation Impact", "",
                "pp = percentage points. ΔVina < 0 = improved (lower energy).", ""]
    header = "| " + " | ".join(COLS) + " |"
    sep    = "| " + " | ".join(["---"] * len(COLS)) + " |"
    md_lines += [header, sep]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    with open(os.path.join(OUT_DIR, "table2_relaxation_impact.md"), 'w') as f:
        f.write("\n".join(md_lines) + "\n")

    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Impact of post-hoc UFF energy relaxation. "
        r"$\Delta$Vina $<$ 0 indicates improved docking energy.}",
        r"\label{tab:relaxation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Vina Pre} & \textbf{Vina Post} & "
        r"\textbf{$\Delta$Vina} & \textbf{Clash Pre} & \textbf{Clash Post} & "
        r"\textbf{$\Delta$Clash} \\",
        r"\midrule",
    ]
    for row in rows:
        def te(s): return s.replace("²", "$^2$").replace("±", r"$\pm$").replace("%", r"\%")
        tex_lines.append(" & ".join([te(c) for c in row]) + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "table2_relaxation_impact.tex"), 'w') as f:
        f.write("\n".join(tex_lines) + "\n")

    json_data = [dict(zip(COLS, row)) for row in rows]
    with open(os.path.join(OUT_DIR, "table2_relaxation_impact.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"  [✓] Table 2 saved (md, tex, json)")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Multi-target SBDD Case Study
# ══════════════════════════════════════════════════════════════════════════════

def table3_case_study():
    print("[Table 3] Multi-target Case Study …")

    res_path = os.path.join(PHASE3_DIR, "phase3_case_studies_results.json")
    results  = load_json(res_path)

    TARGET_NAMES = {
        0:   "CDK2/CDK6 Kinase ATP Cleft",
        2:   "TGFβR1 Heterocyclic Pocket",
        5:   "VEGFR2 Allosteric Channel",
        105: "BRD4 Bromodomain",
    }

    COLS = ["Target", "# Poses", "Best Vina", "Top-5 Vina μ",
            "Severe Clash (%)", "Min Dist (Å)", "QED", "SAS", "MW (Da)"]

    md_lines = ["# Table 3: Multi-target SBDD Case Study", "",
                r"P²Diff (Ours)  ·  K=30 poses per target  ·  Best-of-K lead shown",
                ""]

    if results is None:
        md_lines += ["> Phase 3 results not yet available (sampling in progress).", ""]
        rows = []
    else:
        rows = []
        for r in results:
            tidx = r.get('target_idx', '?')
            tname = TARGET_NAMES.get(tidx, r.get('target_name', str(tidx)))
            poses_count = r.get('n_poses', r.get('poses_generated', '?'))
            best_v = r.get('best_vina', r.get('vina_top1', float('nan')))
            top5_v = r.get('top5_vina_mean', r.get('vina_top5_mean', float('nan')))
            clash_val = r.get('severe_clash_pct', 100 * r.get('severe_clash_rate', 0.0))
            min_dist = r.get('min_pocket_dist', r.get('min_pocket_dist_mean', float('nan')))
            qed_val = r.get('mean_qed', r.get('qed_mean', float('nan')))
            sas_val = r.get('mean_sas', r.get('sas_mean', float('nan')))
            mw_val = r.get('mean_mw', r.get('mw_mean', float('nan')))
            row = [
                tname,
                str(poses_count),
                f"{best_v:.2f}",
                f"{top5_v:.2f}",
                f"{clash_val:.1f}%",
                f"{min_dist:.2f}",
                f"{qed_val:.3f}" if not math.isnan(qed_val) else "—",
                f"{sas_val:.2f}" if not math.isnan(sas_val) else "—",
                f"{mw_val:.1f}" if not math.isnan(mw_val) else "—",
            ]
            rows.append(row)

    header = "| " + " | ".join(COLS) + " |"
    sep    = "| " + " | ".join(["---"] * len(COLS)) + " |"
    md_lines += [header, sep]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    if not rows:
        md_lines.append("| (results pending) | — | — | — | — | — | — | — | — |")

    with open(os.path.join(OUT_DIR, "table3_case_study.md"), 'w') as f:
        f.write("\n".join(md_lines) + "\n")

    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-target SBDD Case Study. "
        r"K=30 poses generated per target. Best-of-K lead metrics reported.}",
        r"\label{tab:case_study}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{Target} & \textbf{\#Poses} & \textbf{Best Vina} & "
        r"\textbf{Top-5 Vina $\mu$} & \textbf{Clash $\downarrow$} & "
        r"\textbf{Min Dist} & \textbf{QED} & \textbf{SAS} & \textbf{MW} \\",
        r"\midrule",
    ]
    for row in rows:
        def te(s): return s.replace("²", "$^2$").replace("±", r"$\pm$").replace("%", r"\%")
        tex_lines.append(" & ".join([te(c) for c in row]) + r" \\")
    if not rows:
        tex_lines.append(r"\multicolumn{9}{c}{\textit{Results pending Phase 3 completion}} \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "table3_case_study.tex"), 'w') as f:
        f.write("\n".join(tex_lines) + "\n")

    json_data = results if results else []
    with open(os.path.join(OUT_DIR, "table3_case_study.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"  [✓] Table 3 saved (md, tex, json)")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — Component Ablation Results
# ══════════════════════════════════════════════════════════════════════════════

def table4_ablation():
    print("[Table 4] Component Ablation …")

    abl_path = os.path.join(PHASE4_DIR, "ablation_sampling_results.json")
    abl_data = load_json(abl_path)

    COLS = ["Ablation ID", "Description", "Steps", "Validity (%)",
            "Match Score", "Vina (kcal/mol)", "Clash (%)", "Runtime (s)"]

    ABLATION_LABELS = {
        "S_500": ("A4", "Full model, standard schedule (500 steps)"),
        "S_250": ("A5", "2× acceleration (250 eff. steps)"),
        "S_100": ("A6", "5× acceleration (100 eff. steps)"),
    }

    rows = []

    # Part I: Conditioning Component Ablations (from Phase 1 benchmarks, N=112)
    cond_models = [
        ("Full Model", "P²Diff (Full Joint Pocket + Pharma)", "500", "p2diff_full_distribution.json"),
        ("A1: w/o Pocket", "PharmaDiff (Pharma Inpainting Only)", "500", "p2diff_no_pocket_distribution.json"),
        ("A2: w/o Pharma", "Pocket-Only SBDD (Pocket Conditioning Only)", "500", "pocket_only_distribution.json"),
        ("A0: Unconditional", "MiDi (Unconditional 3D Graph Diffusion)", "500", "unconditional_distribution.json"),
    ]
    for abl_id, desc, steps, fname in cond_models:
        entries = load_json(os.path.join(ERRBAR_DIR, fname))
        if entries:
            v_mu, v_se = agg(entries, 'is_valid', pct=True)
            ms_mu, ms_se = agg(entries, 'match_score')
            vina_mu, vina_se = agg(entries, 'vina_score')
            clash_mu, clash_se = agg(entries, 'severe_clash', pct=True)
            rows.append([
                abl_id, desc, steps,
                fmt(v_mu, v_se, is_pct=True),
                fmt(ms_mu, ms_se, ndigits=3),
                fmt(vina_mu, vina_se),
                fmt(clash_mu, clash_se, is_pct=True),
                "~59s",
            ])

    # Part II: Trajectory Schedule Acceleration Ablations (from Phase 4, N=20)
    if abl_data is not None:
        for entry in abl_data:
            steps = entry.get('steps', 500)
            if steps == 500:
                abl_id, desc = "A4: Standard Trajectory", "Full schedule (500 steps, S_500)"
            elif steps == 250:
                abl_id, desc = "A5: 2× Acceleration", "Halved schedule (250 steps, S_250)"
            elif steps == 100:
                abl_id, desc = "A6: 5× Acceleration", "Fast schedule (100 steps, S_100)"
            else:
                abl_id, desc = f"A-{steps}", f"Accelerated ({steps} steps)"

            val_pct = entry.get('validity_pct', float('nan'))
            ms_val = entry.get('pharma_match', float('nan'))
            vina_val = entry.get('mean_vina', float('nan'))
            top1_vina = entry.get('top1_vina', float('nan'))
            clash_val = entry.get('severe_clash_pct', float('nan'))
            rt_val = entry.get('mean_time_sec', float('nan'))
            
            rows.append([
                abl_id,
                desc,
                str(steps),
                f"{val_pct:.1f}%",
                f"{ms_val:.3f}",
                f"{vina_val:.2f} (top-1: {top1_vina:.2f})",
                f"{clash_val:.1f}%",
                f"{rt_val:.1f}s",
            ])

    md_lines = ["# Table 4: Comprehensive Systematic Component Ablation Matrix", "",
                "### Part I: Conditioning Architecture Ablation (N=112 Test Complexes)",
                "Evaluates necessity of pocket cross-attention (A1) and spatial pharmacophore anchors (A2).", ""]
    header = "| " + " | ".join(COLS) + " |"
    sep    = "| " + " | ".join(["---"] * len(COLS)) + " |"
    md_lines += [header, sep]
    for row in rows[:4]:
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines += ["", "### Part II: Reverse Diffusion Trajectory Schedule Ablation (N=20 Complexes)",
                 "Evaluates sampling throughput and quality trade-offs under accelerated inference.", ""]
    md_lines += [header, sep]
    for row in rows[4:]:
        md_lines.append("| " + " | ".join(row) + " |")

    with open(os.path.join(OUT_DIR, "table4_ablation.md"), 'w') as f:
        f.write("\n".join(md_lines) + "\n")

    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comprehensive Component Ablation Matrix. Part I evaluates architectural conditioning components (N=112). Part II evaluates reverse diffusion trajectory acceleration (N=20).}",
        r"\label{tab:ablation}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{llcccccr}",
        r"\toprule",
        r"\textbf{ID} & \textbf{Configuration} & \textbf{Steps} & "
        r"\textbf{Validity} & \textbf{Match Score} & "
        r"\textbf{Vina} & \textbf{Severe Clash} & \textbf{Runtime} \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{\textbf{Part I: Conditioning Architecture Ablation (N=112)}}} \\",
    ]
    for row in rows[:4]:
        def te(s): return s.replace("²", "$^2$").replace("±", r"$\pm$").replace("%", r"\%")
        tex_lines.append(" & ".join([te(c) for c in row]) + r" \\")
    tex_lines += [
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{\textbf{Part II: Trajectory Schedule Acceleration (N=20)}}} \\",
    ]
    for row in rows[4:]:
        def te(s): return s.replace("²", "$^2$").replace("±", r"$\pm$").replace("%", r"\%")
        tex_lines.append(" & ".join([te(c) for c in row]) + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "table4_ablation.tex"), 'w') as f:
        f.write("\n".join(tex_lines) + "\n")

    json_data = [dict(zip(COLS, row)) for row in rows]
    with open(os.path.join(OUT_DIR, "table4_ablation.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"  [✓] Table 4 saved (md, tex, json)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6 — Table Generation")
    print("=" * 60)

    table1_main_benchmark()
    table2_relaxation_impact()
    table3_case_study()
    table4_ablation()

    print(f"\nAll tables saved to: {OUT_DIR}")
