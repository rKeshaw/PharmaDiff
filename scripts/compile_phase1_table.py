"""
scripts/compile_phase1_table.py
===============================
Compiles Table 1 (Main Benchmark) for the ICLR 2027 paper on P^2Diff.
Generates both LaTeX and JSON versions in paper_records/phase1_benchmark/.
"""

import json

# Metrics data compiled directly from ground truth calculation and model test logs
data = {
    "metrics": [
        {
            "category": "Chemical Feasibility",
            "name": "Validity (\\%)",
            "arrow": "\\uparrow",
            "gt": "100.0",
            "nopocket": "27.7",
            "full": "2.7"
        },
        {
            "category": "Chemical Feasibility",
            "name": "Uniqueness (\\%)",
            "arrow": "\\uparrow",
            "gt": "100.0",
            "nopocket": "100.0",
            "full": "100.0"
        },
        {
            "category": "Chemical Feasibility",
            "name": "Novelty (\\%)",
            "arrow": "\\uparrow",
            "gt": "-",
            "nopocket": "100.0",
            "full": "100.0"
        },
        {
            "category": "Chemical Feasibility",
            "name": "SAS (Synthesizability)",
            "arrow": "\\downarrow",
            "gt": "3.70 \\pm 0.92",
            "nopocket": "4.39",
            "full": "5.47"
        },
        {
            "category": "Chemical Feasibility",
            "name": "QED (Drug-likeness)",
            "arrow": "\\uparrow",
            "gt": "0.45 \\pm 0.17",
            "nopocket": "0.47",
            "full": "0.49"
        },
        {
            "category": "Chemical Feasibility",
            "name": "PAINS Pass (\\%)",
            "arrow": "\\uparrow",
            "gt": "95.5",
            "nopocket": "100.0",
            "full": "100.0"
        },
        {
            "category": "Chemical Feasibility",
            "name": "Mean Ring Count",
            "arrow": "-",
            "gt": "2.88",
            "nopocket": "1.87",
            "full": "1.67"
        },
        {
            "category": "Pharmacophore Alignment",
            "name": "PGMG Match Score (\\%)",
            "arrow": "\\uparrow",
            "gt": "100.0",
            "nopocket": "44.8",
            "full": "\\textbf{56.7}"
        },
        {
            "category": "Pharmacophore Alignment",
            "name": "Match Precision (\\%MS=1)",
            "arrow": "\\uparrow",
            "gt": "100.0",
            "nopocket": "19.4",
            "full": "\\textbf{33.3}"
        },
        {
            "category": "Pharmacophore Alignment",
            "name": "High-Fidelity (\\%MS $\\ge$ 0.8)",
            "arrow": "\\uparrow",
            "gt": "100.0",
            "nopocket": "32.3",
            "full": "\\textbf{33.3}"
        },
        {
            "category": "Pharmacophore Alignment",
            "name": "RDKit Pharma Match (\\%)",
            "arrow": "\\uparrow",
            "gt": "100.0",
            "nopocket": "38.7",
            "full": "33.3"
        },
        {
            "category": "Receptor Interaction",
            "name": "Vina Top-1 (kcal/mol)",
            "arrow": "\\downarrow",
            "gt": "-0.79",
            "nopocket": "N/A",
            "full": "\\textbf{-13.39}"
        },
        {
            "category": "Receptor Interaction",
            "name": "Vina Top-10 (kcal/mol)",
            "arrow": "\\downarrow",
            "gt": "-0.19",
            "nopocket": "N/A",
            "full": "\\textbf{-6.02}"
        },
        {
            "category": "Receptor Interaction",
            "name": "Pocket Contact Ratio (\\%)",
            "arrow": "\\uparrow",
            "gt": "100.0",
            "nopocket": "N/A",
            "full": "63.4"
        },
        {
            "category": "Receptor Interaction",
            "name": "Severe Clash ($<1.5$ \\AA) (\\%)",
            "arrow": "\\downarrow",
            "gt": "0.89",
            "nopocket": "N/A",
            "full": "84.0"
        },
        {
            "category": "Receptor Interaction",
            "name": "Mild Clash ($<2.0$ \\AA) (\\%)",
            "arrow": "\\downarrow",
            "gt": "1.79",
            "nopocket": "N/A",
            "full": "91.4"
        },
        {
            "category": "Distribution Divergence",
            "name": "Atom Types TV",
            "arrow": "\\downarrow",
            "gt": "0.000",
            "nopocket": "0.138",
            "full": "\\textbf{0.135}"
        },
        {
            "category": "Distribution Divergence",
            "name": "Bond Types TV",
            "arrow": "\\downarrow",
            "gt": "0.000",
            "nopocket": "\\textbf{0.010}",
            "full": "0.048"
        },
        {
            "category": "Distribution Divergence",
            "name": "Valency $W_1$",
            "arrow": "\\downarrow",
            "gt": "0.000",
            "nopocket": "\\textbf{0.219}",
            "full": "1.462"
        },
        {
            "category": "Distribution Divergence",
            "name": "Bond Lengths $W_1$ (\\AA)",
            "arrow": "\\downarrow",
            "gt": "0.000",
            "nopocket": "\\textbf{0.016}",
            "full": "0.044"
        },
        {
            "category": "Distribution Divergence",
            "name": "Bond Angles $W_1$ ($^\\circ$)",
            "arrow": "\\downarrow",
            "gt": "0.000",
            "nopocket": "\\textbf{23.19}",
            "full": "39.02"
        }
    ]
}

latex_table = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Main Benchmark on PLINDER Test Set (112 complexes).} Comparison across native co-crystal ground truth, the pocket-blind scaled baseline ($\mathrm{P^2Diff}$-NoPocket), and the full joint pocket-pharmacophore conditioned model ($\mathrm{P^2Diff}$). Bold numbers indicate superior performance between the two generative diffusion models.}
\label{tab:main_plinder_benchmark}
\begin{tabular}{llccc}
\toprule
\textbf{Evaluation Category} & \textbf{Metric} & \textbf{Native Co-Crystals} & $\mathbf{P^2\text{\textbf{Diff}}}\text{\textbf{-NoPocket}}$ & $\mathbf{P^2\text{\textbf{Diff}}}$ \textbf{(Full)} \\
\midrule
"""

current_cat = ""
for item in data["metrics"]:
    cat_str = item["category"] if item["category"] != current_cat else ""
    current_cat = item["category"]
    arrow = f" (${item['arrow']}$)" if item["arrow"] != "-" else ""
    latex_table += f"{cat_str:28s} & {item['name'] + arrow:36s} & {item['gt']:20s} & {item['nopocket']:20s} & {item['full']:20s} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table*}
"""

with open("paper_records/phase1_benchmark/table1_main_benchmark.tex", "w") as f:
    f.write(latex_table)

with open("paper_records/phase1_benchmark/table1_main_benchmark.json", "w") as f:
    json.dump(data, f, indent=2)

print("Table 1 successfully generated in paper_records/phase1_benchmark/table1_main_benchmark.tex and .json")
