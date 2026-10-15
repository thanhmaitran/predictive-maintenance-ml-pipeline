"""
Regenerates two README figures from result files (never from typed-in numbers):

    reports/figures/model_comparison.png
        <- reports/metrics/model_comparison.csv
    reports/figures/data_quality_gate_failure_demo.png
        <- reports/metrics/data_quality_report_failure_demo.json
           (the real report written when the gate was run on a deliberately
            corrupted copy of the dataset: Torque = -10 and Type = "X")

Colors: categorical slots 1-2 and the reserved status colors of the
validated reference palette; text uses ink tokens, never the series color.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834"}
LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
STATUS = {"PASS": ("#e3f4e3", "#0b6b0b", "✓ PASS"),
          "WARNING": ("#fff1d0", "#7a5200", "! WARNING"),
          "FAIL": ("#fbe0e0", "#9b1c1c", "✗ FAIL")}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2})


def model_comparison(csv_path="reports/metrics/model_comparison.csv",
                     out="reports/figures/model_comparison.png"):
    df = pd.read_csv(csv_path, index_col=0)
    metrics = [("f1", "F1"), ("recall", "Recall"), ("pr_auc", "PR-AUC")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width, gap = 0.34, 0.03
    for i, model in enumerate(["logistic_regression", "random_forest"]):
        xs = [j + (i - 0.5) * (width + gap) for j in range(len(metrics))]
        vals = [df.loc[model, m] for m, _ in metrics]
        bars = ax.bar(xs, vals, width=width, color=SERIES[model], label=LABELS[model], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.set_xticks(range(len(metrics)), [n for _, n in metrics], fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score (held-out test set, threshold 0.5)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper center", ncol=2, labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Model comparison on imbalanced failure detection", loc="left", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)


NAMES = {"schema": "Required columns present", "row_count": "Row count > 0", "duplicate_udi": "Duplicate UDI",
         "product_type": "Valid product types (L/M/H)", "temperature_positive": "Valid temperatures (> 0)",
         "rotational_speed_positive": "Valid rotational speed (> 0)", "torque_non_negative": "Valid torque (>= 0)",
         "tool_wear_non_negative": "Valid tool wear (>= 0)", "target_values_valid": "Valid target ({0,1})",
         "missing_values": "Missing values", "leakage_guard": "Target leakage",
         "failure_mode_consistency": "Failure-mode consistency"}


def gate_table(report_path, out, title):
    rep = json.load(open(report_path))
    rows = [(NAMES[k], v) for k, v in rep["checks"].items()] + [("Overall", rep["status"])]
    fig, ax = plt.subplots(figsize=(7, 5.6), facecolor=SURFACE)
    ax.axis("off")
    tbl = ax.table(cellText=[[n, STATUS[s][2]] for n, s in rows], colLabels=["Check", "Result"],
                   cellLoc="left", colLoc="left", loc="center", colWidths=[0.7, 0.3])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for c in range(2):
        tbl[0, c].set_facecolor("#2a2a28")
        tbl[0, c].set_text_props(color="white", weight="bold")
    for i, (n, s) in enumerate(rows, start=1):
        bg, fg, _ = STATUS[s]
        for c in range(2):
            cell = tbl[i, c]
            cell.set_facecolor(bg)
            cell.set_edgecolor(SURFACE)
            cell.set_text_props(color=INK if c == 0 else fg, weight="bold" if (c == 1 or n == "Overall") else None)
    ax.set_title(title, fontsize=11.5, weight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def compose_stack(compose="docker-compose.yml", out="reports/figures/docker_compose_stack.png"):
    """Service graph read from docker-compose.yml. NOT terminal output of `docker compose up`."""
    import yaml
    from matplotlib.patches import FancyBboxPatch
    svcs = yaml.safe_load(open(compose))["services"]
    layout = {"postgres": (0.5, 1.2), "airflow-init": (1.75, 0.5), "airflow-webserver": (3.0, 1.75),
              "airflow-scheduler": (3.0, 0.5), "pipeline": (4.25, 1.2)}
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=SURFACE)
    ax.set_xlim(-0.2, 4.9); ax.set_ylim(0.0, 2.3); ax.axis("off")
    roles = {"postgres": "metadata DB", "airflow-init": "db migrate + admin user", "airflow-webserver": "",
             "airflow-scheduler": "runs DAGs", "pipeline": "standalone: pytest / src modules"}
    for name, (x, y) in layout.items():
        if name not in svcs:
            continue
        sv = svcs[name]
        port = f"port {sv['ports'][0]}" if sv.get("ports") else roles.get(name, "")
        ax.add_patch(FancyBboxPatch((x - 0.52, y - 0.3), 1.04, 0.6, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc="#e7eef9" if name != "pipeline" else "#fdebe3",
                                    ec=SERIES["logistic_regression"] if name != "pipeline" else SERIES["random_forest"], lw=1.4))
        ax.text(x, y + 0.1, name, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
        ax.text(x, y - 0.12, port, ha="center", va="center", fontsize=7.5, color=INK_2)
    for name, sv in svcs.items():
        deps = sv.get("depends_on") or {}
        for dep, cond in (deps.items() if isinstance(deps, dict) else [(d, {}) for d in deps]):
            (x0, y0), (x1, y1) = layout[dep], layout[name]
            ax.annotate("", xy=(x1 - 0.53, y1), xytext=(x0 + 0.53, y0),
                        arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.1, shrinkA=0, shrinkB=0))
    ax.set_title("docker-compose.yml — services and depends_on (arrow: dependency → dependent)",
                 loc="left", fontsize=12, weight="bold")
    fig.text(0.01, 0.01, "Read from docker-compose.yml. The stack was NOT started in the development sandbox; "
             "this is not `docker compose up` output.", fontsize=8, color=INK_2)
    fig.savefig(out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    model_comparison()
    gate_table("reports/metrics/data_quality_report_failure_demo.json",
               "reports/figures/data_quality_gate_failure_demo.png",
               "Data Quality Gate — injected failure test (Torque = -10, Type = 'X')")
    gate_table("reports/metrics/data_quality_report.json",
               "reports/figures/data_quality_gate.png",
               "Data Quality Gate — validation results (real dataset run)")
    compose_stack()
    print("figures written")
