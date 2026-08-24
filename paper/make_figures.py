"""Generate argument-driven publication figures from the locked result artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIGURE_DIR = HERE / "figures"
LOCK_PATH = HERE / "RESULTS_LOCK.json"

PALETTE = {
    "blue": "#0F4D92",
    "blue_light": "#DCEAF7",
    "green": "#4F9D69",
    "green_light": "#DDF3DE",
    "red": "#B64342",
    "red_light": "#F6CFCB",
    "teal": "#42949E",
    "violet": "#8B5A8C",
    "gray": "#767676",
    "gray_light": "#ECEFF1",
    "ink": "#20272B",
    "white": "#FFFFFF",
}


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def load_locked_results() -> dict:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    artifact = REPO_ROOT / lock["artifact"]
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest().upper()
    if digest != lock["sha256"]:
        raise RuntimeError(
            f"Locked result hash mismatch: expected {lock['sha256']}, found {digest}"
        )
    return json.loads(artifact.read_text(encoding="utf-8"))


def finalize_figure(fig: plt.Figure, basename: str, pad_inches: float = 0.05) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            FIGURE_DIR / f"{basename}.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=pad_inches,
        )
    plt.close(fig)


def _box(
    ax, x, y, width, height, title, lines, color, fill,
    title_size=11.5, text_size=10.5, text_y=0.36, text_x=0.08,
):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.05",
        linewidth=1.5, edgecolor=color, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + 0.08 * width, y + 0.82 * height, title,
            fontsize=title_size, fontweight="bold", color=color, va="center")
    ax.text(x + text_x * width, y + text_y * height, lines,
            fontsize=text_size, color=PALETTE["ink"], va="center", linespacing=1.2)


def _arrow(ax, start, end, color=PALETTE["ink"], style="-"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.4, linestyle=style, color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def architecture_figure() -> None:
    # A deliberately sparse two-lane diagram remains legible at manuscript width.
    fig, ax = plt.subplots(figsize=(8.0, 3.42))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0.30, 4.75)
    ax.axis("off")
    _box(ax, 0.18, 3.30, 3.10, 1.30, "1  Calibration",
         r"confidence $c_k$; context $h_k$" "\n"
         r"calibration label $y_k$" "\n"
         r"score $s_k=f(c_k,h_k)$",
         PALETTE["violet"], "#F3EAF3", title_size=9.6, text_size=8.5)
    _box(ax, 3.72, 3.30, 3.35, 1.30, r"2  Risk gate (fixed $\mathcal{F}\times\mathcal{T}$)",
         r"$U_{\rm CP}\leq\alpha$: admit $K^+$" "\n"
         r"otherwise: $K^+=\varnothing$" "\n"
         r"fixed family; $N\geq n_{\min}$",
         PALETTE["violet"], "#F3EAF3", title_size=9.1, text_size=8.4)

    lower_y, lower_h = 0.98, 1.38
    _box(ax, 0.18, lower_y, 2.05, lower_h, "3a  Data $X$",
         r"raw observations" "\n" r"$X\in\mathbb{R}^{n\times p}$",
         PALETTE["blue"], PALETTE["blue_light"], title_size=9.0, text_size=8.8)
    _box(ax, 2.55, lower_y, 2.20, lower_h, r"3b  Support  $\mathcal{S}(X)$",
          "data-supported\npairs only\nfixed before fusion",
          PALETTE["blue"], PALETTE["blue_light"], title_size=8.8, text_size=8.6)
    _box(ax, 5.07, lower_y, 3.55, lower_h, "3c  Permissioned projection",
          r"data directions + certified $K^+$" "\n"
          r"orient $\mathcal{S}(X)$; no new edges" "\n"
          r"exact DP: $p\leq16$" "\n"
          r"relocation: $p>16$",
          PALETTE["teal"], "#E3F1F2", title_size=9.0, text_size=8.35,
          text_y=0.42, text_x=0.035)
    _box(ax, 8.95, lower_y, 2.83, lower_h, "4  Audited DAG",
          "support-preserving\nand acyclic\nnot graph FDR / SHD",
          PALETTE["green"], "#EDF6EC", title_size=9.0, text_size=8.5)

    _arrow(ax, (3.28, 3.95), (3.72, 3.95), PALETTE["violet"])
    lower_mid = lower_y + lower_h / 2
    _arrow(ax, (2.23, lower_mid), (2.55, lower_mid), PALETTE["blue"])
    _arrow(ax, (4.75, lower_mid), (5.07, lower_mid), PALETTE["blue"])
    _arrow(ax, (8.62, lower_mid), (8.95, lower_mid), PALETTE["teal"])
    _arrow(ax, (5.40, 3.30), (5.40, 2.42), PALETTE["violet"])
    ax.text(5.55, 2.80, "certified only", fontsize=8.6,
            fontweight="bold", color=PALETTE["violet"], va="center")
    ax.text(0.22, 2.72, r"$y_k$: calibration only; never deployed.",
            fontsize=8.5, color=PALETTE["violet"])

    ax.text(6.0, 0.46,
            r"Invariant: $\mathrm{supp}(A)\subseteq\mathcal{S}(X)$ and $A$ is acyclic",
            fontsize=9.5, fontweight="bold", ha="center", color=PALETTE["ink"],
            bbox=dict(facecolor=PALETTE["white"], edgecolor=PALETTE["teal"], pad=3.0))
    # Figure 1 is a compact architecture panel, so retain only a hairline
    # export margin rather than the padding used for plots with tick labels.
    finalize_figure(fig, "figure1_architecture", pad_inches=0.015)


def _condition_map(exp7: dict) -> dict:
    return {
        (
            row["sem_family"], row["confidence_regime"],
            float(row["direction_accuracy"]), float(row["spurious_ratio"]),
        ): row
        for row in exp7["conditions"]
    }


def main_results_figure(payload: dict) -> None:
    exp7 = payload["experiments"]["exp7_risk_controlled_projection"]
    scale = payload["experiments"]["exp7_scale_sensitivity"]
    conditions = _condition_map(exp7)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    accuracies = np.array([0.55, 0.75, 0.90])
    for spurious, color, marker, label in (
        (0.0, PALETTE["blue"], "o", "No spurious pairs"),
        (0.5, PALETTE["teal"], "s", "50% spurious pairs"),
    ):
        means, lower, upper = [], [], []
        for accuracy in accuracies:
            rows = [conditions[(family, "informative", accuracy, spurious)]
                    for family in ("linear_gaussian", "nonlinear_additive")]
            estimates = [row["summary"]["risk_controlled_global_contract"]
                         ["paired_delta_f1_vs_data"]["estimate"] for row in rows]
            lows = [row["summary"]["risk_controlled_global_contract"]
                    ["paired_delta_f1_vs_data"]["ci_lower"] for row in rows]
            highs = [row["summary"]["risk_controlled_global_contract"]
                     ["paired_delta_f1_vs_data"]["ci_upper"] for row in rows]
            means.append(np.mean(estimates))
            lower.append(np.mean(estimates) - min(lows))
            upper.append(max(highs) - np.mean(estimates))
        ax_a.errorbar(
            accuracies, means, yerr=np.vstack([lower, upper]),
            color=color, marker=marker, linewidth=2, markersize=6,
            capsize=3, label=label,
        )
    ax_a.axhline(0, color=PALETTE["gray"], linewidth=1)
    ax_a.set_xlabel("Directional-source accuracy")
    ax_a.set_ylabel(r"Paired $\Delta$ directed $\mathrm{F1}$")
    ax_a.set_xticks(accuracies)
    ax_a.set_title("a  Certified informative evidence can improve orientation")
    ax_a.legend(loc="upper left")

    columns = [(acc, spur) for acc in accuracies for spur in (0.0, 0.5)]
    matrix = np.zeros((2, len(columns)))
    regimes = ("informative", "uninformative")
    for row_index, regime in enumerate(regimes):
        for column_index, (accuracy, spurious) in enumerate(columns):
            statuses = [
                conditions[(family, regime, accuracy, spurious)]
                ["risk_certificate"]["status"] == "certified"
                for family in ("linear_gaussian", "nonlinear_additive")
            ]
            matrix[row_index, column_index] = np.mean(statuses)
    decision_cmap = ListedColormap([PALETTE["red_light"], PALETTE["green_light"]])
    ax_b.pcolormesh(
        np.arange(matrix.shape[1] + 1) - 0.5,
        np.arange(matrix.shape[0] + 1) - 0.5,
        matrix,
        cmap=decision_cmap,
        vmin=0,
        vmax=1,
        shading="flat",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = "C" if matrix[i, j] == 1 else "A"
            ax_b.text(j, i, label, ha="center", va="center", fontsize=12,
                      color=PALETTE["ink"],
                      fontweight="bold")
    ax_b.set_yticks(range(2), ["Informative", "Uninformative"])
    ax_b.set_xticks(range(len(columns)),
                    [f"{acc:.2f}\n{int(spur * 100)}%" for acc, spur in columns])
    ax_b.set_xlabel("Source accuracy / spurious-pair ratio")
    ax_b.set_title("b  The learned gate fails closed (C=certify; A=abstain)")
    ax_b.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, len(regimes), 1), minor=True)
    ax_b.grid(which="minor", color=PALETTE["white"], linewidth=1.4)
    ax_b.tick_params(which="minor", bottom=False, left=False)

    methods = (
        ("risk_controlled_global_contract", "Risk-controlled global", PALETTE["blue"], "o"),
        ("uncalibrated_global_contract", "Uncalibrated global", PALETTE["violet"], "s"),
        ("risk_controlled_local_projection", "Risk-controlled local", PALETTE["teal"], "D"),
        ("naive_unrestricted_union", "Naive prior union", PALETTE["red"], "X"),
    )
    for key, label, color, marker in methods:
        deltas = [row["summary"][key]["paired_delta_f1_vs_data"]["estimate"]
                  for row in exp7["conditions"]]
        harm = [row["summary"][key]["harm_probability_vs_data"]
                for row in exp7["conditions"]]
        ax_c.scatter(np.mean(deltas), max(harm), s=75, marker=marker,
                     color=color, edgecolor="black", linewidth=0.6, label=label, zorder=3)
        ax_c.plot([min(deltas), np.mean(deltas)], [max(harm), max(harm)],
                  color=color, linewidth=2)
        ax_c.plot([min(deltas), min(deltas)], [max(harm) - 0.012, max(harm) + 0.012],
                  color=color, linewidth=1.5)
    ax_c.axvline(0, color=PALETTE["gray"], linewidth=1)
    ax_c.set_xlabel(r"Mean $\Delta\,\mathrm{F1}$")
    ax_c.set_ylabel("Maximum observed harm rate")
    ax_c.set_title("c  Risk control preserves utility while reducing worst-condition harm")
    handles, labels = ax_c.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=PALETTE["ink"], linewidth=1.8,
                          marker="|", markersize=9, markeredgewidth=1.5))
    labels.append("left tick: worst condition")
    ax_c.legend(handles, labels, loc="lower left", fontsize=9.0)

    sample_sizes = sorted({row["n_samples"] for row in scale["settings"]})
    variable_counts = sorted({row["n_variables"] for row in scale["settings"]})
    heat = np.zeros((len(variable_counts), len(sample_sizes)))
    for row in scale["settings"]:
        i = variable_counts.index(row["n_variables"])
        j = sample_sizes.index(row["n_samples"])
        heat[i, j] = row["summary"]["risk_controlled_global_contract"][
            "paired_delta_f1_vs_data"
        ]["estimate"]
    heat_norm = Normalize(vmin=max(0.0, heat.min() - 0.012), vmax=heat.max())
    image = ax_d.pcolormesh(
        np.arange(heat.shape[1] + 1) - 0.5,
        np.arange(heat.shape[0] + 1) - 0.5,
        heat,
        cmap="YlGnBu",
        norm=heat_norm,
        shading="flat",
    )
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax_d.text(j, i, f"{heat[i, j]:.3f}", ha="center", va="center",
                      fontsize=10.5, color="white" if heat[i, j] > 0.13 else PALETTE["ink"])
    ax_d.set_xticks(range(len(sample_sizes)), sample_sizes)
    ax_d.set_yticks(range(len(variable_counts)), variable_counts)
    ax_d.set_xlabel("Samples per graph")
    ax_d.set_ylabel("Variables")
    ax_d.set_title("d  Gain persists across graph and sample scales")
    ax_d.set_xticks(np.arange(-0.5, len(sample_sizes), 1), minor=True)
    ax_d.set_yticks(np.arange(-0.5, len(variable_counts), 1), minor=True)
    ax_d.grid(which="minor", color=PALETTE["white"], linewidth=1.2)
    ax_d.tick_params(which="minor", bottom=False, left=False)
    # Draw the continuous color scale with vector rectangles so the complete
    # figure remains vector-only in the exported PDF.
    cbar_ax = ax_d.inset_axes([1.06, 0.0, 0.045, 1.0])
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(heat_norm.vmin, heat_norm.vmax)
    cmap = plt.get_cmap("YlGnBu")
    color_levels = np.linspace(heat_norm.vmin, heat_norm.vmax, 80)
    for lower, upper in zip(color_levels[:-1], color_levels[1:]):
        cbar_ax.add_patch(Rectangle(
            (0, lower), 1, upper - lower,
            facecolor=cmap(heat_norm((lower + upper) / 2)),
            edgecolor="none",
        ))
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks(np.linspace(heat_norm.vmin, heat_norm.vmax, 4))
    cbar_ax.set_ylabel(r"Paired $\Delta$ directed $\mathrm{F1}$", labelpad=8)
    cbar_ax.spines["top"].set_visible(False)
    cbar_ax.spines["right"].set_visible(True)
    cbar_ax.spines["left"].set_visible(True)
    cbar_ax.spines["bottom"].set_visible(False)

    for ax in (ax_a, ax_c):
        ax.grid(axis="y", color="#D7DBDD", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6)
    finalize_figure(fig, "figure2_controlled_validation")


def gate_sensitivity_figure(payload: dict) -> None:
    gate = payload["experiments"]["exp7_gate_sensitivity"]
    validity = payload["experiments"]["exp11_finite_sample_gate_audit"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.25), sharex=False)
    colors = {0.55: PALETTE["red"], 0.75: PALETTE["teal"], 0.90: PALETTE["blue"]}
    markers = {0.55: "o", 0.75: "s", 0.90: "D"}
    for ax, regime in zip(axes, ("informative", "uninformative")):
        for accuracy in (0.55, 0.75, 0.90):
            rows = sorted(
                [row for row in gate["settings"]
                 if row["confidence_regime"] == regime
                 and float(row["source_accuracy"]) == accuracy],
                key=lambda row: row["n_calibration"],
            )
            x = [row["n_calibration"] for row in rows]
            certificate = [row["certificate_rate"] for row in rows]
            coverage = [row["mean_deployment_coverage_when_certified"] for row in rows]
            ax.plot(x, certificate, color=colors[accuracy], marker=markers[accuracy],
                    markeredgecolor=PALETTE["ink"], markersize=6.5,
                    linewidth=2, label="_nolegend_")
            ax.plot(x, coverage, color=colors[accuracy], marker=markers[accuracy],
                    markeredgecolor=PALETTE["ink"], markersize=6.5,
                    linewidth=1.5, linestyle="--", alpha=0.82,
                    label="_nolegend_")
        ax.set_xticks([100, 300, 600, 1200], [100, 300, 600, 1200])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Calibration proposals")
        ax.grid(axis="y", color="#D7DBDD", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("a  Informative confidence")
    axes[1].set_title("b  Uninformative confidence")
    accuracy_handles = [
        Line2D([0], [0], color=PALETTE["ink"], marker=markers[accuracy],
               markerfacecolor=PALETTE["white"], markeredgecolor=PALETTE["ink"],
               linestyle="None", markersize=6.5, label=f"accuracy {accuracy:.2f}")
        for accuracy in (0.55, 0.75, 0.90)
    ]
    outcome_handles = [
        Line2D([0], [0], color=PALETTE["ink"], linewidth=2, label="certificate"),
        Line2D([0], [0], color=PALETTE["ink"], linewidth=1.5, linestyle="--",
               label="coverage conditional on certificate"),
    ]
    accuracy_legend = fig.legend(
        handles=accuracy_handles, title="Marker: source accuracy", loc="lower left",
        ncol=3, bbox_to_anchor=(0.055, -0.075), fontsize=9.0, title_fontsize=9.0,
        frameon=True, fancybox=False, edgecolor="#8A9499", facecolor=PALETTE["white"],
    )
    fig.add_artist(accuracy_legend)
    fig.legend(
        handles=outcome_handles, title="Line: outcome", loc="lower right",
        ncol=2, bbox_to_anchor=(0.955, -0.075), fontsize=9.0, title_fontsize=9.0,
        frameon=True, fancybox=False, edgecolor="#8A9499", facecolor=PALETTE["white"],
    )
    # Directly audit the theorem's operational target using the locked
    # repeated-calibration/deployment experiment.
    ax = axes[2]
    accuracy_colors = {0.55: PALETTE["red"], 0.75: PALETTE["teal"], 0.90: PALETTE["blue"]}
    accuracy_markers = {0.55: "o", 0.75: "s", 0.90: "D"}
    rows = [row for row in validity["settings"]
            if row["confidence_regime"] == "informative"]
    for accuracy in (0.55, 0.75, 0.90):
        selected = sorted(
            [row for row in rows if float(row["source_accuracy"]) == accuracy],
            key=lambda row: row["n_calibration"],
        )
        x = [row["n_calibration"] for row in selected]
        risk_violation = [row["risk_controlled"]["target_violation_rate"] for row in selected]
        empirical_violation = [row["empirical_threshold"]["target_violation_rate"] for row in selected]
        color = accuracy_colors[accuracy]
        marker = accuracy_markers[accuracy]
        ax.plot(x, risk_violation, color=color, marker=marker, linewidth=2.0,
                markersize=6.2, markeredgecolor=PALETTE["ink"],
                label="_nolegend_")
        ax.plot(x, empirical_violation, color=color, marker=marker, linewidth=1.6,
                linestyle="--", markersize=6.2, markerfacecolor=PALETTE["white"],
                markeredgecolor=PALETTE["ink"],
                label="_nolegend_")
    ax.axhline(validity["protocol"]["failure_probability"], color=PALETTE["ink"],
               linewidth=1.3, linestyle=":", label="_nolegend_")
    ax.set_xticks([60, 100, 300], [60, 100, 300])
    ax.set_ylim(-0.02, 0.48)
    ax.set_xlabel("Calibration proposals")
    ax.set_ylabel(r"Runs with deployment error $>\alpha$")
    ax.set_title(r"c  Empirical deployment-error audit ($\alpha=0.20$)")
    ax.grid(axis="y", color="#D7DBDD", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    method_handles = [
        Line2D([0], [0], color=PALETTE["ink"], linewidth=2.0,
               label="RCEP (solid)"),
        Line2D([0], [0], color=PALETTE["ink"], linewidth=1.6, linestyle="--",
               label="Empirical (dashed)"),
        Line2D([0], [0], color=PALETTE["ink"], linewidth=1.3, linestyle=":",
               label=r"Nominal $\delta=0.05$"),
    ]
    fig.legend(
        handles=method_handles, title="Panel c line: method", loc="lower center",
        ncol=1, bbox_to_anchor=(0.50, -0.075), fontsize=8.5, title_fontsize=8.5,
        frameon=True, fancybox=False, edgecolor="#8A9499", facecolor=PALETTE["white"],
    )
    fig.tight_layout(rect=[0, 0.18, 1, 1], pad=1.4)
    finalize_figure(fig, "figure3_gate_sensitivity")


def external_transfer_figure(payload: dict) -> None:
    exp8 = payload["experiments"]["exp8_external_prior_transfer"]
    overlap = exp8["prior_reference_overlap"]
    deltas = np.array([row["delta_f1"] for row in exp8["replicates"]])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    categories = ["Same direction", "Reversed", "Nonadjacent"]
    counts = [
        overlap["correct_reference_direction"],
        overlap["reversed_reference_direction"],
        overlap["nonadjacent_to_reference"],
    ]
    bars = axes[0].bar(categories, counts,
                       color=[PALETTE["green"], PALETTE["red"], PALETTE["gray"]],
                       edgecolor="black", linewidth=0.8)
    axes[0].bar_label(bars, padding=3, fontsize=10)
    axes[0].set_ylabel("Frozen OmniPath constraints")
    axes[0].set_ylim(0, max(counts) + 2)
    axes[0].set_title("a  External consensus is not benchmark truth")

    ordered = np.sort(deltas)
    cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
    mean = float(np.mean(deltas))
    lower, upper = exp8["summary"]["delta_f1_percentile_interval"]
    positive = exp8["summary"]["probability_delta_f1_positive"]
    axes[1].plot(ordered, cumulative, color=PALETTE["blue"], linewidth=2.2,
                 label="Empirical CDF")
    axes[1].axvline(0, color=PALETTE["ink"], linewidth=2.0, linestyle="--",
                    label="zero gain")
    axes[1].axvline(mean, color=PALETTE["red"], linewidth=2,
                    label=f"mean {mean:+.3f}")
    axes[1].axvspan(lower, upper, color=PALETTE["red_light"], alpha=0.55,
                    label=f"95% percentile [{lower:+.3f}, {upper:+.3f}]")
    axes[1].text(0.97, 0.08, rf"$P(\Delta F1>0)={positive:.2f}$",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 color=PALETTE["ink"])
    axes[1].set_xlabel(r"Bootstrap $\Delta$ directed F1")
    axes[1].set_ylabel("Cumulative fraction of resamples")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("b  Uncalibrated transfer does not improve recovery")
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.grid(axis="y", color="#D7DBDD", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.5)
    finalize_figure(fig, "figure5_external_transfer")


def design_transfer_and_gate_audit_figure(payload: dict) -> None:
    """Compare within-study and cross-study design-direction evidence."""
    within_study = payload["experiments"]["exp9_psychology_design_validation"]["summary"]
    cross_study = payload["experiments"]["exp12_cross_study_design_transfer"]["summary"]
    gate = payload["experiments"]["exp11_finite_sample_gate_audit"]["aggregate"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    studies = ["JOBS II\nwithin study", "Project STAR\ncross study"]
    x = np.arange(len(studies))
    width = 0.34
    edge = {"edgecolor": PALETTE["ink"], "linewidth": 0.8}

    data_coverage = [
        within_study["data_only_design_consistent_coverage"],
        cross_study["data_only_design_consistent_coverage"],
    ]
    controlled_coverage = [
        within_study["risk_controlled_design_consistent_coverage"],
        cross_study["risk_controlled_design_consistent_coverage"],
    ]
    data_bars = axes[0].bar(
        x - width / 2, data_coverage, width, color=PALETTE["gray_light"],
        hatch="///", label="Data only", **edge,
    )
    controlled_bars = axes[0].bar(
        x + width / 2, controlled_coverage, width, color=PALETTE["blue"],
        label="RCEP", **edge,
    )
    axes[0].bar_label(data_bars, fmt="%.3f", padding=3, fontsize=10)
    axes[0].bar_label(controlled_bars, fmt="%.3f", padding=3, fontsize=10)
    lower, upper = cross_study["delta_design_consistent_coverage_interval"]
    axes[0].plot([0.82, 1.18], [0.84, 0.84], color=PALETTE["ink"], linewidth=0.9)
    axes[0].text(
        1.0, 0.86, rf"STAR $\Delta=+{cross_study['mean_delta_design_consistent_coverage']:.3f}$"
        + "\n" + rf"95% CI [{lower:.3f}, {upper:.3f}]",
        ha="center", va="bottom", fontsize=9.5, color=PALETTE["ink"],
    )
    axes[0].set_xticks(x, studies)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Fraction of supported pairs")
    axes[0].set_title("a  Design-consistent direction")
    axes[0].legend(loc="lower left")

    data_reverse = [
        within_study["data_only_reverse_time_rate"],
        cross_study["data_only_reverse_time_rate"],
    ]
    controlled_reverse = [
        within_study["risk_controlled_reverse_time_rate"],
        cross_study["risk_controlled_reverse_time_rate"],
    ]
    data_bars = axes[1].bar(
        x - width / 2, data_reverse, width, color=PALETTE["gray_light"],
        hatch="///", **edge,
    )
    controlled_bars = axes[1].bar(
        x + width / 2, controlled_reverse, width, color=PALETTE["blue"], **edge,
    )
    axes[1].bar_label(data_bars, fmt="%.3f", padding=3, fontsize=10)
    axes[1].bar_label(controlled_bars, fmt="%.3f", padding=3, fontsize=10)
    axes[1].set_xticks(x, studies)
    axes[1].set_ylim(0, 0.68)
    axes[1].set_ylabel("Fraction of supported pairs")
    axes[1].set_title("b  Reverse-time direction")

    methods = ["RCEP\ncertificate", "Empirical\nthreshold"]
    violations = [
        gate["max_risk_controlled_target_violation_rate"],
        gate["max_empirical_target_violation_rate"],
    ]
    point_x = np.arange(2)
    axes[2].vlines(point_x, 0, violations, colors=[PALETTE["blue"], PALETTE["red"]],
                   linewidth=2.0, zorder=2)
    axes[2].scatter(point_x, violations, s=85,
                    color=[PALETTE["blue"], PALETTE["red"]],
                    edgecolor=PALETTE["ink"], linewidth=0.8, zorder=3)
    for position, value in zip(point_x, violations):
        axes[2].annotate(f"{value:.2f}", (position, value), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=10)
    axes[2].axhline(0.20, color=PALETTE["ink"], linewidth=1.0, linestyle="--",
                    label="Target error 0.20")
    axes[2].set_xticks(point_x, methods)
    axes[2].set_xlim(-0.55, 1.55)
    axes[2].set_ylim(-0.035, 0.52)
    axes[2].set_ylabel("Maximum deployment violation rate")
    axes[2].set_title("c  Independent gate audit")
    axes[2].legend(loc="upper left")

    for ax in axes:
        ax.grid(axis="y", color="#D7DBDD", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.4)
    finalize_figure(fig, "figure4_design_transfer_and_gate_audit")


def causalbench_intervention_figure(payload: dict) -> None:
    """Report the independent perturbation audit without hiding abstention."""
    exp16 = payload["experiments"]["exp16_causalbench_intervention_holdout"]
    protocol = exp16["protocol"]
    summary = exp16["summary"]
    direction = summary["directional_positive_labels"]
    support = summary["held_out_support"]
    accepted = summary["held_out_accepted_support"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.85))
    edge = {"edgecolor": PALETTE["ink"], "linewidth": 0.8}

    groups = ["Calibration", "Held-out"]
    candidates = [protocol["n_calibration_candidates"], protocol["n_held_out_candidates"]]
    positives = [
        int(exp16["source_metadata"]["n_positive_labels"]) - protocol["n_held_out_positive_labels"],
        protocol["n_held_out_positive_labels"],
    ]
    x = np.arange(2)
    width = 0.35
    bars = axes[0].bar(x - width / 2, candidates, width, color=PALETTE["gray_light"],
                       hatch="///", label="ChIP candidates", **edge)
    pos = axes[0].bar(x + width / 2, positives, width, color=PALETTE["green"],
                      label="Intervention-confirmed", **edge)
    axes[0].bar_label(bars, padding=2, fontsize=9.5)
    axes[0].bar_label(pos, padding=2, fontsize=9.5)
    axes[0].set_xticks(x, groups)
    axes[0].set_ylabel("Directed TF-target proposals")
    axes[0].set_title("a  TF-disjoint source split")
    axes[0].legend(loc="upper right", fontsize=9.2)

    precision = support["support_precision"]
    coverage = accepted["coverage"]
    bars = axes[1].bar(
        [0, 1], [precision, coverage],
        color=[PALETTE["red"], PALETTE["blue"]], **edge,
    )
    axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    axes[1].set_xticks([0, 1], ["Unfiltered source\npositive-label fraction", "RCEP accepted\ncoverage"])
    axes[1].set_ylim(0, 0.12)
    axes[1].set_ylabel("Held-out fraction")
    axes[1].set_title("b  Certificate rejects sparse labels")
    axes[1].text(0.5, 0.108, "No certified threshold", ha="center", va="top",
                 color=PALETTE["red"], fontsize=10, fontweight="bold")

    values = [direction["data_only_accuracy"], direction["soft_prior_accuracy"], direction["rcep_accuracy"]]
    bars = axes[2].bar(
        [0, 1, 2], values,
        color=[PALETTE["gray"], PALETTE["violet"], PALETTE["blue"]], **edge,
    )
    axes[2].bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    axes[2].set_xticks([0, 1, 2], ["Data only", "Uncalibrated\nsource", "RCEP"])
    axes[2].set_ylim(0, 1.12)
    axes[2].set_ylabel("Accuracy on intervention-confirmed positives (n=15)")
    axes[2].set_title("c  Intervention audit")
    for ax in axes:
        ax.grid(axis="y", color="#D7DBDD", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.3)
    finalize_figure(fig, "figure6_causalbench_intervention_audit")


def causalbench_source_variant_figure(payload: dict) -> None:
    """Show the fixed independent-source construction sensitivity audit."""
    exp17 = payload["experiments"]["exp17_causalbench_source_variants"]
    names = list(exp17["variants"])
    labels = ["K562", "K562+HepG2", "K562+DoRothEA", "Both"]
    variants = exp17["variants"]
    candidates = [variants[name]["n_held_out_candidates"] for name in names]
    precision = [variants[name]["held_out_support"]["support_precision"] for name in names]
    coverage = [variants[name]["held_out_accepted_support"]["coverage"] for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.75))
    edge = {"edgecolor": PALETTE["ink"], "linewidth": 0.8}
    x = np.arange(len(labels))
    bars = axes[0].bar(x, candidates, color=PALETTE["gray_light"], **edge)
    axes[0].bar_label(bars, padding=2, fontsize=9.5)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Held-out proposals")
    axes[0].set_title("a  Population after source filtering")
    bars = axes[1].bar(x, [value if value is not None else 0 for value in precision], color=PALETTE["red"], **edge)
    axes[1].bar_label(bars, labels=["%.3f" % value if value is not None else "no labels" for value in precision], padding=2, fontsize=9.5)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 0.12)
    axes[1].set_ylabel("Held-out positive-label fraction")
    axes[1].set_title("b  Replication does not fix label sparsity")
    bars = axes[2].bar(x, coverage, color=PALETTE["blue"], **edge)
    axes[2].bar_label(bars, fmt="%.2f", padding=2, fontsize=9.5)
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Certified accepted coverage")
    axes[2].set_title("c  All variants abstain")
    for ax in axes:
        ax.grid(axis="y", color="#D7DBDD", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.3)
    finalize_figure(fig, "figure7_causalbench_source_variants")


def stratified_shift_figure(payload: dict) -> None:
    """Visualize pooled versus worst-stratum behavior under mixture shift."""
    summary = payload["experiments"]["exp18_stratified_shift_audit"]["summary"]
    keys = [
        "pooled_certificate", "empirical_pooled",
        "empirical_stratified", "stratified_certificate",
    ]
    labels = ["Pooled CP", "Empirical\npooled", "Empirical\nstratified", "Worst-stratum"]
    colors = [PALETTE["gray"], PALETTE["red"], PALETTE["violet"], PALETTE["blue"]]
    coverage = [summary[key]["mean_accepted_coverage"] for key in keys]
    errors = [summary[key]["mean_deployment_error"] for key in keys]
    violations = [summary[key]["target_violation_fraction"] for key in keys]
    coverage_interval = [summary[key]["accepted_coverage_95_interval"] for key in keys]
    error_interval = [summary[key]["deployment_error_95_interval"] for key in keys]
    coverage_yerr = np.asarray([
        [value - interval[0] for value, interval in zip(coverage, coverage_interval)],
        [interval[1] - value for value, interval in zip(coverage, coverage_interval)],
    ])
    error_yerr = np.asarray([
        [value - interval[0] for value, interval in zip(errors, error_interval)],
        [interval[1] - value for value, interval in zip(errors, error_interval)],
    ])
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7))
    edge = {"edgecolor": PALETTE["ink"], "linewidth": 0.8}
    x = np.arange(4)
    bars = axes[0].bar(x, coverage, color=colors, yerr=coverage_yerr, capsize=3, **edge)
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5)
    axes[0].set_xticks(x, labels, rotation=0, ha="center", fontsize=9.2)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Accepted deployment coverage")
    axes[0].set_title("a  Selective coverage")
    bars = axes[1].bar(x, errors, color=colors, yerr=error_yerr, capsize=3, **edge)
    axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5)
    axes[1].axhline(0.20, color=PALETTE["ink"], linestyle="--", linewidth=1.9,
                    label="Target 0.20", zorder=4)
    axes[1].set_xticks(x, labels, rotation=0, ha="center", fontsize=9.2)
    axes[1].set_ylim(0, 0.31)
    axes[1].set_ylabel("Deployment accepted error")
    axes[1].set_title("b  Shifted-mixture error")
    axes[1].legend(loc="upper right")
    bars = axes[2].bar(x, violations, color=colors, **edge)
    axes[2].bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
    axes[2].set_xticks(x, labels, rotation=0, ha="center", fontsize=9.2)
    axes[2].set_ylim(0, 1.12)
    axes[2].set_ylabel("Target-violation fraction")
    axes[2].set_title("c  200 independent runs")
    for ax in axes:
        ax.grid(axis="y", color="#D7DBDD", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.25)
    finalize_figure(fig, "figure8_stratified_shift_audit")


def cross_cell_context_figure(payload: dict) -> None:
    exp = payload["experiments"]["exp23_cross_cell_context_representation"]
    primary = exp["primary"]
    test = primary["test"]
    certificate = primary["gate_certificate"]
    runs = exp["split_sensitivity"]["runs"]
    certified = [
        row for row in runs
        if row["gate_certificate"]["status"] == "certified"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15))
    edge = dict(edgecolor=PALETTE["ink"], linewidth=0.8)
    bars = axes[0].bar(
        [0, 1],
        [test["accepted_precision"], test["coverage"]],
        color=[PALETTE["green"], PALETTE["blue"]],
        **edge,
    )
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5)
    axes[0].set_xticks([0, 1], ["Accepted\nprecision", "Accepted\ncoverage"])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Held-out fraction")
    axes[0].set_title("a  Independent RPE1 test")

    rows = certificate["threshold_diagnostics"]
    thresholds = np.asarray([row["threshold"] for row in rows], dtype=float)
    empirical = np.asarray([row["empirical_error"] for row in rows], dtype=float)
    upper = np.asarray([row["simultaneous_error_upper"] for row in rows], dtype=float)
    axes[1].plot(thresholds, empirical, color=PALETTE["blue"], linewidth=1.8,
                 label="Empirical error")
    axes[1].plot(thresholds, upper, color=PALETTE["violet"], linewidth=1.8,
                 label="Simultaneous upper")
    axes[1].axhline(0.08, color=PALETTE["red"], linestyle="--", linewidth=1.1,
                    label="Target 0.08")
    selected = certificate["selected"]
    axes[1].scatter(
        [selected["threshold"]], [selected["simultaneous_error_upper"]],
        color=PALETTE["green"], edgecolor=PALETTE["ink"], linewidth=0.7,
        s=42, zorder=4, label="Selected",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 0.24)
    axes[1].set_xlabel("Context-score threshold")
    axes[1].set_ylabel("Calibration error")
    axes[1].set_title("b  Risk certificate")
    axes[1].legend(loc="upper right", fontsize=8.5)

    coverage = [row["test"]["coverage"] for row in certified]
    precision = [row["test"]["accepted_precision"] for row in certified]
    axes[2].scatter(
        coverage, precision, color=PALETTE["teal"], alpha=0.82,
        edgecolor=PALETTE["ink"], linewidth=0.55, s=34,
    )
    axes[2].axhline(0.92, color=PALETTE["red"], linestyle="--", linewidth=1.1,
                    label="Precision target 0.92")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0.89, 1.005)
    axes[2].set_xlabel("Accepted coverage")
    axes[2].set_ylabel("Accepted precision")
    axes[2].set_title("c  20 gene splits")
    axes[2].text(
        0.03, 0.015,
        f"{len(certified)}/20 certified; 0 violations",
        transform=axes[2].transAxes, fontsize=8.8, fontweight="bold",
        color=PALETTE["ink"],
    )
    axes[2].legend(loc="lower right", fontsize=8.5)
    for ax in axes:
        ax.grid(axis="y", color="#D7DBDD", linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.2)
    finalize_figure(fig, "figure9_cross_cell_context")


def main() -> None:
    apply_publication_style()
    payload = load_locked_results()
    architecture_figure()
    main_results_figure(payload)
    gate_sensitivity_figure(payload)
    external_transfer_figure(payload)
    design_transfer_and_gate_audit_figure(payload)
    causalbench_intervention_figure(payload)
    causalbench_source_variant_figure(payload)
    stratified_shift_figure(payload)
    cross_cell_context_figure(payload)


if __name__ == "__main__":
    main()
