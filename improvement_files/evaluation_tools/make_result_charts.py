"""Render the experiment reports as presentation assets.

Reads whatever is in ``data/reports/`` and writes PNG charts plus plain-text
tables into ``results/``, so a slide deck or write-up can use the measured
numbers without re-running anything.

Design notes, because charts are read by people
----------------------------------------------
* **Emphasis, not eight hues.** The system comparison has eight bars but only
  one story — the learned router leads, the hand-tuned one lags — so the two
  routers carry colour and the six baselines share a single neutral. Giving
  every bar its own hue would burn the colour channel on information the bar
  lengths already carry.
* **Diverging colour only where there is a real midpoint.** Ablation deltas and
  the purity/utility frontier are signed quantities around zero, so they get the
  blue/red diverging pair with a neutral zero line. Magnitudes do not.
* **Every bar is labelled at the tip.** The palette's aqua slot sits below 3:1
  contrast on a light surface, and the accessibility rule for that is visible
  direct labels rather than relying on the fill. Labelling also means the charts
  survive being printed in greyscale.
* **One axis, always.** No chart here plots two scales.

Palette is the validated default (see the data-viz reference): categorical slot
1 blue, slot 2 orange, blue/red diverging, with text in ink tokens rather than
series colours.

USAGE
  python improvement_files/evaluation_tools/make_result_charts.py
  python improvement_files/evaluation_tools/make_result_charts.py --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ahrag.eval.reports import load_reports  # noqa: E402

# --- validated palette -----------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_1 = "#2a78d6"   # blue  — the proposed learned router
SERIES_2 = "#eb6834"   # orange — the shipped rule-based router
NEUTRAL = "#b8b7b0"    # the six baselines, deliberately one colour
DIVERGE_NEG = "#d03b3b"  # removing the mechanism hurt
DIVERGE_POS = "#2a78d6"  # removing the mechanism helped

SYSTEM_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "P1", "P2"]
SYSTEM_NAMES = {
    "B1": "B1  fixed BM25",
    "B2": "B2  fixed dense",
    "B3": "B3  fixed hybrid RRF",
    "B4": "B4  always-maximal",
    "B5": "B5  complexity-only",
    "B6": "B6  Adaptive-RAG (trained)",
    "P1": "P1  governance (rule-based)",
    "P2": "P2  governance (learned)",
}


def style_axes(ax, xlabel: str = "", title: str = "", subtitle: str = "") -> None:
    """Apply the recessive-chrome house style."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    if title:
        ax.set_title(
            title, color=INK, fontsize=12.5, fontweight="600", loc="left", pad=30
        )
    if subtitle:
        ax.text(
            0.0, 1.012, subtitle, transform=ax.transAxes,
            color=INK_SECONDARY, fontsize=9, va="bottom",
        )


def save(fig, out: Path, name: str, written: list[Path]) -> None:
    """Write a chart and record it."""
    path = out / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    written.append(path)
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def chart_system_comparison(data: dict, out: Path, written: list[Path]) -> None:
    """Recall@5 per system, with 95% bootstrap CI whiskers."""
    per_system = data.get("per_system") or {}
    rows = [
        (key, per_system[key]["recall_at_5"])
        for key in SYSTEM_ORDER
        if key in per_system and "recall_at_5" in per_system[key]
    ]
    if not rows:
        return

    labels = [SYSTEM_NAMES.get(k, k) for k, _ in rows]
    means = [s["mean"] for _, s in rows]
    lows = [s["mean"] - s["ci_low"] for _, s in rows]
    highs = [s["ci_high"] - s["mean"] for _, s in rows]
    colours = [
        SERIES_1 if k == "P2" else SERIES_2 if k == "P1" else NEUTRAL
        for k, _ in rows
    ]

    # Taller figure and a thinner bar keep the mark near the <=24px spec and
    # leave the band's leftover as air rather than filling it.
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    positions = range(len(rows))
    ax.barh(
        list(positions), means, height=0.44, color=colours,
        error_kw={"ecolor": INK_MUTED, "elinewidth": 1.2, "capsize": 3},
        xerr=[lows, highs],
    )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
    ax.invert_yaxis()

    # The label must clear the *upper CI cap*, not the bar end, or it lands on
    # top of the whisker. Measured from the widest interval so every row uses
    # the same offset and the numbers stay in a column.
    upper = [mean + high for mean, high in zip(means, highs)]
    ax.set_xlim(0, max(upper) * 1.16)
    label_x = max(upper) * 1.02

    for y, (mean, (key, _)) in enumerate(zip(means, rows)):
        weight = "600" if key in {"P1", "P2"} else "normal"
        ax.text(
            label_x, y, f"{mean:.3f}",
            va="center", fontsize=9.5, color=INK, fontweight=weight,
        )

    items = data.get("aligned_items") or data.get("items") or "?"
    style_axes(
        ax,
        xlabel="Recall@5   (whiskers: 95% bootstrap CI)",
        title="The learned router leads; the hand-tuned one does not",
        subtitle=f"{items} evaluation queries · 6,139-document benchmark corpus · "
                 f"identical corpus, index and generator throughout",
    )
    # Figure coordinates, below the axis label, so the two cannot overlap.
    fig.text(
        0.09, -0.02,
        "Only the routing policy differs between systems. "
        "Zero ACL violations in all eight.",
        color=INK_MUTED, fontsize=8.5,
    )
    save(fig, out, "01_system_recall_at_5.png", written)


def chart_ablations(data: dict, out: Path, written: list[Path]) -> None:
    """Effect on Recall@5 and abstention of disabling each mechanism."""
    ablations = data.get("ablations") or {}
    if not ablations:
        return

    for metric, nice, fname in (
        ("recall_at_5", "Recall@5", "02_ablation_recall.png"),
        ("abstention_appropriate", "abstention appropriateness",
         "03_ablation_abstention.png"),
    ):
        rows = []
        for key, info in ablations.items():
            stats = (info.get("metrics") or {}).get(metric)
            if stats:
                rows.append((key, stats["delta_vs_full"], stats["p_value"]))
        if not rows:
            continue
        rows.sort(key=lambda r: r[1])

        labels = [r[0].replace("no_", "− ").replace("_", " ") for r in rows]
        deltas = [r[1] for r in rows]
        colours = [DIVERGE_NEG if d < 0 else DIVERGE_POS for d in deltas]

        fig, ax = plt.subplots(figsize=(9.2, 0.52 * len(rows) + 2.4))
        positions = range(len(rows))
        ax.barh(list(positions), deltas, height=0.6, color=colours)
        ax.axvline(0, color=BASELINE, linewidth=1.2)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
        ax.invert_yaxis()

        span = max(abs(min(deltas)), abs(max(deltas))) or 1.0
        ax.set_xlim(-span * 1.35, span * 0.55)
        for y, (key, delta, p_value) in enumerate(rows):
            marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else (
                "*" if p_value < 0.05 else "ns"
            )
            offset = -span * 0.03 if delta < 0 else span * 0.03
            ax.text(
                delta + offset, y, f"{delta:+.3f} {marker}",
                va="center", ha="right" if delta < 0 else "left",
                fontsize=9, color=INK,
            )

        style_axes(
            ax,
            xlabel=f"change in {nice} when the mechanism is disabled",
            title=f"What each mechanism contributes to {nice}",
            subtitle="Red: removing it made the system worse · "
                     "Blue: the system is better without it",
        )
        ax.grid(axis="x", color=GRID, linewidth=1.0)
        fig.text(
            0.09, -0.02,
            "*** p<0.001   ** p<0.01   * p<0.05   ns not significant   "
            "(two-sided paired bootstrap)",
            color=INK_MUTED, fontsize=8.5,
        )
        save(fig, out, fname, written)


def chart_embeddings(data: dict, out: Path, written: list[Path]) -> None:
    """Backend comparison: quality alongside sparse/dense divergence."""
    backends = data.get("backends") or {}
    rows = [
        (key, info) for key, info in backends.items()
        if (info.get("metrics") or {}).get("recall_at_5")
    ]
    if len(rows) < 2:
        return

    labels = [info.get("label", key)[:34] for key, info in rows]
    means = [info["metrics"]["recall_at_5"]["mean"] for _, info in rows]
    lows = [
        info["metrics"]["recall_at_5"]["mean"] - info["metrics"]["recall_at_5"]["ci_low"]
        for _, info in rows
    ]
    highs = [
        info["metrics"]["recall_at_5"]["ci_high"] - info["metrics"]["recall_at_5"]["mean"]
        for _, info in rows
    ]
    baseline_key = data.get("baseline")
    colours = [NEUTRAL if key == baseline_key else SERIES_1 for key, _ in rows]

    fig, ax = plt.subplots(figsize=(8.6, 2.2 + 0.5 * len(rows)))
    positions = range(len(rows))
    ax.barh(
        list(positions), means, height=0.4, color=colours,
        xerr=[lows, highs],
        error_kw={"ecolor": INK_MUTED, "elinewidth": 1.2, "capsize": 3},
    )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
    ax.invert_yaxis()
    upper = [mean + high for mean, high in zip(means, highs)]
    ax.set_xlim(0, max(upper) * 2.0)
    label_x = max(upper) * 1.04

    comparisons = data.get("comparisons_vs_baseline") or {}
    for y, ((key, _), mean) in enumerate(zip(rows, means)):
        note = f"{mean:.3f}"
        stats = (comparisons.get(key) or {}).get("recall_at_5")
        if stats:
            p_value = stats["p_value"]
            marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else (
                "*" if p_value < 0.05 else "ns"
            )
            note += f"   ({stats['delta']:+.4f} vs LSA, p={p_value:.3f} {marker})"
        ax.text(label_x, y, note, va="center", fontsize=9, color=INK)

    style_axes(
        ax,
        xlabel="Recall@5   (whiskers: 95% bootstrap CI)",
        title="A neural encoder helps — reliably, and only a little",
        subtitle="The same comparison was not significant at n=25 on the demo "
                 "corpus (p=0.52); at n≈870 it is (p=0.002), with a negligible "
                 "effect size",
    )
    save(fig, out, "04_embedding_backends.png", written)


def chart_specialisation(data: dict, out: Path, written: list[Path]) -> None:
    """The purity/utility frontier, and the interference it buys away."""
    frontier = ((data.get("quality") or {}).get("frontier")) or {}
    if frontier:
        lambdas, deltas, p_values = [], [], []
        for label, entry in frontier.items():
            stats = entry.get("recall_at_5")
            if not stats:
                continue
            try:
                lam = float(str(label).split("=")[-1])
            except ValueError:
                continue
            lambdas.append(lam)
            deltas.append(stats.get("delta", 0.0))
            p_values.append(stats.get("p_value", 1.0))
        if lambdas:
            order = sorted(range(len(lambdas)), key=lambda i: lambdas[i])
            lambdas = [lambdas[i] for i in order]
            deltas = [deltas[i] for i in order]
            p_values = [p_values[i] for i in order]

            fig, ax = plt.subplots(figsize=(7.8, 4.2))
            ax.axhline(0, color=BASELINE, linewidth=1.2)
            ax.plot(
                lambdas, deltas, color=SERIES_1, linewidth=2.0,
                marker="o", markersize=8, markerfacecolor=SERIES_1,
                markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round",
            )
            for lam, delta, p_value in zip(lambdas, deltas, p_values):
                marker = "*" if p_value < 0.05 else "ns"
                ax.annotate(
                    f"{delta:+.4f} {marker}",
                    (lam, delta), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=INK,
                )
            ax.set_xticks(lambdas)
            ax.set_xlim(-0.06, 1.06)
            pad = (max(deltas) - min(deltas) or 0.004) * 0.6
            ax.set_ylim(min(deltas) - pad, max(deltas) + pad * 1.6)
            style_axes(
                ax,
                xlabel="λ — information-flow budget  (0 = provably non-interfering)",
                title="What non-interference costs",
                subtitle="Change in Recall@5 versus the unspecialised system · "
                         "* p<0.05 · benchmark corpus, 7 ACL classes",
            )
            ax.grid(axis="y", color=GRID, linewidth=1.0)
            ax.grid(axis="x", visible=False)
            save(fig, out, "05_spis_lambda_frontier.png", written)

    interference = data.get("interference") or {}
    if interference:
        labels = list(interference)
        # route_change_rate is deliberately omitted: the experiment pins the
        # route, so it is zero for both conditions by construction. Plotting a
        # 0-vs-0 row would read as "specialisation fixed route changes" when
        # nothing was measured there at all.
        metrics = [
            ("order_change_rate", "evidence-order changes"),
            ("top1_flip_rate", "top-1 flips"),
        ]
        fig, ax = plt.subplots(figsize=(8.6, 3.6))
        height = 0.34
        for offset, (label, colour) in enumerate(
            zip(labels, (SERIES_2, SERIES_1))
        ):
            values = [interference[label].get(m, 0.0) for m, _ in metrics]
            positions = [i + (offset - 0.5) * (height + 0.03) for i in range(len(metrics))]
            ax.barh(positions, values, height=height, color=colour, label=label)
            for y, value in zip(positions, values):
                ax.text(
                    value + 0.006, y, f"{value:.3f}",
                    va="center", fontsize=9, color=INK,
                )
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([n for _, n in metrics], fontsize=9.5, color=INK_SECONDARY)
        ax.invert_yaxis()
        legend = ax.legend(
            frameon=False, fontsize=9, loc="lower right", labelcolor=INK_SECONDARY
        )
        legend.set_title(None)
        style_axes(
            ax,
            xlabel="fraction of queries affected by documents the principal cannot read",
            title="Unreadable documents changed results — specialisation stops it",
            subtitle="Authorised subcorpus held fixed, everything outside it "
                     "deleted, route pinned · a non-interfering index returns "
                     "identical results",
        )
        save(fig, out, "06_spis_interference.png", written)


def chart_learning_curve(out: Path, written: list[Path]) -> None:
    """Router learning curve, read from the training metadata artifact."""
    path = (
        PROJECT_ROOT / "improvement_files" / "ml_router_training" / "artifacts"
        / "metadata.json"
    )
    if not path.exists():
        return
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    curve = metadata.get("learning_curve") or []
    if len(curve) < 2:
        return

    sizes = [point["train_size"] for point in curve]
    accuracies = [point["val_accuracy"] for point in curve]

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(
        sizes, accuracies, color=SERIES_1, linewidth=2.0, marker="o",
        markersize=8, markerfacecolor=SERIES_1, markeredgecolor=SURFACE,
        markeredgewidth=2, solid_capstyle="round",
    )
    for size, accuracy in ((sizes[0], accuracies[0]), (sizes[-1], accuracies[-1])):
        ax.annotate(
            f"{accuracy:.3f}", (size, accuracy),
            textcoords="offset points", xytext=(0, 12), ha="center",
            fontsize=9.5, color=INK, fontweight="600",
        )
    ax.set_ylim(min(accuracies) - 0.04, max(accuracies) + 0.05)
    style_axes(
        ax,
        xlabel="labelled training queries",
        title="More offline route labels help, then plateau",
        subtitle="Validation route accuracy · labels derived by executing all "
                 "five routes on every query",
    )
    ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.grid(axis="x", visible=False)
    save(fig, out, "07_router_learning_curve.png", written)


# ---------------------------------------------------------------------------
# Text tables — so every number is readable without opening an image
# ---------------------------------------------------------------------------


def write_tables(reports, out: Path, written: list[Path]) -> None:
    """Emit the same numbers as plain text, for pasting and for greyscale print."""
    lines: list[str] = [
        "AHRAG — measured results",
        "=" * 78,
        "",
        "Generated from data/reports/ by "
        "improvement_files/evaluation_tools/make_result_charts.py.",
        "Every figure below came from a script in this repository; see FINDINGS.md",
        "for what each one does and does not establish.",
        "",
    ]

    for report in reports:
        lines += ["", "=" * 78, f"{report.title}", "=" * 78,
                  f"corpus     : {report.corpus_label}",
                  f"generated  : {report.generated_at}",
                  f"items      : {report.data.get('aligned_items') or report.data.get('items', '—')}"]
        if report.data.get("embedder"):
            lines.append(f"embedder   : {report.data['embedder']}")
        lines.append("")

        per_system = report.data.get("per_system") or {}
        if per_system:
            metrics = ["recall_at_5", "mrr", "ndcg_at_10", "abstention_appropriate"]
            lines.append(f"  {'sys':5s}" + "".join(f"{m[:18]:>22s}" for m in metrics))
            lines.append("  " + "-" * 93)
            for key in SYSTEM_ORDER:
                stats = per_system.get(key)
                if not stats:
                    continue
                cells = ""
                for metric in metrics:
                    entry = stats.get(metric)
                    cells += (
                        f"{entry['mean']:.3f} [{entry['ci_low']:.2f},{entry['ci_high']:.2f}]".rjust(22)
                        if entry else f"{'—':>22s}"
                    )
                lines.append(f"  {key:5s}{cells}")
            lines.append("")

        comparisons = report.data.get("comparisons") or report.data.get(
            "comparisons_vs_baseline"
        ) or {}
        if comparisons:
            reference = report.data.get("proposed") or report.data.get("baseline") or "reference"
            lines += [f"  Pairwise versus {reference}:",
                      f"  {'vs':10s} {'metric':26s} {'delta':>10s} {'p':>9s} {'d':>8s}",
                      "  " + "-" * 70]
            for other, metrics in comparisons.items():
                for metric, stats in metrics.items():
                    marker = ("***" if stats["p_value"] < 0.001 else
                              "**" if stats["p_value"] < 0.01 else
                              "*" if stats["p_value"] < 0.05 else "ns")
                    lines.append(
                        f"  {other:10s} {metric:26s} {stats['delta']:+10.4f} "
                        f"{stats['p_value']:9.4f} {stats.get('cohens_d', 0):+8.3f} {marker}"
                    )
            lines.append("")

        ablations = report.data.get("ablations") or {}
        if ablations:
            lines += [f"  {'ablation':28s} {'metric':26s} {'delta':>10s} {'p':>9s} {'d':>8s}",
                      "  " + "-" * 88]
            for key, info in ablations.items():
                for metric, stats in (info.get("metrics") or {}).items():
                    marker = ("***" if stats["p_value"] < 0.001 else
                              "**" if stats["p_value"] < 0.01 else
                              "*" if stats["p_value"] < 0.05 else "ns")
                    lines.append(
                        f"  {key:28s} {metric:26s} {stats['delta_vs_full']:+10.4f} "
                        f"{stats['p_value']:9.4f} {stats.get('cohens_d', 0):+8.3f} {marker}"
                    )
            lines.append("")

        by_type = report.data.get("recall_by_query_type") or {}
        if by_type:
            keys = [k for k in SYSTEM_ORDER if any(k in v for v in by_type.values())]
            lines += ["  Recall@5 by query type "
                      "(— means no gold chunks; scored by abstention instead):",
                      f"  {'query type':34s}" + "".join(f"{k:>8s}" for k in keys),
                      "  " + "-" * (34 + 8 * len(keys))]
            for query_type, per_key in sorted(by_type.items()):
                cells = "".join(
                    f"{per_key[k]:8.3f}" if k in per_key and per_key[k] is not None
                    else f"{'—':>8s}"
                    for k in keys
                )
                lines.append(f"  {query_type:34s}{cells}")
            lines.append("")

        interference = report.data.get("interference") or {}
        if interference:
            lines += ["  Interference from unreadable documents:",
                      f"  {'condition':22s} {'tau':>7s} {'top1':>7s} {'order':>7s} "
                      f"{'route':>7s}  pure?",
                      "  " + "-" * 68]
            for label, info in interference.items():
                lines.append(
                    f"  {label:22s} {info.get('mean_kendall_tau', 0):7.3f} "
                    f"{info.get('top1_flip_rate', 0):7.3f} "
                    f"{info.get('order_change_rate', 0):7.3f} "
                    f"{info.get('route_change_rate', 0):7.3f}  "
                    f"{'yes' if info.get('non_interfering') else 'NO'}"
                )
            lines.append("")

        lattice = report.data.get("lattice") or {}
        if lattice:
            lines += ["  ACL lattice:",
                      f"    distinct classes      : {lattice.get('distinct_classes')}",
                      f"    principals            : {lattice.get('principals')}",
                      f"    total chunks          : {lattice.get('total_chunks')}",
                      f"    class sizes           : {lattice.get('class_sizes')}",
                      f"    smallest class share  : {lattice.get('smallest_class_fraction')}",
                      ""]

    path = out / "RESULTS.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(path)
    print(f"  {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render experiment reports as assets")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out = (PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reports = load_reports()
    if not reports:
        print("No reports in data/reports/. Run an experiment first; see RUNNING.md.")
        sys.exit(1)

    print(f"Reading {len(reports)} report(s); writing to {out.relative_to(PROJECT_ROOT)}/")
    written: list[Path] = []
    for report in reports:
        if report.kind in {"baselines", "baselines_heldout"}:
            chart_system_comparison(report.data, out, written)
        elif report.kind == "ablations":
            chart_ablations(report.data, out, written)
        elif report.kind == "embeddings":
            chart_embeddings(report.data, out, written)
        elif report.kind == "specialisation":
            chart_specialisation(report.data, out, written)
    chart_learning_curve(out, written)
    write_tables(reports, out, written)
    print(f"\n{len(written)} file(s) written.")


if __name__ == "__main__":
    main()
