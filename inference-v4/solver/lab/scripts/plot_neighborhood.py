#!/usr/bin/env python3
r"""Plot saved neighborhood studies without running optimization.

Example (capacity grows in proportion to n in these fixture cohorts):
  python3 plot_neighborhood.py STUDY FIGURES --axis n \
    --normalize kernel-contraction:capacity:n --normalize kernel-shared:capacity:n

One figure per family and remaining parameter regime shows success at the
budget beside conditional time/work-to-target. All eligible runs remain in the
success denominator, including failures and unavailable references. The latter
are explicitly counted as unassessed. Random sampling's work is omitted because
samples and solver work units are not comparable. Medians use experiment.rs's
upper empirical quantile, not interpolation. Raw records enforce the plotting
budget even when the experiment ran longer; summary checkpoint counts are
checked independently when summary.json is available.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


METHODS = ["exact", "random", "greedy", "anneal", "joint", "lns"]
LABELS = {
    "exact": "Exact", "random": "Random samples", "greedy": "Coordinate greedy",
    "anneal": "Single-variable exploration", "joint": "Joint greedy", "lns": "LNS",
}
COLORS = dict(zip(METHODS, ["#555b66", "#95674c", "#d59217", "#9b59b6", "#008777", "#2365b0"]))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def target(cost, optimum):
    if cost is None or optimum is None:
        return False
    # Exact integer comparison, including the zero-optimum case.
    return cost * 100 <= optimum * 105


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else None


def aggregate(records, budget):
    result = dict(runs=len(records), reached=0, exact_references=0,
                  no_feasible=0, errors=0, unknown_references=0,
                  target_ms=None, target_work=None)
    times, works = [], []
    for record in records:
        outcome = (record.get("reference") or {}).get("outcome")
        optimum = outcome.get("Optimal") if isinstance(outcome, dict) else None
        points = [p for p in record["points"] if p["elapsed_ms"] <= budget]
        result["exact_references"] += optimum is not None
        result["unknown_references"] += optimum is None
        result["errors"] += record["status"] in ("error", "startup-error")
        result["no_feasible"] += not points or points[-1]["cost"] is None
        hit = next((p for p in points if target(p["cost"], optimum)), None)
        if hit:
            times.append(hit["elapsed_ms"])
            works.append(hit["work"])
    result.update(reached=len(times), target_ms=median(times), target_work=median(works))
    return result


def load_records(study, budget):
    paths = sorted((study / "runs").glob("*.study.json"))
    if not paths:
        raise ValueError(f"No runs/*.study.json records in {study}")
    records, excluded = [], 0
    identities = set()
    for path in paths:
        record = json.loads(path.read_text())
        identity = (record["instance"], record["method"], record["seed"])
        if identity in identities:
            raise ValueError(f"Duplicate run identity: {identity}")
        identities.add(identity)
        if record["settings"]["milliseconds"] < budget:
            excluded += 1
        else:
            records.append(record)
    if not records:
        raise ValueError(f"No runs have a declared budget of at least {budget} ms")
    summary = study / "summary.json"
    if summary.exists():
        groups = defaultdict(list)
        for record in records:
            groups[(record["family"], canonical(record["parameters"]), record["method"])].append(record)
        for row in json.loads(summary.read_text()):
            checkpoint = next((p for p in row["checkpoints"] if p["budget_ms"] == budget), None)
            if checkpoint is None:
                continue
            key = (row["family"], canonical(json.loads(row["regime"])), row["method"])
            computed = aggregate(groups.pop(key, []), budget)
            for actual, saved in [("runs", "runs"), ("reached", "within_five_percent"),
                                  ("exact_references", "exact_references"), ("no_feasible", "no_feasible")]:
                if computed[actual] != checkpoint[saved]:
                    raise ValueError(f"Summary/records disagree for {key}, {actual}: "
                                     f"{computed[actual]} != {checkpoint[saved]}; study may still be updating")
        if groups and budget in (10, 30, 100, 300, 1000):
            raise ValueError("Record groups missing from summary; study may still be updating")
    return records, excluded


def regime_for(parameters, family, axis, normalizations):
    regime = dict(parameters)
    for restricted_family, numerator, denominator in normalizations:
        if restricted_family is not None and restricted_family != family:
            continue
        if numerator == axis:
            raise ValueError("Cannot normalize the selected x-axis parameter")
        ratio = Fraction(parameters[numerator], parameters[denominator])
        del regime[numerator]
        regime[f"{numerator}/{denominator}"] = str(ratio)
    del regime[axis]
    return regime


def draw(family, regime, groups, study, output, axis, budget):
    xs = sorted({x for x, _ in groups})
    methods = [m for m in METHODS if any(key[1] == m for key in groups)]
    methods += sorted({key[1] for key in groups} - set(METHODS))
    rows = {(x, m): aggregate(group, budget) for (x, m), group in groups.items()}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.4), sharex=True)
    fig.subplots_adjust(left=.064, right=.976, bottom=.29, top=.69, wspace=.27)
    comparison = f"{axis} sweep" if len(xs) > 1 else f"{axis}={xs[0]}"
    family_label = "repeated (independent serial)" if family == "repeated" else family
    fig.suptitle(f"{family_label}  |  {study.name}  |  {comparison}", x=.064, ha="left", y=.97,
                 fontsize=17, fontweight="bold")
    description = ", ".join(f"{key}={value}" for key, value in sorted(regime.items()))
    fig.text(.064, .918, textwrap.fill(description, 135), va="top", fontsize=9, color="#46505e")
    labels = ["Target attainment", "Time to target, successes only", "Work to target, successes only"]
    for ax, label in zip(axes, labels):
        ax.set_title(label, loc="left", fontsize=11, pad=13)
        ax.set_xlabel(f"{axis} (declared model parameter)", fontsize=9)
        if xs[0] > 0:
            ax.set_xscale("log", base=2)
        else:
            ax.set_xscale("symlog", base=2, linthresh=1)
        ax.set_xticks(xs, [str(x) for x in xs])
        if len(xs) == 1:
            ax.set_xlim((xs[0] / 1.3, xs[0] * 1.3) if xs[0] > 0 else (-.5, .5))
        ax.grid(True, alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel(f"Within 5% of optimum by {budget:g} ms")
    axes[0].set_ylim(-.06, 1.10)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].axhline(.95, color="#8a8a8a", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Median observed ms (log scale)")
    axes[2].set_ylabel("Median reported solver work (log scale)")
    for ax in axes[1:]:
        ax.set_yscale("log")
    axes[1].axhline(budget, color="#8a8a8a", linestyle=":", linewidth=1)
    positive_times = [r["target_ms"] for r in rows.values() if r["target_ms"]]
    axes[1].set_ylim(max(min(positive_times, default=1) * .5, .00001), budget * 1.8)

    for method in methods:
        color = COLORS.get(method, "#111111")
        available = [(x, rows[x, method]) for x in xs if (x, method) in rows]
        axes[0].plot([x for x, _ in available], [r["reached"] / r["runs"] for _, r in available],
                     color=color, marker="o", markersize=4, linewidth=1.6, alpha=.9)
        for panel, field in [(1, "target_ms"), (2, "target_work")]:
            if panel == 2 and method == "random":
                continue
            ys = [r[field] if r[field] is not None and r[field] > 0 else np.nan for _, r in available]
            axes[panel].plot([x for x, _ in available], ys, color=color, linewidth=1.4, alpha=.85)
            for (x, row), y in zip(available, ys):
                if np.isfinite(y):
                    axes[panel].plot(x, y, marker="o", markersize=5, markeredgecolor=color,
                                     markerfacecolor=color if row["reached"] == row["runs"] else "white")
                elif panel == 1:
                    # These crosses are a censoring annotation, not a measured median.
                    axes[panel].plot(x, budget, marker="x", color=color, markersize=6)

    handles = [Line2D([0], [0], color=COLORS.get(m, "black"), marker="o", markersize=4,
                      label=LABELS.get(m, m)) for m in methods]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.057, .834), ncol=3,
               frameon=False, fontsize=9, columnspacing=2.5)
    counts = []
    for method in methods:
        cells = []
        for x in xs:
            row = rows.get((x, method))
            if row:
                suffix = f"; {row['unknown_references']} unassessed" if row["unknown_references"] else ""
                cells.append(f"{x}: {row['reached']}/{row['runs']}{suffix}")
        counts.append(f"{LABELS.get(method, method)} — " + ", ".join(cells))
    fig.text(.064, .207, "Reached target / all eligible runs, by " + axis + ":", fontsize=9,
             fontweight="bold")
    for index, label in enumerate(counts):
        column, row = index % 2, index // 2
        fig.text(.064 + column * .46, .177 - row * .030, label, fontsize=8,
                 color=COLORS.get(methods[index], "black"))
    note = ("Hollow markers: partial success. × at the time limit: no target; no median is inferred. "
            "Dotted lines: 95% attainment / time budget.\n"
            "Times include model construction, initialization and validation; process startup is excluded. "
            "Random sample counts are omitted from work.\n"
            "All non-axis parameters are held fixed within each figure (any explicit normalization is shown above).")
    fig.text(.064, .025, note, fontsize=8, color="#525a64", va="bottom", linespacing=1.45)
    digest = hashlib.sha256(canonical(regime).encode()).hexdigest()[:8]
    stem = f"{family}-{axis}-{digest}"
    for suffix in ("png", "svg"):
        metadata = {"Date": None, "Creator": "plot_neighborhood.py"} if suffix == "svg" else {
            "Software": "plot_neighborhood.py"
        }
        fig.savefig(output / f"{stem}.{suffix}", dpi=170, facecolor="white", metadata=metadata)
    plt.close(fig)
    return {"figure": stem, "family": family, "axis": axis, "regime": regime,
            "points": [dict(x=x, method=method, **value) for (x, method), value in sorted(rows.items())]}


def overview(records, study, output, budget):
    """Compare a small ablation cohort without aggregating its different regimes."""
    groups = defaultdict(list)
    regimes = {}
    for record in records:
        key = (record["family"], canonical(record["parameters"]))
        regimes[key] = record["parameters"]
        groups[key, record["method"]].append(record)
    keys = sorted(regimes)
    if len(keys) > 12:
        raise ValueError("Overview supports at most 12 regimes; use axis-specific figures")
    methods = [m for m in METHODS if any(method == m for _, method in groups)]
    common = {k: v for k, v in regimes[keys[0]].items()
              if all(p[k] == v for p in regimes.values())}
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.4), sharey=True)
    fig.subplots_adjust(left=.30, right=.972, bottom=.17, top=.75, wspace=.26)
    fig.suptitle(f"{study.name}  |  separate regimes", x=.035, ha="left", y=.96,
                 fontsize=17, fontweight="bold")
    fig.text(.035, .902, "Common parameters: " + ", ".join(f"{k}={v}" for k, v in sorted(common.items())),
             fontsize=9, color="#46505e")
    fig.legend(handles=[Line2D([0], [0], marker="o", color=COLORS[m], label=LABELS[m]) for m in methods],
               loc="upper left", bbox_to_anchor=(.03, .86), frameon=False, ncol=3, fontsize=10)
    names = []
    output_rows = []
    for index, key in enumerate(keys):
        varying = ", ".join(f"{k}={v}" for k, v in regimes[key].items() if k not in common)
        names.append(key[0] + "\n" + textwrap.fill(varying, 47))
        for mi, method in enumerate(methods):
            if (key, method) not in groups:
                continue
            row = aggregate(groups[key, method], budget)
            output_rows.append(dict(family=key[0], parameters=regimes[key], method=method, **row))
            y = index + (mi - (len(methods) - 1) / 2) * .19
            color = COLORS[method]
            rate = row["reached"] / row["runs"]
            axes[0].plot(rate, y, marker="o", color=color)
            axes[0].annotate(f"{row['reached']}/{row['runs']}", (rate, y), xytext=(6, -3),
                             textcoords="offset points", fontsize=8, color=color)
            for panel, field in [(1, "target_ms"), (2, "target_work")]:
                if panel == 2 and method == "random":
                    continue
                if row[field] is not None and row[field] > 0:
                    axes[panel].plot(row[field], y, marker="o", markeredgecolor=color,
                                     markerfacecolor=color if row["reached"] == row["runs"] else "white")
                elif panel == 1:
                    axes[panel].plot(budget, y, marker="x", color=color)
    axes[0].set_yticks(range(len(keys)), names, fontsize=9)
    axes[0].set_ylim(len(keys) - .5, -.5)
    axes[0].set_xlim(-.04, 1.18)
    axes[0].set_xticks([0, .5, 1])
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].axvline(.95, color="#8a8a8a", linestyle=":", linewidth=1)
    axes[1].axvline(budget, color="#8a8a8a", linestyle=":", linewidth=1)
    for ax, title in zip(axes, [f"Within 5% by {budget:g} ms", "Conditional median ms", "Conditional median solver work"]):
        ax.set_title(title, fontsize=10, loc="left", pad=14)
        ax.grid(True, alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1:]:
        ax.set_xscale("log")
        ax.tick_params(axis="y", length=0)
    fig.text(.035, .07,
             "Counts retain every eligible run in each regime. Hollow markers: partial success. "
             "× at the time limit: no target; no median is inferred.\n"
             "Times include construction, initialization and validation. Dotted lines mark 95% attainment and the time budget. "
             "Each row is a distinct model regime.", fontsize=9, color="#525a64", linespacing=1.5)
    for suffix in ("png", "svg"):
        metadata = {"Date": None, "Creator": "plot_neighborhood.py"} if suffix == "svg" else {"Software": "plot_neighborhood.py"}
        fig.savefig(output / f"overview.{suffix}", dpi=170, facecolor="white", metadata=metadata)
    plt.close(fig)
    return {"figure": "overview", "points": output_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", choices=("n", "d", "capacity", "repeat"), default="n")
    parser.add_argument("--sweeps-only", action="store_true",
                        help="Only plot strata with at least two values of the selected axis")
    parser.add_argument("--budget-ms", type=int, default=1000)
    parser.add_argument("--overview", action="store_true", help="One panel row per distinct regime (up to 12)")
    parser.add_argument("--normalize", action="append", default=[], metavar="[FAMILY:]PARAM:DIVISOR",
                        help="Explicit ratio grouping, optionally confined to one family; never inferred")
    args = parser.parse_args()
    if args.budget_ms <= 0:
        parser.error("--budget-ms must be positive")
    normalizations = []
    for value in args.normalize:
        pair = value.split(":")
        if len(pair) == 2:
            pair.insert(0, None)
        if len(pair) != 3:
            parser.error("--normalize requires [FAMILY:]PARAM:DIVISOR")
        normalizations.append(tuple(pair))
    if args.overview and (normalizations or args.sweeps_only):
        parser.error("--overview keeps complete regimes; do not combine it with normalization or --sweeps-only")
    records, excluded = load_records(args.study, args.budget_ms)
    figures = defaultdict(lambda: defaultdict(list))
    for record in records:
        parameters = record["parameters"]
        regime = regime_for(parameters, record["family"], args.axis, normalizations)
        figures[(record["family"], canonical(regime))][parameters[args.axis], record["method"]].append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "magnitude-solver-neighborhood-study"})
    manifest = {"study": str(args.study.resolve()), "budget_ms": args.budget_ms,
                "eligible_runs": len(records), "shorter_budget_runs_excluded": excluded,
                "axis": args.axis, "normalizations": normalizations,
                "single_value_strata_omitted": 0, "plotted_runs": 0, "figures": []}
    if args.overview:
        manifest.update(axis=None, plotted_runs=len(records),
                        figures=[overview(records, args.study, args.output, args.budget_ms)])
        figures = {}
    for (family, regime), groups in sorted(figures.items()):
        if args.sweeps_only and len({x for x, _ in groups}) < 2:
            manifest["single_value_strata_omitted"] += 1
            continue
        manifest["plotted_runs"] += sum(len(group) for group in groups.values())
        manifest["figures"].append(draw(family, json.loads(regime), groups, args.study,
                                      args.output, args.axis, args.budget_ms))
    (args.output / "plot-data.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest['figures'])} PNG/SVG figure pairs to {args.output}; "
          f"{manifest['plotted_runs']}/{len(records)} eligible runs plotted, "
          f"{excluded} shorter-budget runs and {manifest['single_value_strata_omitted']} "
          "single-value strata excluded.")


if __name__ == "__main__":
    main()
