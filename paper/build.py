import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(rows, columns):

    def escape(value):
        return str(value).replace("_", "\\_").replace("%", "\\%")

    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(map(escape, columns)) + " \\\\",
        "\\midrule",
    ]
    lines.extend(
        (
            " & ".join((escape(row[column]) for column in columns)) + " \\\\"
            for row in rows
        )
    )
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def build_tables(output):
    specs = {
        "fixed_detection": (
            "detection/fixed_results.csv",
            [
                "eta",
                "n",
                "detector",
                "true_positives",
                "n_watermarked",
                "false_positives",
                "n_unwatermarked",
            ],
        ),
        "benchmarks": (
            "benchmarks/results.csv",
            ["benchmark", "num_examples", "unwm_score", "wm_score"],
        ),
        "diversity": (
            "diversity/results.csv",
            ["setting", "metric", "mean", "ci95_lower", "ci95_upper", "holm_p"],
        ),
        "substitution": (
            "attacks/substitution_results.csv",
            [
                "scheme",
                "tokens",
                "substitution_rate",
                "true_positives",
                "n_watermarked",
                "false_positives",
                "n_unwatermarked",
            ],
        ),
        "blackbox": (
            "attacks/blackbox_results.csv",
            ["scheme", "test", "setting", "p"],
        ),
    }
    for name, (path, columns) in specs.items():
        with (ROOT / "experiments" / path).open() as handle:
            rows = list(csv.DictReader(handle))
        (output / f"{name}.tex").write_text(table(rows, columns))


def build_plots(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (ROOT / "experiments/detection/figure_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        groups[
            row["family"], row["generation_model"], row["detector_model"], row["eta"]
        ].append(row)
    for index, (key, group) in enumerate(sorted(groups.items())):
        (fig, ax) = plt.subplots(figsize=(4.5, 3.2))
        curves = defaultdict(list)
        for row in group:
            curves[row["construction"], row["detector"]].append(row)
        for (construction, detector), curve in sorted(curves.items()):
            curve = sorted(curve, key=lambda row: int(row["tokens"]))
            x = [int(row["tokens"]) for row in curve]
            y = [float(row["tpr_percent"]) for row in curve]
            ax.plot(x, y, marker="o", markersize=3, label=f"{construction}: {detector}")
            ax.fill_between(
                x,
                [float(r["ci_lower_percent"]) for r in curve],
                [float(r["ci_upper_percent"]) for r in curve],
                alpha=0.12,
            )
        ax.set(
            xlabel="Generated tokens",
            ylabel="Detection rate (%)",
            ylim=(0, 102),
            title=f"{key[1]}, eta={key[3]}",
        )
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output / f"detection_{index:02d}.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/paper")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build_tables(args.output)
    if args.plots:
        build_plots(args.output)


if __name__ == "__main__":
    main()
