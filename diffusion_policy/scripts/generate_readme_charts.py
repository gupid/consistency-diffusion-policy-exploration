if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASELINE_COLOR = "#475569"
CONSISTENCY_COLOR = "#0F766E"
ACCENT_COLOR = "#F59E0B"
GRID_COLOR = "#D7DEE7"
TEXT_COLOR = "#0F172A"


def parse_eval_entries(path: Path):
    evals = []
    with path.open("r") as f:
        for line in f:
            obj = json.loads(line)
            if "test/mean_score" not in obj:
                continue
            rewards = {
                int(key.rsplit("_", 1)[-1]): value
                for key, value in obj.items()
                if key.startswith("test/sim_max_reward_")
            }
            evals.append(
                {
                    "epoch": int(obj["epoch"]),
                    "test_mean_score": float(obj["test/mean_score"]),
                    "train_mean_score": float(obj.get("train/mean_score", np.nan)),
                    "train_action_mse_error": float(obj.get("train_action_mse_error", np.nan)),
                    "rewards": rewards,
                }
            )
    if not evals:
        raise RuntimeError(f"No evaluation entries found in {path}")
    return evals


def get_best_entry(entries):
    return max(entries, key=lambda x: x["test_mean_score"])


def format_percent_ratio(a: float, b: float):
    return 100.0 * a / b


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")


def save_fig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_score_vs_steps(baseline, consistency, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    xs = [baseline["inference_steps"], consistency["inference_steps"]]
    ys = [baseline["best"]["test_mean_score"], consistency["best"]["test_mean_score"]]

    ax.scatter(
        [xs[0]],
        [ys[0]],
        s=160,
        color=BASELINE_COLOR,
        label="Diffusion Policy",
        zorder=3,
    )
    ax.scatter(
        [xs[1]],
        [ys[1]],
        s=160,
        color=CONSISTENCY_COLOR,
        label="Consistency Policy",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xlim(3, 140)
    ax.set_ylim(0.82, 0.88)
    ax.set_xlabel("Inference Steps (log scale)")
    ax.set_ylabel("Best Test Mean Score")
    ax.set_title("Push-T: Similar Score with Far Fewer Inference Steps", fontsize=13, pad=12)

    ratio = baseline["inference_steps"] / consistency["inference_steps"]
    retained = format_percent_ratio(consistency["best"]["test_mean_score"], baseline["best"]["test_mean_score"])
    ax.annotate(
        f"{ratio:.0f}x fewer steps\n{retained:.2f}% score retained",
        xy=(consistency["inference_steps"], consistency["best"]["test_mean_score"]),
        xytext=(10, 0.854),
        textcoords="data",
        fontsize=10,
        color=TEXT_COLOR,
        arrowprops=dict(arrowstyle="->", color=ACCENT_COLOR, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF7ED", ec="#FED7AA"),
    )

    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    save_fig(fig, out_path)


def plot_score_vs_epoch(baseline, consistency, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for run, color, label in [
        (baseline, BASELINE_COLOR, "Diffusion Policy"),
        (consistency, CONSISTENCY_COLOR, "Consistency Policy"),
    ]:
        epochs = [entry["epoch"] for entry in run["entries"]]
        scores = [entry["test_mean_score"] for entry in run["entries"]]
        ax.plot(epochs, scores, marker="o", markersize=4, linewidth=2.2, color=color, label=label)

    ax.set_xlim(-3, max(baseline["entries"][-1]["epoch"], consistency["entries"][-1]["epoch"]) + 10)
    ax.set_ylim(0.1, 0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Mean Score")
    ax.set_title("Push-T: Test Score over Training", fontsize=13, pad=12)
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    save_fig(fig, out_path)


def plot_reward_sorted(baseline, consistency, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    baseline_rewards = sorted(baseline["best"]["rewards"].values(), reverse=True)
    consistency_rewards = sorted(consistency["best"]["rewards"].values(), reverse=True)
    x = np.arange(1, len(baseline_rewards) + 1)

    ax.plot(x, baseline_rewards, linewidth=2.4, color=BASELINE_COLOR, label="Diffusion Policy")
    ax.plot(x, consistency_rewards, linewidth=2.4, color=CONSISTENCY_COLOR, label="Consistency Policy")
    ax.fill_between(x, baseline_rewards, alpha=0.08, color=BASELINE_COLOR)
    ax.fill_between(x, consistency_rewards, alpha=0.08, color=CONSISTENCY_COLOR)

    ax.set_xlim(1, len(x))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Test Episode Rank (sorted by reward)")
    ax.set_ylabel("Max Reward")
    ax.set_title("Push-T: Reward Distribution across 50 Test Seeds", fontsize=13, pad=12)
    ax.legend(frameon=False, loc="lower left")
    style_axes(ax)
    save_fig(fig, out_path)


def plot_success_thresholds(baseline, consistency, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    thresholds = [0.90, 0.95, 0.99]
    labels = [">= 0.90", ">= 0.95", ">= 0.99"]
    x = np.arange(len(thresholds))
    width = 0.34

    def counts(run):
        rewards = list(run["best"]["rewards"].values())
        return [sum(reward >= thr for reward in rewards) for thr in thresholds]

    baseline_counts = counts(baseline)
    consistency_counts = counts(consistency)

    ax.bar(x - width / 2, baseline_counts, width=width, color=BASELINE_COLOR, label="Diffusion Policy")
    ax.bar(x + width / 2, consistency_counts, width=width, color=CONSISTENCY_COLOR, label="Consistency Policy")

    for xpos, values in [(x - width / 2, baseline_counts), (x + width / 2, consistency_counts)]:
        for xi, yi in zip(xpos, values):
            ax.text(xi, yi + 0.6, str(int(yi)), ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Number of Test Seeds")
    ax.set_title("Push-T: High-Reward Episode Counts", fontsize=13, pad=12)
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    save_fig(fig, out_path)


def write_summary_json(baseline, consistency, out_path: Path):
    summary = {
        "baseline": {
            "inference_steps": baseline["inference_steps"],
            "best_epoch": baseline["best"]["epoch"],
            "best_test_mean_score": baseline["best"]["test_mean_score"],
        },
        "consistency": {
            "inference_steps": consistency["inference_steps"],
            "best_epoch": consistency["best"]["epoch"],
            "best_test_mean_score": consistency["best"]["test_mean_score"],
        },
        "step_reduction_ratio": baseline["inference_steps"] / consistency["inference_steps"],
        "score_retained_percent": format_percent_ratio(
            consistency["best"]["test_mean_score"], baseline["best"]["test_mean_score"]
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate README charts for Push-T baseline vs consistency runs.")
    parser.add_argument(
        "--baseline-log",
        default="data/outputs/2026.03.13/10.40.52_train_diffusion_unet_lowdim_pusht_lowdim/logs.json.txt",
    )
    parser.add_argument(
        "--consistency-log",
        default="data/outputs/2026.03.21/09.56.59_train_consistency_unet_lowdim_pusht_lowdim/logs.json.txt",
    )
    parser.add_argument("--baseline-steps", type=int, default=100)
    parser.add_argument("--consistency-steps", type=int, default=4)
    parser.add_argument("--output-dir", default="assets/readme")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    baseline_entries = parse_eval_entries(Path(args.baseline_log))
    consistency_entries = parse_eval_entries(Path(args.consistency_log))

    baseline = {
        "entries": baseline_entries,
        "best": get_best_entry(baseline_entries),
        "inference_steps": args.baseline_steps,
    }
    consistency = {
        "entries": consistency_entries,
        "best": get_best_entry(consistency_entries),
        "inference_steps": args.consistency_steps,
    }

    plot_score_vs_steps(baseline, consistency, output_dir / "pusht_score_vs_steps.png")
    plot_score_vs_epoch(baseline, consistency, output_dir / "pusht_score_vs_epoch.png")
    plot_reward_sorted(baseline, consistency, output_dir / "pusht_reward_sorted.png")
    plot_success_thresholds(baseline, consistency, output_dir / "pusht_success_thresholds.png")
    write_summary_json(baseline, consistency, output_dir / "pusht_summary.json")

    print(f"Generated charts in {output_dir}")


if __name__ == "__main__":
    main()
