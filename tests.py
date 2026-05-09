
import argparse
import os
import math
import collections
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Core import TaskCategory, TaskGenerator, linear_penalty
from Environment import Environment
from PPOPointerNetwork import PPO_Pointer_Network, MicroEnv, evaluate_policy
from PPOSchedulerAgent import PPOSchedulerAgent
from utils import Log, plot_general


def make_env(steps):
    long_cat = TaskCategory(
        name="Long", task_id=2, category_seed=2,
        mean_time=10, std_time=2,
        mean_buffer_time=4.0, std_buffer_time=0.3,
        mean_reward=2.0, std_reward=1.0,
        penalty_fn=linear_penalty
    )
    short_cat = TaskCategory(
        name="Short", task_id=1, category_seed=1,
        mean_time=2, std_time=1,
        mean_buffer_time=2.0, std_buffer_time=0.3,
        mean_reward=20.0, std_reward=0.3,
        penalty_fn=linear_penalty
    )
    generators = [
        TaskGenerator(long_cat,  generator_seed=10, probability=0.05),   # rare but large
        TaskGenerator(short_cat, generator_seed=11, probability=0.3),   # frequent, tiny
    ]
    env = Environment(generators=generators, timesteps=steps * 10)
    return MicroEnv(env)

def plot_position_heatmap(logger: Log, H: int, filename: str):

    after = [r for r in logger.logs if r.get("event_type") == "after_insert"]
    if not after:
        print(f"  skip {filename} — no insertions logged"); return

    task_ids = sorted(set(r["task_type"] for r in after))
    grid = np.zeros((len(task_ids), H))
    for r in after:
        row = task_ids.index(r["task_type"])
        if 0 <= r["position"] < H:
            grid[row, r["position"]] += 1

    fig, ax = plt.subplots(figsize=(max(8, H * 1.1), 3 + len(task_ids) * 0.8))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(H))
    ax.set_xticklabels([f"pos {i}" for i in range(H)], fontsize=8)
    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels(["Short (id=1)" if t == 1 else "Long (id=2)" for t in task_ids], fontsize=9)
    ax.set_xlabel("Schedule window position", fontsize=10)
    ax.set_title("Long vs Short — Placement Heatmap (task × position)", fontsize=11)
    plt.colorbar(im, ax=ax, label="# placements")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ok {filename}")


def plot_timing_heatmap(logger: Log, H: int, filename: str, bin_size: int = 10):

    after = [r for r in logger.logs if r.get("event_type") == "after_insert"]
    if not after:
        print(f"  skip {filename}"); return

    max_t  = max(r["t"] for r in after)
    n_bins = max(1, math.ceil(max_t / bin_size))
    grid   = np.zeros((n_bins, H))
    for r in after:
        b = min(int(r["t"] // bin_size), n_bins - 1)
        if 0 <= r["position"] < H:
            grid[b, r["position"]] += 1

    fig, ax = plt.subplots(figsize=(max(8, H * 1.1), max(4, n_bins * 0.45)))
    im = ax.imshow(grid, aspect="auto", cmap="Blues", interpolation="nearest", origin="upper")
    ax.set_xticks(range(H))
    ax.set_xticklabels([f"pos {i}" for i in range(H)], fontsize=8)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels([f"t={b*bin_size}–{(b+1)*bin_size}" for b in range(n_bins)], fontsize=7)
    ax.set_xlabel("Schedule window position", fontsize=10)
    ax.set_ylabel("Timestep bin", fontsize=10)
    ax.set_title("Long vs Short — Timing Heatmap (time-bin × position)", fontsize=11)
    plt.colorbar(im, ax=ax, label="# placements")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ok {filename}")


def plot_pred_vs_true(logger: Log, filename: str):
    by_type = collections.defaultdict(lambda: {"pred": [], "true": []})
    for r in logger.logs:
        if r.get("event_type") != "after_insert": continue
        true_len = sum(1 for s in r["schedule"] if s == r["job_id"])
        by_type[r["task_type"]]["pred"].append(r["pred_length"])
        by_type[r["task_type"]]["true"].append(true_len)

    if not by_type:
        print(f"  skip {filename}"); return

    labels = {1: "Short", 2: "Long"}
    n = len(by_type)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (tid, vals) in zip(axes[0], sorted(by_type.items())):
        preds = np.array(vals["pred"])
        trues = np.array(vals["true"])
        ax.scatter(trues, preds, alpha=0.5, s=25, color="steelblue")
        lim = max(preds.max(), trues.max()) + 1
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="perfect")
        ax.set_xlabel("True slots used", fontsize=9)
        ax.set_ylabel("Predicted length", fontsize=9)
        ax.set_title(f"{labels.get(tid, tid)}  (n={len(preds)})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Long vs Short — Predicted vs True Length", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ok {filename}")


def plot_occupancy_with_insertions(logger: Log, H: int, filename: str):
    before = [r for r in logger.logs if r.get("event_type") == "before_insert"]
    after  = [r for r in logger.logs if r.get("event_type") == "after_insert"]
    if not before:
        print(f"  skip {filename}"); return

    ts  = [r["t"] for r in before]
    occ = [sum(1 for x in r["schedule"] if x != -1) / H for r in before]

    short_ts = [r["t"] for r in after if r["task_type"] == 1]
    long_ts  = [r["t"] for r in after if r["task_type"] == 2]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(ts, occ, alpha=0.25, color="steelblue")
    ax.plot(ts, occ, color="steelblue", linewidth=1.5, label="Occupancy")
    ax.scatter(short_ts, [1.02] * len(short_ts), marker="|", color="darkorange",
               s=60, label="Short inserted", zorder=5)
    ax.scatter(long_ts,  [1.05] * len(long_ts),  marker="|", color="darkred",
               s=80, label="Long inserted",  zorder=5)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_ylabel("Occupancy (fraction of H)", fontsize=10)
    ax.set_title("Long vs Short — Occupancy & Insertion Events", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ok {filename}")

def run(model_path, H, steps, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    model = PPO_Pointer_Network(H)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"Loaded {model_path}  H={H}\n")

    env       = make_env(steps)
    scheduler = PPOSchedulerAgent(model, H)

    # evaluate_policy already handles:
    #   - logsched_step (before_insert / after_insert)
    #   - save_logs_to_file
    #   - plot_pairs  (the before/after Gantt you already have)
    rewards = evaluate_policy(
        env, model, scheduler,
        num_episodes=5,
        steps=steps,
        visualize=True,
        visual_prefix=os.path.join(out_dir, "long_vs_short"),
        snapshot_every=1,
    )

    logger = scheduler.logger
    pfx    = os.path.join(out_dir, "long_vs_short")

    # new plots on top of what evaluate_policy already logged
    plot_position_heatmap          (logger, H,   f"{pfx}_heatmap_position.png")
    plot_timing_heatmap            (logger, H,   f"{pfx}_heatmap_timing.png")
    plot_pred_vs_true              (logger,      f"{pfx}_pred_vs_true.png")
    plot_occupancy_with_insertions (logger, H,   f"{pfx}_occupancy.png")
    plot_general(rewards, "Long vs Short — Episode Rewards", f"{pfx}_rewards.png")

    n_short = sum(1 for r in logger.logs if r.get("event_type") == "after_insert" and r["task_type"] == 1)
    n_long  = sum(1 for r in logger.logs if r.get("event_type") == "after_insert" and r["task_type"] == 2)
    print(f"\nInsertions — Short: {n_short}  Long: {n_long}")
    print(f"Rewards: {[round(r, 1) for r in rewards]}")
    print(f"Mean reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Long vs Spurious Short Test")
    parser.add_argument("--model", required=True, help="Path to saved .pth model")
    parser.add_argument("--H",     type=int, default=8,   help="Schedule window size")
    parser.add_argument("--steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--out",   default="test_long_vs_short", help="Output directory")
    args = parser.parse_args()
    run(args.model, args.H, args.steps, args.out)
