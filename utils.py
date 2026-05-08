import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import random

##IGNORE OTHER GRAPGHS DELETING IN ONE SEC JUST INSERT PAIRS
def plot_diagnostics(metrics, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        ("policy_loss", "Policy Loss"),
        ("value_loss", "Value Loss"),
        ("dur_loss", "Duration Loss"),
        ("entropy", "Entropy"),
    ]

    for ax, (key, label) in zip(axes.flat, panels):
        if key in metrics and metrics[key]:
            ax.plot(metrics[key], linewidth=1.5, color="steelblue")
            ax.set_title(f"{title} — {label}", fontsize=11)
            ax.set_xlabel("Iteration", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_general(rewards, title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(rewards, linewidth=1.5, color="steelblue")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Reward", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


class Log:
    def __init__(self):
        self.logs = []

    def logsched_step(self, t, job_id, task_type, position, pred_length, inserted, schedule, event_type="step"):
        row = {
            "t": t,
            "event_type": event_type,
            "job_id": job_id,
            "task_type": task_type,
            "position": position,
            "pred_length": pred_length,
            "inserted": inserted,
            "schedule": schedule.copy()
        }

        self.logs.append(row)

    def save_logs_to_file(self, filename="schedule_logs.txt"):
        if not self.logs:
            print("No scheduler logs to save.")
            return

        with open(filename, "w") as f:
            for row in self.logs:
                f.write(
                    f"t={row['t']} | "
                    f"event={row.get('event_type', 'step')} | "
                    f"job_id={row['job_id']} | "
                    f"task_type={row['task_type']} | "
                    f"position={row['position']} | "
                    f"pred_length={row['pred_length']} | "
                    f"inserted={row['inserted']} | "
                    f"schedule={row['schedule']}\n"
                )

        print(f"Saved scheduler logs to {filename}")
    
    def plot_pairs(self, filename="schedule_insert_pairs.png", max_pairs=25):
        import matplotlib.pyplot as plt
        import random

        if not self.logs:
            print("No scheduler logs to plot.")
            return

        grouped = {}

        for row in self.logs:
            if row.get("event_type") in ["before_insert", "after_insert"]:
                grouped.setdefault(row["t"], {})
                grouped[row["t"]][row.get("event_type")] = row

        pairs = []

        for t in sorted(grouped.keys()):
            if "before_insert" in grouped[t] and "after_insert" in grouped[t]:
                pairs.append((grouped[t]["before_insert"], grouped[t]["after_insert"]))

        pairs = pairs[:max_pairs]

        if not pairs:
            print("No before/after insert pairs found.")
            return

        unique_jobs = set()
        for before, after in pairs:
            for row in [before, after]:
                for job_id in row["schedule"]:
                    if job_id is not None and job_id != -1:
                        unique_jobs.add(job_id)

        random.seed(42)

        job_colors = {
            job: (
                random.random(),
                random.random(),
                random.random()
            )
            for job in sorted(unique_jobs)
        }

        plt.figure(figsize=(14, max(6, len(pairs) * 0.7)))

        y = 0
        y_positions = []
        y_labels = []

        for before, after in pairs:
            for label, row in [("BEFORE", before), ("AFTER", after)]:
                schedule = row["schedule"]

                start = None
                current_job = None

                for i, job_id in enumerate(schedule + [None]):
                    if job_id != current_job:
                        if current_job is not None and current_job != -1:
                            duration = i - start

                            plt.barh(
                                y=y,
                                width=duration,
                                left=start,
                                height=0.7,
                                color=job_colors[current_job]
                            )

                            plt.text(
                                start + duration / 2,
                                y,
                                str(current_job),
                                ha="center",
                                va="center",
                                fontsize=8,
                                fontweight="bold"
                            )

                        current_job = job_id
                        start = i

                y_positions.append(y)
                y_labels.append(f"t={row['t']} {label}")
                y += 1

            y += 0.7

        plt.yticks(y_positions, y_labels, fontsize=8)
        plt.xlabel("Schedule position")
        plt.ylabel("Before / After Insert Pairs")
        plt.title("Schedule Change Caused by Insertions")
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved insert-pair Gantt chart to {filename}")