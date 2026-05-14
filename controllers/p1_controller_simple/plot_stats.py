import json
import os
from pathlib import Path
import math

import matplotlib.pyplot as plt


def get_latest_stats_file():

    stats_folder = Path("results/training_stats")

    files = list(stats_folder.glob("*.json"))

    if not files:
        raise FileNotFoundError("No stats files found.")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)

    return str(latest_file)


def load_stats(filename):

    with open(filename, "r") as f:
        data = json.load(f)

    return data["stats"], data["config"], data["timestamp"]


def create_plot_folder(controller_name, timestamp):

    folder = (
        f"results/plots/"
        f"{controller_name}/"
        f"{timestamp}"
    )

    os.makedirs(folder, exist_ok=True)

    return folder

def create_trajectory_folder(controller_name):

    folder = (
        f"results/trajectories/"
        f"{controller_name}"
    )

    os.makedirs(folder, exist_ok=True)

    return folder


def plot_fitness(stats, config, plot_folder):

    generations = [s["generation"] for s in stats]

    best_fitness = [s["best_fitness"] for s in stats]
    avg_fitness = [s["avg_fitness"] for s in stats]

    plt.figure()

    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(generations, avg_fitness, label="Average Fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.title(
        f"{config['controller']} Fitness Convergence"
    )

    plt.legend()
    plt.grid(True)

    filepath = f"{plot_folder}/fitness.png"

    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_distance(stats, config, plot_folder):

    generations = [s["generation"] for s in stats]

    best_distance = [s["best_distance"] for s in stats]
    avg_distance = [s["avg_distance"] for s in stats]

    plt.figure()

    plt.plot(generations, best_distance, label="Best Distance")
    plt.plot(generations, avg_distance, label="Average Distance")

    plt.xlabel("Generation")
    plt.ylabel("Distance (meters)")

    plt.title(
        f"{config['controller']} Distance Evolution"
    )

    plt.legend()
    plt.grid(True)

    filepath = f"{plot_folder}/distance.png"

    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.show()

def plot_trajectories(
        stats,
        config,
        plot_folder,
):

    plt.figure(figsize=(8, 8))

    best_generation = max(
        stats,
        key=lambda s: s["best_fitness"]
    )["generation"]

    for generation_data in stats:

        trajectory = generation_data.get(
            "best_trajectory",
            []
        )

        if not trajectory:
            continue

        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]

        generation = generation_data["generation"]

        # destacar melhor geração
        if generation == best_generation:

            plt.plot(
                xs,
                ys,
                linewidth=3,
                label=f"Best Gen {generation}"
            )

        else:

            plt.plot(
                xs,
                ys,
                alpha=0.25
            )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title(
        f"{config['controller']} Trajectories"
    )

    plt.axis("equal")
    plt.grid(True)

    filepath = (
        f"{plot_folder}/"
        f"trajectories_overlay.png"
    )

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_trajectory_grid(
        stats,
        config,
        plot_folder,
):

    cols = 4

    rows = math.ceil(len(stats) / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(14, 10)
    )

    axes = axes.flatten()

    for ax, generation_data in zip(axes, stats):

        trajectory = generation_data.get(
            "best_trajectory",
            []
        )

        if trajectory:

            xs = [p[0] for p in trajectory]
            ys = [p[1] for p in trajectory]

            ax.plot(xs, ys)

        ax.set_title(
            f"Gen {generation_data['generation']}"
        )

        ax.set_aspect("equal")
        ax.grid(True)

    # esconder plots vazios
    for ax in axes[len(stats):]:
        ax.axis("off")

    fig.suptitle(
        f"{config['controller']} Trajectory Evolution"
    )

    filepath = (
        f"{plot_folder}/"
        f"trajectories_grid.png"
    )

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def main():

    stats_file = get_latest_stats_file()

    stats, config, timestamp = load_stats(stats_file)

    plot_folder = create_plot_folder(
        config["controller"],
        timestamp,
    )

    plot_fitness(stats, config, plot_folder)

    plot_distance(stats, config, plot_folder)

    plot_trajectories(
        stats,
        config,
        plot_folder,
    )

    plot_trajectory_grid(
        stats,
        config,
        plot_folder,
    )

    print(f"\nPlots saved to: {plot_folder}")


if __name__ == "__main__":
    main()