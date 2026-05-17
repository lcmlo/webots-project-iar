import json
import os
from pathlib import Path
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np


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

def plot_trajectory(
        stats,
        config,
        plot_folder,
):

    plt.figure(figsize=(12, 8))

    # =========================================================
    # Best generation overall
    # =========================================================

    best_generation_data = max(
        stats,
        key=lambda s: s["best_fitness"]
    )

    trajectory = best_generation_data.get(
        "best_trajectory",
        []
    )

    if not trajectory:
        return

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]

    generation = best_generation_data["generation"]

    best_fitness = best_generation_data["best_fitness"]

    best_distance = best_generation_data.get(
        "best_distance",
        0
    )

    collision_count = best_generation_data.get(
        "best_collision_count",
        "N/A"
    )

    time_on_line = best_generation_data.get(
        "best_time_on_line",
        "N/A"
    )

    # =========================================================
    # Best trajectory found during evolution
    # =========================================================

    plt.plot(
        xs,
        ys,
        linewidth=3,
        label=(
            f"Best Individual\n"
            f"Generation: {generation}\n"
            f"Fitness: {best_fitness:.2f}\n"
            f"Distance: {best_distance:.2f} m\n"
            f"Collisions: {collision_count}\n"
            f"Time on line: {time_on_line}"
        )
    )

    # =========================================================
    # Ideal track from Webots world
    # =========================================================

    ax = plt.gca()

    addIdealPath(ax)

    obstacles = best_generation_data.get(
        "obstacles",
        []
    )

    addObstacles(ax, obstacles)

    plt.plot(
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Ideal Arena Exploration Path"
    )

    plt.xlabel("Arena X Position")
    plt.ylabel("Arena Y Position")

    plt.title(
        f"{config['controller']} - "
        f"Best Overall Trajectory"
    )

    # =========================================================
    # Fixed arena limits
    # =========================================================

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.margins(0)

    # =========================================================
    # Legend outside plot
    # =========================================================

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0)
    )

    plt.grid(True)

    plt.tight_layout(
        rect=[0, 0, 0.78, 1]
    )

    filepath = (
        f"{plot_folder}/"
        f"best_trajectory.png"
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

    # =========================================================
    # Número fixo de plots
    # =========================================================

    FIXED_PLOTS = 9

    cols = 3
    rows = 3

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(14, 14)
    )

    axes = axes.flatten()

    # =========================================================
    # Dividir gerações igualmente
    # =========================================================

    generation_groups = np.array_split(
        stats,
        FIXED_PLOTS
    )

    # remover grupos vazios
    generation_groups = [
        group
        for group in generation_groups
        if len(group) > 0
    ]


    # =========================================================
    # Plot groups
    # =========================================================

    for ax, group in zip(axes, generation_groups):

        colors = plt.cm.tab10(
            np.linspace(0, 1, len(group))
        )

        # -----------------------------------------------------
        # Ideal exploration path
        # -----------------------------------------------------

        addIdealPath(ax)

        ax.plot(
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label="Ideal Path"
        )

        # -----------------------------------------------------
        # Plot all generations in this group
        # -----------------------------------------------------

        first_generation = group[0]["generation"]
        last_generation = group[-1]["generation"]

        single_generation_plot = (
                first_generation == last_generation
        )

        if single_generation_plot:
            obstacles = group[0].get(
                "obstacles",
                []
            )

            addObstacles(ax, obstacles)

        for generation_data, color in zip(group, colors):

            trajectory = generation_data.get(
                "best_trajectory",
                []
            )

            generation = generation_data["generation"]

            if not trajectory:
                continue

            xs = [p[0] for p in trajectory]
            ys = [p[1] for p in trajectory]

            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2,
                label=f"Gen {generation}"
            )

        # -----------------------------------------------------
        # Arena setup
        # -----------------------------------------------------

        if first_generation == last_generation:

            title = f"Gen {first_generation}"

        else:

            title = (
                f"Gen {first_generation} - "
                f"{last_generation}"
            )

        ax.set_title(title)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        ax.set_aspect("equal")

        ax.grid(True)

        ax.legend(
            fontsize=7,
            loc="upper right"
        )

    # esconder plots vazios
    for ax in axes[len(generation_groups):]:
        ax.axis("off")

    fig.suptitle(
        f"{config['controller']} "
        f"Trajectory Evolution"
    )

    plt.subplots_adjust(
        hspace=0.4,
        wspace=0.3,
        top=0.93
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


def addIdealPath(ax):
    track_width = 0.1

    # top
    ax.add_patch(
        Rectangle(
            (-0.9, 0.94),
            1.8,
            track_width,
            facecolor="orange",
            alpha=0.25,
            edgecolor="orange",
            linewidth=1
        )
    )

    # bottom
    ax.add_patch(
        Rectangle(
            (-0.9, -1.04),
            1.8,
            track_width,
            facecolor="orange",
            alpha=0.25,
            edgecolor="orange",
            linewidth=1
        )
    )

    # left
    ax.add_patch(
        Rectangle(
            (-1.11, -0.8),
            track_width,
            1.6,
            facecolor="orange",
            alpha=0.25,
            edgecolor="orange",
            linewidth=1
        )
    )

    # right
    ax.add_patch(
        Rectangle(
            (0.94, -0.9),
            track_width,
            1.8,
            facecolor="orange",
            alpha=0.25,
            edgecolor="orange",
            linewidth=1,
            label="Ideal Track"
        )
    )
def addObstacles(ax, obstacles):
    if not obstacles:
        return

    for obstacle in obstacles:

        x = obstacle["x"]
        y = obstacle["y"]

        size_x = obstacle["size_x"]
        size_y = obstacle["size_y"]

        rectangle = Rectangle(
            (
                x - size_x / 2,
                y - size_y / 2,
            ),
            size_x,
            size_y,
            facecolor="gray",
            edgecolor="black",
            linewidth=1,
            alpha=0.7,
            zorder=1,
        )

        ax.add_patch(rectangle)


def main():

    stats_file = get_latest_stats_file()

    stats, config, timestamp = load_stats(stats_file)

    plot_folder = create_plot_folder(
        config["controller"],
        timestamp,
    )

    plot_fitness(stats, config, plot_folder)

    plot_distance(stats, config, plot_folder)

    plot_trajectory(stats, config, plot_folder)

    plot_trajectory_grid(
        stats,
        config,
        plot_folder,
    )

    print(f"\nPlots saved to: {plot_folder}")


if __name__ == "__main__":
    main()