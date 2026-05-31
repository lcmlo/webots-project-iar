import json
import os
from pathlib import Path
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np

MODE = "single_controller"
#MODE = "compare_latest_runs"

#caso nao seja o mais recente e quiserem especificar
TEST_CONTROLLER_NAME = "AdvancedANNController"
# se none vai buscar o mais recente
TEST_TIMESTAMP = None
# TEST_TIMESTAMP = "20260530_143914"

COMPARE_LAST_N = 3

def get_latest_stats_file():

    return str(
        get_matching_stats_files()[0]
    )

def get_matching_stats_files():

    stats_folder = Path(
        "results/training_stats"
    )

    files = list(
        stats_folder.glob("*.json")
    )

    matching_files = []

    for f in files:

        filename = f.name

        if (
                TEST_CONTROLLER_NAME is not None
                and
                TEST_CONTROLLER_NAME not in filename
        ):
            continue

        if (
                TEST_TIMESTAMP is not None
                and
                TEST_TIMESTAMP not in filename
        ):
            continue

        matching_files.append(f)

    if not matching_files:
        raise FileNotFoundError(
            "No matching stats files found."
        )

    matching_files.sort(
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )

    return matching_files

def compare_recent_experiments():

    compare_recent_fitness()

    compare_recent_distances()

    compare_recent_trajectories()

def compare_recent_fitness():

    files = get_matching_stats_files()

    files = files[:COMPARE_LAST_N]

    plt.figure(figsize=(12, 8))

    for file in reversed(files):

        stats, config, timestamp = load_stats(file)

        generations = [
            s["generation"]
            for s in stats
        ]

        best_fitness = [
            s["best_fitness"]
            for s in stats
        ]

        best_generation_data = max(
            stats,
            key=lambda s: s["best_fitness"]
        )

        best_gen = best_generation_data["generation"]

        total_gens = stats[-1]["generation"]

        label = f"{config['controller']}_{timestamp}"

        if "layers" in config:
            label += (
                f" | {config['layers']}"
            )

        label += (
            f" | BestGen "
            f"{best_gen}/{total_gens}"
        )

        plt.plot(
            generations,
            best_fitness,
            linewidth=2,
            label=label
        )

    plt.xlabel("Generation")

    plt.ylabel("Best Fitness")

    plt.title(
        f"Comparison of Last "
        f"{COMPARE_LAST_N} Experiments"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

def compare_recent_distances():

    files = get_matching_stats_files()

    files = files[:COMPARE_LAST_N]

    plt.figure(figsize=(12, 8))

    for file in reversed(files):

        stats, config, timestamp = load_stats(file)

        generations = [
            s["generation"]
            for s in stats
        ]

        best_distance = [
            s["best_distance"]
            for s in stats
        ]

        best_generation_data = max(
            stats,
            key=lambda s: s["best_fitness"]
        )

        best_gen = best_generation_data["generation"]

        total_gens = stats[-1]["generation"]

        label = f"{config['controller']}_{timestamp}"

        if "layers" in config:
            label += (
                f" | Layers {config['layers']}"
            )

        label += (
            f" | BestGen "
            f"{best_gen}/{total_gens}"
        )

        plt.plot(
            generations,
            best_distance,
            linewidth=2,
            label=label
        )

    plt.xlabel("Generation")

    plt.ylabel("Best Distance")

    plt.title(
        f"Distance Comparison "
        f"({COMPARE_LAST_N} Runs)"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

def compare_recent_trajectories():

    files = get_matching_stats_files()

    files = files[:COMPARE_LAST_N]

    cols = 2

    rows = int(
        np.ceil(len(files) / cols)
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(14, 7 * rows)
    )

    axes = np.array(axes).flatten()

    for ax, file in zip(axes, reversed(files)):

        stats, config, timestamp = load_stats(file)

        best_generation = max(
            stats,
            key=lambda s: s["best_fitness"]
        )

        trajectory = best_generation.get(
            "best_trajectory",
            []
        )

        if not trajectory:
            ax.axis("off")
            continue

        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]

        # =====================================================
        # Arena
        # =====================================================

        addIdealPath(ax)

        obstacles = best_generation.get(
            "obstacles",
            []
        )

        addObstacles(
            ax,
            obstacles
        )

        # =====================================================
        # Trajectory
        # =====================================================

        ax.plot(
            xs,
            ys,
            linewidth=3,
            alpha=0.9,
        )

        # start point
        ax.scatter(
            xs[0],
            ys[0],
            s=80,
            marker="o",
            zorder=5
        )

        # end point
        ax.scatter(
            xs[-1],
            ys[-1],
            s=80,
            marker="x",
            linewidths=2,
            zorder=5
        )

        # =====================================================
        # Labels
        # =====================================================

        label = config["controller"]

        if "layers" in config:
            label += (
                f"\nLayers: "
                f"{config['layers']}"
            )

        label = (
            f"{config['controller']}_{timestamp}"
        )

        if "layers" in config:
            label += (
                f"\nLayers: "
                f"{config['layers']}"
            )

        total_gens = stats[-1]["generation"]

        label += (
            f"\nBest Gen: "
            f"{best_generation['generation']}"
            f"/{total_gens}"
        )

        label += (
            f"\nFitness: "
            f"{best_generation['best_fitness']:.0f}"
        )

        label += (
            f"\nDistance: "
            f"{best_generation['best_distance']:.2f}"
        )

        population_size = config.get(
            "population_size",
            "N/A"
        )

        label += (
            f"\nPop size: "
            f"{population_size}"
        )

        evaluation_time = config.get(
            "evaluation_time",
            "N/A"
        )

        label += (
            f"\nEval time: "
            f"{evaluation_time}"
        )

        ax.set_title(label)

        # =====================================================
        # Arena setup
        # =====================================================

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        ax.set_aspect("equal")

        ax.grid(True)

    # esconder plots vazios
    for ax in axes[len(files):]:
        ax.axis("off")

    fig.suptitle(
        f"Best Trajectories Comparison "
        f"({COMPARE_LAST_N} Runs)"
    )

    plt.tight_layout()

    plt.show()

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


def plot_fitness(stats, config,timestamp, plot_folder):

    generations = [s["generation"] for s in stats]

    best_fitness = [s["best_fitness"] for s in stats]
    avg_fitness = [s["avg_fitness"] for s in stats]

    plt.figure()

    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(generations, avg_fitness, label="Average Fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.title(
        f"{config['controller']}_{timestamp} Fitness Convergence "
        f"{timestamp}"
    )

    plt.legend()
    plt.grid(True)

    filepath = f"{plot_folder}/fitness.png"

    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_distance(stats, config,timestamp, plot_folder):

    generations = [s["generation"] for s in stats]

    best_distance = [s["best_distance"] for s in stats]
    avg_distance = [s["avg_distance"] for s in stats]

    plt.figure()

    plt.plot(generations, best_distance, label="Best Distance")
    plt.plot(generations, avg_distance, label="Average Distance")

    plt.xlabel("Generation")
    plt.ylabel("Distance (meters)")

    plt.title(
        f"{config['controller']}_{timestamp} Distance Evolution "
    )

    plt.legend()
    plt.grid(True)

    filepath = f"{plot_folder}/distance.png"

    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.show()

def plot_trajectory(
        stats,
        config,
        timestamp,
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
        f"{config['controller']}_{timestamp} - "
        f"Best Overall Trajectory "
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
        timestamp,
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
        f"{config['controller']}_{timestamp} "
        f"Trajectory Evolution "
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

    if MODE == "single_controller":

        stats_file = (
            get_latest_stats_file()
        )

        stats, config, timestamp = (
            load_stats(stats_file)
        )

        plot_folder = create_plot_folder(
            config["controller"],
            timestamp,
        )

        plot_fitness(
            stats,
            config,
            timestamp,
            plot_folder,
        )

        plot_distance(
            stats,
            config,
            timestamp,
            plot_folder,
        )

        plot_trajectory(
            stats,
            config,
            timestamp,
            plot_folder,
        )

        plot_trajectory_grid(
            stats,
            config,
            timestamp,
            plot_folder,
        )

    elif MODE == "compare_latest_runs":

        compare_recent_experiments()


if __name__ == "__main__":
    main()