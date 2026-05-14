import json
from pathlib import Path


MIN_GENERATIONS = 5
MIN_DISTANCE = 1.0

stats_folder = Path(
    "results/training_stats"
)

best_folder = Path(
    "results/best_individuals"
)


def main():

    stats_files = list(
        stats_folder.glob("*.json")
    )

    removed_runs = 0

    for stats_file in stats_files:

        with open(stats_file, "r") as f:
            data = json.load(f)

        stats = data.get("stats", [])

        if not stats:

            print(
                f"Removing empty run: "
                f"{stats_file.name}"
            )

            remove_run(stats_file, data)

            removed_runs += 1
            continue

        generation_count = len(stats)

        last_generation = stats[-1]

        best_distance = last_generation.get(
            "best_distance",
            0.0
        )

        should_remove = (
            generation_count < MIN_GENERATIONS
            or
            best_distance < MIN_DISTANCE
        )

        if not should_remove:
            continue

        print(
            f"Removing run: {stats_file.name} | "
            f"Generations={generation_count} | "
            f"Best Distance={best_distance:.2f}"
        )

        remove_run(stats_file, data)

        removed_runs += 1

    print(
        f"\nRemoved {removed_runs} runs."
    )


def remove_run(stats_file, data):

    timestamp = data.get("timestamp")

    stats_file.unlink()

    matching_best_files = list(
        best_folder.glob(
            f"*{timestamp}*.json"
        )
    )

    for best_file in matching_best_files:

        print(
            f"Removing: "
            f"{best_file.name}"
        )

        best_file.unlink()


if __name__ == "__main__":
    main()