import json
import matplotlib.pyplot as plt


def load_stats(filename="training_stats.json"):
    with open(filename, "r") as f:
        return json.load(f)


def plot_fitness(stats):

    generations = [s["generation"] for s in stats]
    best_fitness = [s["best_fitness"] for s in stats]
    avg_fitness = [s["avg_fitness"] for s in stats]

    plt.figure()

    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(generations, avg_fitness, label="Average Fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Braitenberg Fitness Convergence")

    plt.legend()
    plt.grid(True)

    plt.show()


def plot_distance(stats):

    generations = [s["generation"] for s in stats]
    best_distance = [s["best_distance"] for s in stats]
    avg_distance = [s["avg_distance"] for s in stats]

    plt.figure()

    plt.plot(generations, best_distance, label="Best Distance")
    plt.plot(generations, avg_distance, label="Average Distance")

    plt.xlabel("Generation")
    plt.ylabel("Distance (meters)")
    plt.title("Braitenberg Distance Evolution")

    plt.legend()
    plt.grid(True)

    plt.show()


def main():

    stats = load_stats()

    plot_fitness(stats)
    plot_distance(stats)


if __name__ == "__main__":
    main()