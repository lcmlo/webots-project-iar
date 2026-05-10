import json
import math
import random

import numpy as np
from controller import Supervisor

from controllers import (
    BraitenbergController,
    SimpleANNController,
    AdvancedANNController,
)


# ============================================================
# Simulation / Evolution parameters
# ============================================================

TIME_STEP = 5

POPULATION_SIZE = 25
PARENTS_KEEP = 5
GENERATIONS = 50

MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05

EVALUATION_TIME = 60  

RANGE = 2
MAX_SPEED = 9


# CONTROLLER_CLASS = BraitenbergController
# CONTROLLER_CLASS = SimpleANNController
CONTROLLER_CLASS = BraitenbergController
# CONTROLLER_CLASS = AdvancedANNController


MODE = "train"
# MODE = "test"


# ============================================================
# Utility functions
# ============================================================

def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return [0, 0, 1, angle]


def random_position(min_radius, max_radius, z):
    radius = np.random.uniform(min_radius, max_radius)
    angle = np.random.uniform(0, 2 * np.pi)

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    return [x, y, z]


# ============================================================
# Evolutionary Algorithm
# ============================================================

class Evolution:
    def __init__(self, controller_class):
        self.controller_class = controller_class
        self.genome_size = controller_class.GENOME_SIZE

        self.stats = []

        self.evaluation_start_time = 0
        self.collision = False

        self.best_global_fitness = -float("inf")
        self.best_global_genome = None

        # Supervisor to reset robot position
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()

        self.robot_node = self.supervisor.getFromDef("ROBOT")

        if self.robot_node is None:
            raise RuntimeError(
                "Could not find robot node with DEF ROBOT. "
                "Make sure your Thymio robot has DEF ROBOT in the Webots world."
            )

        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep() * TIME_STEP)

        # Motors
        self.left_motor = self.supervisor.getDevice("motor.left")
        self.right_motor = self.supervisor.getDevice("motor.right")

        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        # Horizontal proximity sensors
        self.__ir_0 = self.supervisor.getDevice("prox.horizontal.0")
        self.__ir_1 = self.supervisor.getDevice("prox.horizontal.1")
        self.__ir_2 = self.supervisor.getDevice("prox.horizontal.2")
        self.__ir_3 = self.supervisor.getDevice("prox.horizontal.3")
        self.__ir_4 = self.supervisor.getDevice("prox.horizontal.4")
        self.__ir_5 = self.supervisor.getDevice("prox.horizontal.5")
        self.__ir_6 = self.supervisor.getDevice("prox.horizontal.6")

        # Ground sensors
        self.__ground_0 = self.supervisor.getDevice("prox.ground.0")
        self.__ground_1 = self.supervisor.getDevice("prox.ground.1")

        self.ground_sensors = [
            self.__ground_0,
            self.__ground_1,
        ]

        all_sensors = [
            self.__ir_0,
            self.__ir_1,
            self.__ir_2,
            self.__ir_3,
            self.__ir_4,
            self.__ir_5,
            self.__ir_6,
            self.__ground_0,
            self.__ground_1,
        ]

        for sensor in all_sensors:
            sensor.enable(self.timestep)

        self.__n = 0
        self.total_distance = 0.0
        self.prev_position = self.robot_node.getPosition()

    # ------------------------------------------------------------
    # Reset robot
    # ------------------------------------------------------------

    def reset(self):
        random_rotation = random_orientation()

        self.rotation_field.setSFRotation(random_rotation)

        self.translation_field.setSFVec3f([0, 0, 0])

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        self.collision = False
        self.__n = 0
        self.total_distance = 0.0

        self.supervisor.simulationResetPhysics()
        self.supervisor.step(self.timestep)

        self.prev_position = self.robot_node.getPosition()

    # ------------------------------------------------------------
    # Sensor reading
    # ------------------------------------------------------------

    def get_sensor_data(self):
        ground_left_raw = self.ground_sensors[0].getValue()
        ground_right_raw = self.ground_sensors[1].getValue()

        ground_left = (ground_left_raw / 1023 - 0.6) / 0.2 > 0.3
        ground_right = (ground_right_raw / 1023 - 0.6) / 0.2 > 0.3

        return {
            "ground_left": ground_left,
            "ground_right": ground_right,

            "ground_left_raw": ground_left_raw,
            "ground_right_raw": ground_right_raw,

            "prox_left": self.__ir_0.getValue(),
            "prox_center": self.__ir_2.getValue(),
            "prox_right": self.__ir_4.getValue(),
        }

    # ------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------

    def detect_collision(self):
        return bool(
            self.__n > 10 and (
                self.__ir_0.getValue() > 4300 or
                self.__ir_1.getValue() > 4300 or
                self.__ir_2.getValue() > 4300 or
                self.__ir_3.getValue() > 4300 or
                self.__ir_4.getValue() > 4300 or
                self.__ir_5.getValue() > 4300 or
                self.__ir_6.getValue() > 4300
            )
        )

    # ------------------------------------------------------------
    # One simulation step
    # ------------------------------------------------------------

    def runStep(self, active_controller):
        sensors = self.get_sensor_data()

        self.collision = self.detect_collision()

        left_speed, right_speed = active_controller.compute_speeds(sensors)

        left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
        right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        self.supervisor.step(self.timestep)

        current_position = self.robot_node.getPosition()

        step_distance = self.calculate_step_distance(
            self.prev_position,
            current_position,
        )

        ground_sensor_left = sensors["ground_left"]
        ground_sensor_right = sensors["ground_right"]

        if not ground_sensor_left and not ground_sensor_right:
            self.total_distance += step_distance

        self.prev_position = current_position
        self.__n += 1

        return self.get_step_fitness(sensors, step_distance)

    # ------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------

    def calculate_step_distance(self, previous_position, current_position):
        """
        Calculates distance travelled in the horizontal plane.

        Important:
        If your Webots world uses X/Z as the floor plane, keep [0] and [2].
        If it uses X/Y as the floor plane, change this to [0] and [1].
        """

        return math.dist(
            [previous_position[0], previous_position[2]],
            [current_position[0], current_position[2]],
        )

    # ------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------

    def get_step_fitness(self, sensors, step_distance):
        fitness = 0.0

        ground_sensor_left = sensors["ground_left"]
        ground_sensor_right = sensors["ground_right"]

        on_line_or_path = not ground_sensor_left and not ground_sensor_right

        if on_line_or_path:
            fitness += 1.0
            fitness += step_distance * 100.0
        else:
            fitness -= 0.05

        if self.collision:
            fitness -= 10.0

        return fitness

    # ------------------------------------------------------------
    # Population
    # ------------------------------------------------------------

    def create_population(self):
        return [
            np.random.uniform(-RANGE, RANGE, self.genome_size)
            for _ in range(POPULATION_SIZE)
        ]

    # ------------------------------------------------------------
    # Individual evaluation
    # ------------------------------------------------------------

    def evaluate_individual(self, genome):
        self.reset()

        active_controller = self.controller_class(genome)

        fitness = 0.0

        start_time = self.supervisor.getTime()

        while (
            self.supervisor.getTime() - start_time < EVALUATION_TIME
            and not self.collision
        ):
            step_fitness = self.runStep(active_controller)
            fitness += step_fitness

        if self.collision:
            fitness -= 20.0

        return {
            "fitness": fitness,
            "distance": self.total_distance,
        }

    # ------------------------------------------------------------
    # Run evolution
    # ------------------------------------------------------------

    def run(self):
        population = self.create_population()

        for generation in range(GENERATIONS):
            results = [
                self.evaluate_individual(individual)
                for individual in population
            ]

            fitnesses = np.array([r["fitness"] for r in results], dtype=float)
            distances = np.array([r["distance"] for r in results], dtype=float)

            best_index = int(np.argmax(fitnesses))
            best_fitness = float(fitnesses[best_index])
            best_distance = float(distances[best_index])

            avg_fitness = float(np.mean(fitnesses))
            avg_distance = float(np.mean(distances))

            best_genome = population[best_index]

            if best_fitness > self.best_global_fitness:
                self.best_global_fitness = best_fitness
                self.best_global_genome = best_genome.copy()

                self.save_best_individual(
                    self.best_global_genome,
                    self.best_global_fitness,
                    generation,
                    best_distance,
                )

            print(
                f"Generation {generation}: "
                f"Best Fitness = {best_fitness:.2f}, "
                f"Avg Fitness = {avg_fitness:.2f}, "
                f"Best Distance = {best_distance:.2f}, "
                f"Avg Distance = {avg_distance:.2f}, "
                f"Global Best = {self.best_global_fitness:.2f}"
            )

            self.stats.append({
                "generation": generation,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_distance": best_distance,
                "avg_distance": avg_distance,
                "global_best_fitness": float(self.best_global_fitness),
            })

            self.save_stats()

            parents_indices = np.argsort(fitnesses)[-PARENTS_KEEP:]
            parents = [population[i].copy() for i in parents_indices]

            population = self.create_next_generation(parents)

        print("\nEvolution finished.")
        print(f"Best global fitness: {self.best_global_fitness:.2f}")
        print(f"Best global genome: {self.best_global_genome}")

    # ------------------------------------------------------------
    # Next generation
    # ------------------------------------------------------------

    def create_next_generation(self, parents):
        next_population = []

        # Elitism: keep best parents directly
        for parent in parents:
            next_population.append(parent.copy())

        while len(next_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)

            child = self.crossover(parent1, parent2)
            child = self.mutate(child)

            next_population.append(child)

        return next_population

    # ------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.genome_size - 1)

        child = np.concatenate((
            parent1[:crossover_point],
            parent2[crossover_point:],
        ))

        return child

    # ------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------

    def mutate(self, genome):
        mutated = genome.copy()

        for i in range(len(mutated)):
            if random.random() < MUTATION_RATE:
                mutated[i] += np.random.uniform(-MUTATION_SIZE, MUTATION_SIZE)
                mutated[i] = max(min(mutated[i], RANGE), -RANGE)

        return mutated

    # ------------------------------------------------------------
    # Save best individual
    # ------------------------------------------------------------

    def save_best_individual(self, genome, fitness, generation, distance):
        best_individual = {
            "controller": self.controller_class.__name__,
            "genome_size": self.genome_size,
            "genome": genome.tolist(),
            "fitness": float(fitness),
            "generation": int(generation),
            "distance": float(distance),
        }

        with open("best_individual.json", "w") as f:
            json.dump(best_individual, f, indent=4)

    # ------------------------------------------------------------
    # Save stats
    # ------------------------------------------------------------

    def save_stats(self):
        with open("training_stats.json", "w") as f:
            json.dump(self.stats, f, indent=4)


# ============================================================
# Test best individual
# ============================================================

def test_best_individual(controller_class):
    with open("best_individual.json", "r") as f:
        best_individual = json.load(f)

    genome = np.array(best_individual["genome"], dtype=float)
    fitness = best_individual["fitness"]

    print(f"Testing best individual")
    print(f"Controller: {controller_class.__name__}")
    print(f"Fitness: {fitness}")
    print(f"Genome: {genome}")

    evolution = Evolution(controller_class)
    evolution.reset()

    active_controller = controller_class(genome)

    while True:
        evolution.runStep(active_controller)


# ============================================================
# Main
# ============================================================

def main():
    if MODE == "train":
        evolution = Evolution(CONTROLLER_CLASS)
        evolution.run()

        # Useful if running Webots in batch/headless mode.
        # Comment this if you want Webots to stay open after training.
        # evolution.supervisor.simulationQuit(0)

    elif MODE == "test":
        test_best_individual(CONTROLLER_CLASS)

    else:
        raise ValueError(f"Invalid MODE: {MODE}")


if __name__ == "__main__":
    main()