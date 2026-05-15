import json
import math
import random

import numpy as np
from controller import Supervisor
from pathlib import Path
from datetime import datetime
import os

from controllers import (
    BraitenbergController,
    SimpleANNController,
    AdvancedANNController,
)


# ============================================================
# Simulation / Evolution parameters
# ============================================================

TIME_STEP = 6.4

POPULATION_SIZE = 50
PARENTS_KEEP = 5
GENERATIONS = 50

MUTATION_RATE = 0.2
MUTATION_SIZE = 0.1

EVALUATION_TIME = 300  

RANGE = 5
MAX_SPEED = 9

EARLY_STOPPING = True
STAGNATION_LIMIT = 10
MIN_IMPROVEMENT = 0.1

SEED = random.randint(0, 1_000_000)
#SEED = 870207
#SEED = None

K_POINT_CROSSOVER = 2
TOURNAMENT_SIZE = 5

#Buffer de celulas ja visitadas na linha
MAX_BUFFER_SIZE = 150
CELL_SIZE = 0.1

#CONTROLLER_CLASS = BraitenbergController
CONTROLLER_CLASS = SimpleANNController
#CONTROLLER_CLASS = AdvancedANNController


MODE = "train"
# MODE = "test"

# testar o melhor de uma geracao especifica do ultimo controlador testado
# MODE = "test_generation"
# so usado se MODE = "test_generation"
GENERATION_TO_TEST = 5


# ============================================================
# Utility functions
# ============================================================

def random_orientation(rng):
    angle = rng.uniform(0, 2 * np.pi)

    # rodar à volta do eixo Z
    return [0, 0, 1, angle]

def random_position(rng):

    spawn_margin = 0.5 # # quanto maior o valor, menor a área de spawn

    min_x = -1.2 + spawn_margin
    max_x = 1.2 - spawn_margin

    min_y = -1.2 + spawn_margin
    max_y = 1.2 - spawn_margin

    x = rng.uniform(min_x, max_x)
    y = rng.uniform(min_y, max_y)

    # pequena altura acima do chão
    z = 0.02

    return [x, y, z]


# ============================================================
# Evolutionary Algorithm
# ============================================================

class Evolution:
    def __init__(self, controller_class):
        self.controller_class = controller_class
        self.genome_size = controller_class.GENOME_SIZE

        self.stats = []

        self.collision = False
        self.collision_count = 0
        self.time_on_line = 0

        # gerir ultimas posicoes na linha, para penalizar repiticoes
        # TODO fazer tune a MAX_BUFFER_SIZE e cell size
        self.recent_positions = {}
        self.position_cell_size = CELL_SIZE

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
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.spawn_rng = np.random.default_rng(SEED)
        self.root = self.supervisor.getRoot()
        self.children_field = self.root.getField("children")

        self.obstacle_defs = []


        self.best_individual_filename = (
            "results/best_individuals/"
            f"best_individual_"
            f"{self.controller_class.__name__}_"
            f"{self.timestamp}.json"
        )

        self.stats_filename = (
            "results/training_stats/"
            f"training_stats_"
            f"{self.controller_class.__name__}_"
            f"{self.timestamp}.json"
        )
        os.makedirs("results/best_individuals", exist_ok=True)
        os.makedirs("results/training_stats", exist_ok=True)

    # ============================================================
    # For controlling novelty on the line
    # ============================================================

    def get_position_cell(self, position):
        x = position[0]
        y = position[1]

        cell_x = int(x / self.position_cell_size)
        cell_y = int(y / self.position_cell_size)

        return cell_x, cell_y

    # ============================================================
    # Obstacles for ANN advanced
    # ============================================================

    def random_obstacle_position(self):

        radius = self.spawn_rng.uniform(0.6, 1.0)
        angle = self.spawn_rng.uniform(0, 2 * np.pi)

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        return [x, y, 0.1]

    def remove_obstacles(self):

        for obstacle_def in self.obstacle_defs:

            obstacle = self.supervisor.getFromDef(
                obstacle_def
            )

            if obstacle is not None:
                obstacle.remove()

        self.obstacle_defs.clear()

    def generate_obstacles(self):

        # Apenas usar obstáculos no controlador avançado
        if self.controller_class != AdvancedANNController:
            return

        obstacle_count = 8

        for i in range(obstacle_count):
            obstacle_def = f"OBSTACLE_{i}"

            position = self.random_obstacle_position()

            rotation = self.spawn_rng.uniform(
                0,
                2 * np.pi
            )

            size_x = self.spawn_rng.uniform(0.15, 0.3)
            size_y = self.spawn_rng.uniform(0.15, 0.3)

            obstacle_string = f"""
            DEF {obstacle_def} Solid {{
              translation {position[0]} {position[1]} {position[2]}
              rotation 0 0 1 {rotation}

              children [
                Shape {{
                  appearance Appearance {{
                    material Material {{
                      diffuseColor 1 1 1
                    }}
                  }}

                  geometry Box {{
                    size {size_x} {size_y} 0.2
                  }}
                }}
              ]

              boundingObject Box {{
                size {size_x} {size_y} 0.2
              }}

              physics Physics {{
                density 1000
              }}
            }}
            """

            self.children_field.importMFNodeFromString(
                -1,
                obstacle_string
            )

            self.obstacle_defs.append(
                obstacle_def
            )


    # ------------------------------------------------------------
    # Reset robot
    # ------------------------------------------------------------

    def reset(self):
        self.recent_positions.clear()

        random_rotation = random_orientation(self.spawn_rng)
        random_translation = random_position(self.spawn_rng)

        self.rotation_field.setSFRotation(random_rotation)
        self.translation_field.setSFVec3f(random_translation)

        self.robot_node.resetPhysics()

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        self.collision = False
        self.collision_count = 0
        self.time_on_line = 0
        self.__n = 0
        self.total_distance = 0.0

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
        if self.collision:
            self.collision_count += 1

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

        # checkar se ja passou aqui recentemente para poder penalizar
        revisited_recently = False

        if not ground_sensor_left and not ground_sensor_right:
            self.total_distance += step_distance
            self.time_on_line += 1

            position_cell = self.get_position_cell(current_position)
            revisited_recently = (position_cell in self.recent_positions)

            if not revisited_recently:
                self.recent_positions[position_cell] = True

                if len(self.recent_positions) > MAX_BUFFER_SIZE: #abrir espaco no buffer quando atinge max size
                    oldest_position = next(iter(self.recent_positions))
                    del self.recent_positions[oldest_position]
            

        self.prev_position = current_position
        self.__n += 1
        
        return self.get_step_fitness(sensors, step_distance, left_speed, right_speed, revisited_recently)

    # ------------------------------------------------------------
    # Run evolution
    # ------------------------------------------------------------

    def run(self):
        population = self.create_population()
        stagnation_counter = 0

        for generation in range(GENERATIONS):
            self.remove_obstacles()
            self.generate_obstacles()
            results = [
                self.evaluate_individual(individual)
                for individual in population
            ]

            fitnesses = np.array([r["fitness"] for r in results], dtype=float)
            distances = np.array([r["distance"] for r in results], dtype=float)
            trajectories = [r["trajectory"] for r in results]
            collision_counts = [
                r["collision_count"]
                for r in results
            ]

            time_on_line_values = [
                r["time_on_line"]
                for r in results
            ]

            best_index = int(np.argmax(fitnesses))
            best_fitness = float(fitnesses[best_index])
            best_distance = float(distances[best_index])
            best_trajectory = trajectories[best_index]
            best_collision_count = int(collision_counts[best_index])
            best_time_on_line = int(time_on_line_values[best_index])

            avg_fitness = float(np.mean(fitnesses))
            avg_distance = float(np.mean(distances))

            best_genome = population[best_index]

            if best_fitness > self.best_global_fitness + MIN_IMPROVEMENT:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if best_fitness > self.best_global_fitness:
                self.best_global_fitness = best_fitness
                self.best_global_genome = best_genome.copy()

                self.save_best_individual(
                    self.best_global_genome,
                    self.best_global_fitness,
                    generation,
                    best_distance,
                )

            self.remove_obstacles()
            
            print(
                f"Generation {generation}: "
                f"Best Fitness = {best_fitness:.2f}, "
                f"Avg Fitness = {avg_fitness:.2f}, "
                f"Best Distance = {best_distance:.2f}, "
                f"Avg Distance = {avg_distance:.2f}, "
                f"Collisions = {best_collision_count}, "
                f"Time On Line = {best_time_on_line}, "
                f"Global Best = {self.best_global_fitness:.2f}"
            )

            self.stats.append({
                "generation": generation,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_distance": best_distance,
                "avg_distance": avg_distance,
                "global_best_fitness": float(self.best_global_fitness),
                "best_genome": best_genome.tolist(),
                "best_trajectory": best_trajectory,
                "best_collision_count": best_collision_count,
                "best_time_on_line": best_time_on_line,
            })

            self.save_stats()

            if EARLY_STOPPING and stagnation_counter >= STAGNATION_LIMIT:
                print(
                    f"\nEarly stopping triggered after "
                    f"{STAGNATION_LIMIT} stagnant generations."
                )
                break

            parents = self.tournament_selection(population,fitnesses,)

            population = self.create_next_generation(parents)

        print("\nEvolution finished.")
        print(f"Best global fitness: {self.best_global_fitness:.2f}")
        print(f"Best global genome: {self.best_global_genome}")

    # ------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------

    def calculate_step_distance(self, previous_position, current_position):
        """
        Calculates distance travelled in the arena plane (X/Y).
        """

        return math.dist(
            [previous_position[0], previous_position[1]],
            [current_position[0], current_position[1]],
        )

    # ------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------

    def get_step_fitness(
        self,
        sensors,
        step_distance,
        left_speed,
        right_speed,
        revisited_recently
    ):
        fitness = 0.0
    
        ground_sensor_left = sensors["ground_left"]
        ground_sensor_right = sensors["ground_right"]
    
        on_line = (
            not ground_sensor_left and
            not ground_sensor_right
        )
    
        motion = (left_speed + right_speed) / 2.0
        normalized_motion = motion / MAX_SPEED
    
        # =========================================================
        # Recompensa principal:
        # distancia percorrida na linha
        # =========================================================
        # dar pontos por estar na linha
        # nao permitir estar parado ou frente e tras
        if on_line:
            if revisited_recently:
                fitness -= step_distance * 50.0
            else:
                fitness += step_distance * 250.0

    
        else:
            fitness -= 1.0
    
        # =========================================================
        # Penalizar marcha atras
        # =========================================================
    
        if left_speed < 0 and right_speed < 0:
            fitness -= abs(normalized_motion) * 2.0
    
        # =========================================================
        # Penalizar colisoes
        # =========================================================
    
        if self.collision:
            fitness -= 10.0
    
        return fitness

    # ------------------------------------------------------------
    # Population
    # ------------------------------------------------------------

    def create_population(self):
        population = []
        while len(population) < POPULATION_SIZE:
            genome = np.random.uniform(-RANGE,RANGE,self.genome_size)
            population.append(genome)
        return population


    # ------------------------------------------------------------
    # Individual evaluation
    # ------------------------------------------------------------

    def evaluate_individual(self, genome):
        self.reset()

        active_controller = self.controller_class(genome)

        fitness = 0.0

        start_time = self.supervisor.getTime()

        trajectory = []

        while self.supervisor.getTime() - start_time < EVALUATION_TIME:
            step_fitness = self.runStep(active_controller)
            fitness += step_fitness
            current_position = self.robot_node.getPosition()

            trajectory.append([
                current_position[0],
                current_position[1],
            ])

        return {
            "fitness": fitness,
            "distance": self.total_distance,
            "trajectory": trajectory,
            "collision_count": self.collision_count,
            "time_on_line": self.time_on_line,
        }

    # ------------------------------------------------------------
    # Next generation
    # ------------------------------------------------------------

    def create_next_generation(self, parents):

        next_population = []

        # Elitism
        for parent in parents:
            next_population.append(parent.copy())

        while len(next_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)

            child = self.crossover(parent1, parent2)
            child = self.mutate(child)

            next_population.append(child)

        return next_population

    def tournament_selection(
            self,
            population,
            fitnesses,
    ):

        selected_parents = []

        for _ in range(PARENTS_KEEP):
            tournament_indices = random.sample(
                range(len(population)),
                TOURNAMENT_SIZE
            )

            best_index = max(
                tournament_indices,
                key=lambda i: fitnesses[i]
            )

            selected_parents.append(
                population[best_index].copy()
            )

        return selected_parents

    # ------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------

    def crossover(self, parent1, parent2):

        # =========================================================
        # K-Point crossover
        # =========================================================

        max_points = self.genome_size - 1

        k = min(K_POINT_CROSSOVER, max_points)

        crossover_points = sorted(
            random.sample(
                range(1, self.genome_size),
                k
            )
        )

        child = []
        current_parent = parent1

        previous_point = 0

        for point in crossover_points:
            child.extend(
                current_parent[previous_point:point]
            )

            current_parent = (
                parent2
                if current_parent is parent1
                else parent1
            )

            previous_point = point

        child.extend(
            current_parent[previous_point:]
        )

        return np.array(child)

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
    # Save CONFIG
    # ------------------------------------------------------------
    def get_experiment_config(self):

        return {
            "controller": self.controller_class.__name__,
            "population_size": POPULATION_SIZE,
            "parents_keep": PARENTS_KEEP,
            "generations": GENERATIONS,
            "mutation_rate": MUTATION_RATE,
            "mutation_size": MUTATION_SIZE,
            "evaluation_time": EVALUATION_TIME,
            "range": RANGE,
            "max_speed": MAX_SPEED,
            "seed": SEED,

            "k_point_crossover": K_POINT_CROSSOVER,
            "tournament_size": TOURNAMENT_SIZE,
        }

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
            "timestamp": self.timestamp,
        }

        with open(self.best_individual_filename, "w") as f:
            json.dump(best_individual, f, indent=4)

    # ------------------------------------------------------------
    # Save stats
    # ------------------------------------------------------------

    def save_stats(self):

        data = {
            "config": self.get_experiment_config(),
            "stats": self.stats,
            "timestamp": self.timestamp,
        }

        with open(self.stats_filename, "w") as f:
            json.dump(data, f, indent=4)



# ============================================================
# Test best individual
# ============================================================

def test_best_individual(controller_class):

    folder = "results/best_individuals"

    matching_files = [
        f for f in os.listdir(folder)
        if (
            controller_class.__name__ in f and
            f.endswith(".json")
        )
    ]

    if not matching_files:
        raise FileNotFoundError(
            f"No saved individuals found for "
            f"{controller_class.__name__}"
        )

    latest_file = max(
        matching_files,
        key=lambda f: os.path.getmtime(
            os.path.join(folder, f)
        )
    )

    filepath = os.path.join(folder, latest_file)

    with open(filepath, "r") as f:
        best_individual = json.load(f)

    genome = np.array(
        best_individual["genome"],
        dtype=float
    )

    fitness = best_individual["fitness"]

    print(f"\nTesting latest best individual")
    print(f"File: {latest_file}")
    print(f"Controller: {controller_class.__name__}")
    print(f"Fitness: {fitness:.2f}")

    evolution = Evolution(controller_class)

    evolution.reset()

    active_controller = controller_class(genome)

    while True:
        evolution.runStep(active_controller)

# ============================================================
# Test best from generation
# ============================================================

def test_generation(
        controller_class,
        generation,
):

    # =========================================================
    # Carrega o ficheiro mais recente do controlador
    # =========================================================

    folder = Path("results/training_stats")

    matching_files = [
        f for f in folder.glob("*.json")
        if controller_class.__name__ in f.name
    ]

    if not matching_files:
        raise FileNotFoundError(
            f"No training stats found for "
            f"{controller_class.__name__}"
        )

    latest_file = max(
        matching_files,
        key=lambda f: f.stat().st_mtime
    )

    with open(latest_file, "r") as f:
        data = json.load(f)

    stats = data["stats"]

    # =========================================================
    # Procurar geração pretendida
    # =========================================================

    matching_generation = None

    for generation_data in stats:

        if generation_data["generation"] == generation:
            matching_generation = generation_data
            break

    if matching_generation is None:
        raise ValueError(
            f"Generation {generation} not found."
        )

    genome = np.array(
        matching_generation["best_genome"],
        dtype=float
    )

    fitness = matching_generation["best_fitness"]

    print(f"\nTesting generation {generation}")
    print(f"Fitness: {fitness:.2f}")

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

    elif MODE == "test":

        test_best_individual(
            CONTROLLER_CLASS
        )

    elif MODE == "test_generation":

        test_generation(
            controller_class=CONTROLLER_CLASS,
            generation=GENERATION_TO_TEST,
        )

    else:
        raise ValueError(
            f"Invalid MODE: {MODE}"
        )


if __name__ == "__main__":
    main()