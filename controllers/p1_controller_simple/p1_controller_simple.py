import json
import math
import random
from collections import OrderedDict
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

MUTATION_RATE = 0.15
MUTATION_SIZE = 0.05

EVALUATION_TIME = 100
EVALUATION_RUNS = 3

TOURNAMENT_SIZE = 3
K_POINT_CROSSOVER = 3

RANGE = 5
MAX_SPEED = 9.53

# Buffer de celulas ja visitadas na linha
LINE_BUFFER_SIZE = 300 # 6m de celulas
CELL_SIZE = 0.02

CONTROLLER_CLASS = BraitenbergController
#CONTROLLER_CLASS = SimpleANNController
#CONTROLLER_CLASS = AdvancedANNController

# para reproduzir exatamente as condicoes de treino
# usar o numero da seed do treino
# para averiguar se aprendeu mesmo deixar random ou None
SEED = random.randint(0, 1_000_000)
#SEED = 454733
# SEED = None

EARLY_STOPPING = True
STAGNATION_LIMIT = 10
MIN_IMPROVEMENT_PERCENT = 0.005

#para validar melhor a fitness enquanto treina, metricas por individuo e nao so por geracao
DEBUG_INDIVIDUALS = True
MODE = "train"

CONTINUE_TRAINING = False

# se NONE usa o mais recente desse CONTROLLER_CLASS
# o timestamp vai buscar esse especifico desse CONTROLLER_CLASS
CONTROLLER_TIMESTAMP = None
#CONTROLLER_TIMESTAMP = "20260604_190737"

# ============================================================
# TESTS
# ============================================================
#MODE = "test"

#MODE = "test_generation" # testar o melhor de uma geracao especifica do ultimo controlador testado
GENERATION_TO_TEST = 49 # so usado se MODE = "test_generation"

#Para as metricas dos testes, esta em metros #TODO maybe meter n max de colisoes
SUCCESS_DISTANCE = 2.0
# ============================================================
# Utility functions
# ============================================================

def random_orientation(rng):
    angle = rng.uniform(0, 2 * np.pi)

    # rodar à volta do eixo Z
    return [0, 0, 1, angle]


def random_position(rng):
    spawn_margin = 0.7  # # quanto maior o valor, menor a área de spawn

    min_x = -1.2 + spawn_margin
    max_x = 1.2 - spawn_margin

    min_y = -1.2 + spawn_margin
    max_y = 1.2 - spawn_margin

    x = rng.uniform(min_x, max_x)
    y = rng.uniform(min_y, max_y)

    # pequena altura acima do chão
    z = 0.001

    return [x, y, z]


# ============================================================
# Evolutionary Algorithm
# ============================================================

class Evolution:
    def __init__(self, controller_class):
        self.current_obstacles = None
        self.current_spawn_rotation = None
        self.current_spawn_translation = None
        self.controller_class = controller_class
        self.genome_size = controller_class.GENOME_SIZE

        self.stats = []

        self.collision = False
        self.collision_count = 0
        self.time_on_line = 0
        self.reverse_steps = 0
        self.has_found_line = False
        self.steps_without_line = 0
        self.same_cell_steps = 0
        self.previous_track_cell = None
        self.previous_distance_cell = None

        self.debug_distance_reward = 0.0
        self.debug_revisit_penalty = 0.0
        self.debug_off_line_penalty = 0.0
        self.debug_collision_penalty = 0.0
        self.debug_lost_line_penalty = 0.0


        self.line_recent_positions = OrderedDict()  # guarda so ultimas posicoes na linha
        self.position_cell_size = CELL_SIZE

        self.line_cells_visited = 0
        self.line_cells_revisited = 0

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
        self.total_distance = 0.0 #com ambos na linha
        self.distance_any_sensor = 0.0
        self.prev_position = self.robot_node.getPosition()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.spawn_rng = np.random.default_rng(SEED)
        self.root = self.supervisor.getRoot()
        self.children_field = self.root.getField("children")

        self.obstacle_defs = []
        self.generation_scenarios = []

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
    # TEST BEST GENOME
    # ============================================================
    def evaluate_genome_benchmark(
            self,
            genome,
            episodes=30,
    ):
        results = []

        for episode in range(episodes):

            self.reset(new_spawn=True)

            self.remove_obstacles()
            self.generate_obstacles()

            controller = self.controller_class(genome)

            start_time = self.supervisor.getTime()

            while (
                    self.supervisor.getTime() - start_time
                    < EVALUATION_TIME
            ):
                self.runStep(controller)

            results.append({
                "distance with 2 sensors": self.total_distance,
                "distance with 1-2 sensors": self.distance_any_sensor,
                "collisions": self.collision_count,
                "time_on_line": self.time_on_line,
            })

        distances_2_sensors = [
            r["distance with 2 sensors"]
            for r in results
        ]

        distances_any_sensor = [
            r["distance with 1-2 sensors"]
            for r in results
        ]

        collisions = [
            r["collisions"]
            for r in results
        ]

        successful_runs = sum(
            d >= SUCCESS_DISTANCE
            for d in distances_2_sensors
        )

        success_rate = (
                               successful_runs
                               / episodes
                       ) * 100

        print("\n========== BENCHMARK ==========")
        print(f"Episodes: {episodes}")

        print(
            f"Avg Distance (2 sensors): "
            f"{np.mean(distances_2_sensors):.2f}"
        )

        print(
            f"Median Distance: "
            f"{np.median(distances_2_sensors):.2f}"
        )

        print(
            f"Distance Std: {np.std(distances_2_sensors):.2f}"
        )

        print(
            f"Distance Max: {np.max(distances_2_sensors):.2f}"
        )

        print(
            f"Distance Min: {np.min(distances_2_sensors):.2f}"
        )

        print(
            f"Avg Distance (>=1 sensor): "
            f"{np.mean(distances_any_sensor):.2f}"
        )
        print(
            f"Median Distance (>=1 sensor): "
            f"{np.median(distances_any_sensor):.2f}"
        )

        print(
            f"Distance Std (>=1 sensor): "
            f"{np.std(distances_any_sensor):.2f}"
        )

        print(
            f"Distance Max (>=1 sensor): "
            f"{np.max(distances_any_sensor):.2f}"
        )

        print(
            f"Distance Min (>=1 sensor): "
            f"{np.min(distances_any_sensor):.2f}"
        )

        print(
            f"Collisions Avg: {np.mean(collisions):.2f}"
        )

        print(
            f"Successful Runs: "
            f"{successful_runs}/{episodes}"
        )

        print(
            f"Success Rate (>={SUCCESS_DISTANCE}m): "
            f"{success_rate:.1f}%"
        )



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

    def point_to_track_distance(self, x, y):

        track_segments = [

            # topo
            ((-0.8, 1.0), (0.8, 1.0)),

            # direita
            ((1.0, 0.8), (1.0, -0.8)),

            # baixo
            ((0.8, -1.0), (-0.8, -1.0)),

            # esquerda
            ((-1.0, -0.8), (-1.0, 0.8)),
        ]

        min_distance = float("inf")

        for start, end in track_segments:

            x1, y1 = start
            x2, y2 = end

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                distance = math.dist((x, y), (x1, y1))

            else:
                t = (
                        ((x - x1) * dx + (y - y1) * dy)
                        / (dx * dx + dy * dy)
                )

                t = max(0.0, min(1.0, t))

                proj_x = x1 + t * dx
                proj_y = y1 + t * dy

                distance = math.dist((x, y), (proj_x, proj_y))

            min_distance = min(min_distance, distance)

        return min_distance

    def is_valid_obstacle_position(
            self,
            x,
            y,
            size_x,
            size_y,
            existing_obstacles,
    ):

        obstacle_radius = max(size_x, size_y) * 0.5

        # =========================================================
        # evitar spawn em cima do robot
        # =========================================================

        spawn_x = self.current_spawn_translation[0]
        spawn_y = self.current_spawn_translation[1]

        robot_radius = 0.12

        spawn_distance = math.dist(
            (x, y),
            (spawn_x, spawn_y)
        )

        if spawn_distance < (
                obstacle_radius
                + robot_radius
                + 0.08
        ):
            return False

        # =========================================================
        # evitar paredes exteriores
        # =========================================================

        wall_clearance = 0.18

        if (
                abs(x) > 1.2 - wall_clearance - obstacle_radius
                or
                abs(y) > 1.2 - wall_clearance - obstacle_radius
        ):
            return False

        # =========================================================
        # evitar cantos da pista
        # =========================================================

        track_corner_clearance = 0.30

        track_corners = [
            (-1.0, 0.8),
            (1.0, 0.8),
            (-1.0, -0.8),
            (1.0, -0.8),
        ]

        for corner_x, corner_y in track_corners:

            corner_distance = math.dist(
                (x, y),
                (corner_x, corner_y)
            )

            if corner_distance < (
                    track_corner_clearance
                    + obstacle_radius
            ):
                return False

        # =========================================================
        # distancia minima à linha
        # =========================================================

        line_clearance = 0.16

        distance_to_track = self.point_to_track_distance(x, y)

        if (
                distance_to_track
                <
                obstacle_radius + line_clearance
        ):
            return False

        # =========================================================
        # distancia minima entre obstaculos
        # =========================================================

        obstacle_min_distance = 0.22

        for other_x, other_y, other_radius in existing_obstacles:

            distance = math.dist(
                (x, y),
                (other_x, other_y)
            )

            if (
                    distance
                    <
                    obstacle_radius
                    + other_radius
                    + obstacle_min_distance
            ):
                return False

        return True

    def random_obstacle_position(self):

        x = self.spawn_rng.uniform(-1.1, 1.1)
        y = self.spawn_rng.uniform(-1.1, 1.1)

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

        if self.controller_class != AdvancedANNController:
            return

        self.current_obstacles = []

        # =========================================================
        # anel inicial random de obstaculos perto do spawn
        # para evitar saidas faceis
        # =========================================================

        existing_obstacles = []

        spawn_x = self.current_spawn_translation[0]
        spawn_y = self.current_spawn_translation[1]

        ring_obstacle_count = self.spawn_rng.integers(4, 7)

        for i in range(ring_obstacle_count):

            placed = False
            attempts = 0

            while not placed and attempts < 200:

                attempts += 1

                angle = self.spawn_rng.uniform(
                    0,
                    2 * np.pi
                )

                distance = self.spawn_rng.uniform(
                    0.30,
                    0.50
                )

                x = spawn_x + math.cos(angle) * distance
                y = spawn_y + math.sin(angle) * distance

                size_x = self.spawn_rng.uniform(0.10, 0.20)
                size_y = self.spawn_rng.uniform(0.10, 0.20)

                obstacle_radius = max(size_x, size_y) * 0.5

                valid = self.is_valid_obstacle_position(
                    x,
                    y,
                    size_x,
                    size_y,
                    existing_obstacles,
                )

                if not valid:
                    continue

                rotation = self.spawn_rng.uniform(
                    0,
                    2 * np.pi
                )

                obstacle_def = f"OBSTACLE_RING_{i}"

                obstacle_string = f"""
                DEF {obstacle_def} Solid {{
                  translation {x} {y} 0.1
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

                existing_obstacles.append(
                    (
                        x,
                        y,
                        obstacle_radius,
                    )
                )

                self.obstacle_defs.append(
                    obstacle_def
                )

                self.current_obstacles.append({
                    "x": x,
                    "y": y,
                    "size_x": size_x,
                    "size_y": size_y,
                    "rotation": rotation,
                })

                placed = True

        obstacle_count = self.spawn_rng.integers(4, 9)

        for i in range(obstacle_count):

            obstacle_def = f"OBSTACLE_{i}"

            placed = False
            attempts = 0

            while not placed and attempts < 500:

                attempts += 1

                relax_factor = min(
                    1.0,
                    attempts / 500.0
                )

                position = self.random_obstacle_position()

                rotation = self.spawn_rng.uniform(
                    0,
                    2 * np.pi
                )

                size_x = self.spawn_rng.uniform(0.10, 0.24)
                size_y = self.spawn_rng.uniform(0.10, 0.24)

                obstacle_radius = max(size_x, size_y) * 0.5

                valid = self.is_valid_obstacle_position(
                    position[0],
                    position[1],
                    size_x,
                    size_y,
                    existing_obstacles,
                )

                if not valid:
                    continue

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

                existing_obstacles.append(
                    (
                        position[0],
                        position[1],
                        obstacle_radius,
                    )
                )

                self.obstacle_defs.append(obstacle_def)

                self.current_obstacles.append({
                    "x": position[0],
                    "y": position[1],
                    "size_x": size_x,
                    "size_y": size_y,
                    "rotation": rotation,
                })

                placed = True

    # ------------------------------------------------------------
    # Reset robot
    # ------------------------------------------------------------

    def create_generation_scenarios(self):

        self.generation_scenarios = []

        for _ in range(EVALUATION_RUNS):
            rotation = random_orientation(
                self.spawn_rng
            )

            translation = random_position(
                self.spawn_rng
            )

            self.current_spawn_rotation = rotation
            self.current_spawn_translation = translation

            self.remove_obstacles()
            self.generate_obstacles()

            scenario = {
                "rotation": rotation,
                "translation": translation,

                "obstacles": (
                    self.current_obstacles.copy()
                    if self.current_obstacles is not None
                    else None
                )
            }

            self.generation_scenarios.append(scenario)

        self.remove_obstacles()

    def spawn_obstacles_from_data(
            self,
            obstacles_data,
    ):

        if obstacles_data is None:
            return

        for i, obstacle_data in enumerate(obstacles_data):
            obstacle_def = f"OBSTACLE_{i}"

            obstacle_string = f"""
            DEF {obstacle_def} Solid {{
              translation
                {obstacle_data['x']}
                {obstacle_data['y']}
                0.1

              rotation 0 0 1 {obstacle_data['rotation']}

              children [
                Shape {{
                  appearance Appearance {{
                    material Material {{
                      diffuseColor 1 1 1
                    }}
                  }}

                  geometry Box {{
                    size
                      {obstacle_data['size_x']}
                      {obstacle_data['size_y']}
                      0.2
                  }}
                }}
              ]

              boundingObject Box {{
                size
                  {obstacle_data['size_x']}
                  {obstacle_data['size_y']}
                  0.2
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

    def reset(
            self,
            new_spawn=True,
            spawn_rotation=None,
            spawn_translation=None,
    ):
        self.line_recent_positions.clear()

        if spawn_rotation is not None:
            self.current_spawn_rotation = spawn_rotation

        elif new_spawn:
            self.current_spawn_rotation = random_orientation(
                self.spawn_rng
            )

        if spawn_translation is not None:
            self.current_spawn_translation = spawn_translation

        elif new_spawn:
            self.current_spawn_translation = random_position(
                self.spawn_rng
            )

        # parar motores
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        self.supervisor.step(self.timestep)

        # reposicionar
        self.rotation_field.setSFRotation(
            self.current_spawn_rotation
        )

        self.translation_field.setSFVec3f(
            self.current_spawn_translation
        )

        # reset fisica
        self.robot_node.resetPhysics()

        # deixar estabilizar
        for _ in range(3):
            self.supervisor.step(self.timestep)

        self.has_found_line = False
        self.steps_without_line = 0

        self.collision = False
        self.collision_count = 0
        self.time_on_line = 0
        self.__n = 0
        self.total_distance = 0.0
        self.distance_any_sensor = 0.0
        self.same_cell_steps = 0
        self.previous_track_cell = None
        self.previous_distance_cell = None

        self.debug_distance_reward = 0.0
        self.debug_revisit_penalty = 0.0
        self.debug_off_line_penalty = 0.0
        self.debug_collision_penalty = 0.0
        self.debug_lost_line_penalty = 0

        self.line_cells_visited = 0
        self.line_cells_revisited = 0

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

        sensors_on_line = (
                int(not ground_sensor_left)
                +
                int(not ground_sensor_right)
        )

        position_cell = self.get_position_cell(current_position)

        # =========================================================
        # Line state
        # =========================================================

        on_line = sensors_on_line >= 1
        fully_on_line = sensors_on_line == 2

        if on_line:
            self.has_found_line = True
            self.steps_without_line = 0
        else:
            if self.has_found_line:
                self.steps_without_line += 1

        # =========================================================
        # Cell tracking (1 ou 2 sensores)
        # =========================================================

        line_revisited = False

        if on_line:
            self.distance_any_sensor += step_distance
            entered_new_cell = (
                    position_cell != self.previous_track_cell
            )

            if entered_new_cell:

                line_revisited = (
                        position_cell in self.line_recent_positions
                )

                if line_revisited:
                    self.line_cells_revisited += 1

                else:
                    self.line_cells_visited += 1

                    self.line_recent_positions[position_cell] = True

                    if len(self.line_recent_positions) > LINE_BUFFER_SIZE:
                        self.line_recent_positions.popitem(last=False)

                self.previous_track_cell = position_cell

        else:
            self.previous_track_cell = None

        # =========================================================
        # Distance tracking (apenas 2 sensores)
        # =========================================================

        if fully_on_line:

            self.total_distance += step_distance
            self.time_on_line += 1

            if position_cell == self.previous_distance_cell:
                self.same_cell_steps += 1
            else:
                self.same_cell_steps = 0

            self.previous_distance_cell = position_cell

        else:

            self.same_cell_steps = 0
            self.previous_distance_cell = None

        self.prev_position = current_position
        self.__n += 1

        if left_speed < -1 and right_speed < -1:
            self.reverse_steps += 1
        else:
            self.reverse_steps = 0

        return self.get_step_fitness(sensors, step_distance, left_speed, right_speed, line_revisited)

    # ------------------------------------------------------------
    # Run evolution
    # ------------------------------------------------------------

    def run(self):
        population = self.create_population()
        stagnation_counter = 0
        previous_best_fitness = -float("inf")

        for generation in range(GENERATIONS):
            self.create_generation_scenarios()
            results = [
                self.evaluate_individual(individual)
                for individual in population
            ]

            fitnesses = np.array([r["fitness"] for r in results], dtype=float)
            distances = np.array([r["distance"] for r in results], dtype=float)
            distances_any_sensor = np.array([r["distance_any_sensor"] for r in results],dtype=float)
            trajectories = [r["trajectory"] for r in results]
            obstacles_data = [r["obstacles"] for r in results]

            collision_counts = [
                r["collision_count"]
                for r in results
            ]

            time_on_line_values = [
                r["time_on_line"]
                for r in results
            ]

            visited_counts = [
                r["line_cells_visited"]
                for r in results
            ]

            revisited_counts = [
                r["line_cells_revisited"]
                for r in results
            ]


            best_index = int(np.argmax(fitnesses))
            best_fitness = float(fitnesses[best_index])
            best_distance = float(distances[best_index])
            best_distance_any_sensor = float(distances_any_sensor[best_index])
            best_trajectory = trajectories[best_index]
            best_collision_count = int(collision_counts[best_index])
            best_time_on_line = int(time_on_line_values[best_index])
            best_obstacles = obstacles_data[best_index]
            best_visited = visited_counts[best_index]
            best_revisited = revisited_counts[best_index]

            avg_fitness = float(np.mean(fitnesses))
            avg_distance = float(np.mean(distances))
            avg_distance_any_sensor = float(np.mean(distances_any_sensor))

            best_genome = population[best_index]

            improvement = best_fitness - previous_best_fitness
            required_improvement = (abs(previous_best_fitness) * MIN_IMPROVEMENT_PERCENT)

            if improvement > required_improvement:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            previous_best_fitness = best_fitness

            if best_fitness > self.best_global_fitness:
                self.best_global_fitness = best_fitness
                self.best_global_genome = best_genome.copy()

                self.save_best_individual(
                    self.best_global_genome,
                    self.best_global_fitness,
                    generation,
                    best_distance,
                    best_distance_any_sensor
                )

            self.remove_obstacles()

            print(
                f"Generation {generation}: "
                f"Best's Fitness = {best_fitness:.2f}, "
                f"Avg Fitness = {avg_fitness:.2f}, "
                f"Best's Distance with two sensors = {best_distance:.2f}, "
                f"Best's Distance Any Sensor = {best_distance_any_sensor:.2f}, "
                f"Cells Visited = {best_visited:.2f}, "
                f"Cells Revisited = {best_revisited:.2f}, "
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
                "best_distance_any_sensor": best_distance_any_sensor,
                "avg_distance_any_sensor": avg_distance_any_sensor,
                "global_best_fitness": float(self.best_global_fitness),
                "best_genome": best_genome.tolist(),
                "best_trajectory": best_trajectory,
                "best_collision_count": best_collision_count,
                "best_time_on_line": best_time_on_line,
                "obstacles": best_obstacles,
            })

            self.save_stats()

            if EARLY_STOPPING and stagnation_counter >= STAGNATION_LIMIT:
                print(
                    f"\nEarly stopping triggered after "
                    f"{STAGNATION_LIMIT} stagnant generations."
                )
                break

            selected_parents = self.tournament_selection(
                population,
                fitnesses,
            )

            population = self.create_next_generation(
                population,
                fitnesses,
                selected_parents,
            )

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
            line_revisited,
    ):
        fitness = 0.0

        ground_sensor_left = sensors["ground_left"]
        ground_sensor_right = sensors["ground_right"]

        sensors_on_line = (
                int(not ground_sensor_left)
                +
                int(not ground_sensor_right)
        )

        # =========================================================
        # Recompensa principal:
        # distancia percorrida na linha em locais novos
        # =========================================================

        if sensors_on_line == 2:
            if line_revisited:
                penalty = step_distance * 650 + 0.1
                fitness -= penalty
                self.debug_revisit_penalty += penalty
            else:
                line_progress_reward = step_distance * 600
                fitness += line_progress_reward
                self.debug_distance_reward += line_progress_reward

        elif sensors_on_line == 1:
            if line_revisited:
                penalty = step_distance * 60 + 0.01
                fitness -= penalty
                self.debug_revisit_penalty += penalty
            else:
                fitness += step_distance * 20

            fitness -= 0.2

        else:
            fitness -= 0.5
            self.debug_off_line_penalty += 0.5

        if self.same_cell_steps > 40: #40 x 64ms = 2.56s, estar preso numa celula durante 2.56s comeca a penalizar
            fitness -= min(self.same_cell_steps * 0.002,0.3)

        # =========================================================
        # Penalizar perder a pista durante muito tempo apos ja a ter encontrado
        # =========================================================

        if self.steps_without_line > 40:
            lost_line_penalty = min(self.steps_without_line * 0.0003,0.1)
            fitness -= lost_line_penalty
            self.debug_lost_line_penalty += lost_line_penalty

        # =========================================================
        # Penalizar marcha atras prolongada
        # =========================================================

        if self.reverse_steps > 40: # TODO na melhor run estava a 15, se nao melhorar reverter para reprodutibilidade
            fitness -= 0.5

        # =========================================================
        # Penalizar colisoes
        # =========================================================

        if self.collision:
            fitness -= 1.5

            self.debug_collision_penalty += 1.5

        return fitness

    # ------------------------------------------------------------
    # Population
    # ------------------------------------------------------------

    def create_population(self):
        population = []

        #usa o melhor individuo do ultimo ou do timestamp fornecido, do respetivo controller claro
        if CONTINUE_TRAINING:

            seed_genome = np.array(load_controller_genome(
                self.controller_class,
                CONTROLLER_TIMESTAMP,
            )[0]["genome"], dtype=float)

            # preservar o individuo original
            population.append(seed_genome.copy())

            while len(population) < POPULATION_SIZE:
                mutated = seed_genome.copy()

                #gerar um vetor com os indices a serem mutados,
                # cerca de 50% serao selecionados
                mutation_mask = np.random.rand(self.genome_size) < 0.5

                # gerar um valor aleatorio para somar a cada gene selecionado,
                # usando uma distribuicao normal com media 0 e
                # desvio padrao MUTATION_SIZE * 2
                mutated[mutation_mask] += np.random.normal(
                    0, MUTATION_SIZE * 2, np.sum(mutation_mask)
                )

                population.append(mutated)

        else: #iniciar do zero
            while len(population) < POPULATION_SIZE:
                genome = np.random.uniform(-RANGE, RANGE, self.genome_size)
                population.append(genome)
        return population

    # ------------------------------------------------------------
    # Individual evaluation
    # ------------------------------------------------------------

    def evaluate_individual(self, genome):

        best_run_fitness = -float("inf")
        best_trajectory = None
        best_obstacles = None

        total_fitness = 0.0
        total_distance = 0.0
        total_collision_count = 0
        total_time_on_line = 0
        total_distance_any_sensor = 0.0

        for scenario in self.generation_scenarios:

            self.reset(
                new_spawn=False,
                spawn_rotation=scenario["rotation"],
                spawn_translation=scenario["translation"],
            )

            self.remove_obstacles()

            self.spawn_obstacles_from_data(
                scenario["obstacles"]
            )

            active_controller = self.controller_class(genome)

            run_fitness = 0.0
            trajectory = []

            start_time = self.supervisor.getTime()

            while (
                    self.supervisor.getTime() - start_time
                    <
                    EVALUATION_TIME
            ):
                step_fitness = self.runStep(
                    active_controller
                )

                run_fitness += step_fitness

                current_position = (
                    self.robot_node.getPosition()
                )

                trajectory.append([
                    current_position[0],
                    current_position[1],
                ])

            total_line_events = (
                    self.line_cells_visited
                    +
                    self.line_cells_revisited
            )

            if total_line_events > 0:
                revisit_ratio = (
                        self.line_cells_revisited
                        / total_line_events
                )
            else:
                revisit_ratio = 0.0

            revisit_ratio_penalty = (revisit_ratio * self.line_cells_revisited * 2)
            run_fitness -= revisit_ratio_penalty

            total_fitness += run_fitness
            total_distance += self.total_distance
            total_collision_count += self.collision_count
            total_time_on_line += self.time_on_line
            total_distance_any_sensor += self.distance_any_sensor

            if DEBUG_INDIVIDUALS:
                print(
                    f"Fitness={run_fitness:.1f} | "
                    f"DistReward={self.debug_distance_reward:.1f} | "
                    f"Visited={self.line_cells_visited} | "
                    f"Revisited={self.line_cells_revisited} | "
                    f"RevisitRatio={revisit_ratio:.2f} | "
                    f"RevisitPenalty={self.debug_revisit_penalty:.1f} | "
                    f"OffLinePenalty={self.debug_off_line_penalty:.1f} | "
                    f"CollisionPenalty={self.debug_collision_penalty:.1f} | "
                    f"LostLinePenalty={self.debug_lost_line_penalty:.1f} | "
                    f"Distance 2 sensors={self.total_distance:.2f} | "
                    f"Distance Any Sensor={self.distance_any_sensor:.2f} | "
                    f"TimeOnLine={self.time_on_line}"
                )

            if run_fitness > best_run_fitness:
                best_run_fitness = run_fitness
                best_trajectory = trajectory
                best_obstacles = scenario["obstacles"]

                best_line_cells_visited = self.line_cells_visited
                best_line_cells_revisited = self.line_cells_revisited

        return {
            "fitness": (
                    total_fitness
                    / EVALUATION_RUNS
            ),

            "distance": (
                    total_distance
                    / EVALUATION_RUNS
            ),

            "distance_any_sensor": (
                    total_distance_any_sensor
                    / EVALUATION_RUNS
            ),

            "trajectory": best_trajectory,
            "obstacles": best_obstacles,

            "collision_count": (
                    total_collision_count
                    / EVALUATION_RUNS
            ),

            "time_on_line": (
                    total_time_on_line
                    / EVALUATION_RUNS
            ),
            "line_cells_visited": best_line_cells_visited,
            "line_cells_revisited": best_line_cells_revisited,
        }

    # ------------------------------------------------------------
    # Next generation
    # ------------------------------------------------------------

    def create_next_generation(
            self,
            population,
            fitnesses,
            selected_parents,
    ):

        next_population = []

        # =========================================================
        # Real elitism
        # =========================================================

        elite_indices = np.argsort(fitnesses)[-PARENTS_KEEP:]

        for index in elite_indices:
            next_population.append(
                population[index].copy()
            )

        # =========================================================
        # Remaining population
        # =========================================================

        while len(next_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(
                selected_parents,
                2
            )

            child = self.crossover(
                parent1,
                parent2
            )

            child = self.mutate(child)

            next_population.append(child)

        return next_population

    # ------------------------------------------------------------
    # Tournament selection
    # ------------------------------------------------------------

    def tournament_selection(
            self,
            population,
            fitnesses,
    ):

        selected_parents = []

        while len(selected_parents) < POPULATION_SIZE:
            tournament_indices = random.sample(
                range(len(population)),
                TOURNAMENT_SIZE
            )

            best_index = max(
                tournament_indices,
                key=lambda i: fitnesses[i]
            )

            selected_parents.append(
                population[best_index]
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

        config = {
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

        if hasattr(
                self.controller_class,
                "LAYERS"
        ):
            config["layers"] = (
                self.controller_class.LAYERS
            )

        return config

    # ------------------------------------------------------------
    # Save best individual
    # ------------------------------------------------------------

    def save_best_individual(self, genome, fitness, generation, distance, distance_any_sensor):

        best_individual = {
            "controller": self.controller_class.__name__,
            "genome_size": self.genome_size,
            "genome": genome.tolist(),
            "fitness": float(fitness),
            "generation": int(generation),
            "distance": float(distance),
            "distance_any_sensor": float(distance_any_sensor),
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
# File helpers
# ============================================================

def find_best_individual_file(
        controller_class,
        timestamp=None,
):
    controller_name = (
        controller_class.__name__
    )

    matching_files = [
        f
        for f in os.listdir(
            "results/best_individuals"
        )
        if controller_name in f
    ]

    if timestamp is not None:
        matching_files = [
            f
            for f in matching_files
            if timestamp in f
        ]

    if not matching_files:
        raise FileNotFoundError(
            f"No saved individual found for "
            f"{controller_name}"
            +
            (
                f" with timestamp {timestamp}"
                if timestamp
                else ""
            )
        )

    return os.path.join(
        "results/best_individuals",
        max(
            matching_files,
            key=lambda f: os.path.getmtime(
                os.path.join(
                    "results/best_individuals",
                    f
                )
            )
        )
    )

def load_controller_genome(
        controller_class,
        timestamp=None,
):
    file_path = find_best_individual_file(
        controller_class,
        timestamp,
    )

    with open(
            file_path,
            "r",
            encoding="utf-8"
    ) as f:
        data = json.load(f)

    return data, file_path

def find_training_stats_file(
        controller_class,
        timestamp=None,
):
    controller_name = (
        controller_class.__name__
    )

    matching_files = [
        f
        for f in Path(
            "results/training_stats"
        ).glob("*.json")
        if controller_name in f.name
    ]

    if timestamp is not None:
        matching_files = [
            f
            for f in matching_files
            if timestamp in f.name
        ]

    if not matching_files:
        raise FileNotFoundError(
            f"No training stats found for "
            f"{controller_name}"
        )

    return max(
        matching_files,
        key=lambda f: f.stat().st_mtime
    )

# ============================================================
# Test best individual
# ============================================================

def test_best_individual(
        controller_class,
):
    data, file_path = (
        load_controller_genome(
            controller_class,
            CONTROLLER_TIMESTAMP,
        )
    )

    genome = np.array(
        data["genome"],
        dtype=float
    )

    fitness = data["fitness"]

    print("\nTesting best individual")
    print(
        f"File: "
        f"{os.path.basename(file_path)}"
    )
    print(
        f"Controller: "
        f"{controller_class.__name__}"
    )
    print(
        f"Fitness: "
        f"{fitness:.2f}"
    )

    evolution = Evolution(
        controller_class
    )

    evolution.reset(
        new_spawn=True
    )

    evolution.remove_obstacles()
    evolution.generate_obstacles()

    evolution.evaluate_genome_benchmark(
        genome,
        episodes=30,
    )


# ============================================================
# Test best from generation
# ============================================================

def test_generation(
        controller_class,
        generation,
):
    stats_file = (
        find_training_stats_file(
            controller_class,
            CONTROLLER_TIMESTAMP,
        )
    )

    with open(
            stats_file,
            "r",
            encoding="utf-8"
    ) as f:
        data = json.load(f)

    generation_data = next(
        (
            g
            for g in data["stats"]
            if g["generation"] == generation
        ),
        None
    )

    if generation_data is None:
        raise ValueError(
            f"Generation {generation} "
            f"not found."
        )

    genome = np.array(
        generation_data["best_genome"],
        dtype=float
    )

    print(
        f"\nTesting generation "
        f"{generation}"
    )

    print(
        f"Fitness: "
        f"{generation_data['best_fitness']:.2f}"
    )

    evolution = Evolution(
        controller_class
    )

    evolution.reset(
        new_spawn=True
    )

    evolution.remove_obstacles()
    evolution.generate_obstacles()

    evolution.evaluate_genome_benchmark(
        genome,
        episodes=30,
    )


# ============================================================
# Main
# ============================================================

def main():

    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        
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