import numpy as np
from controller import Supervisor
import random
import math
import numpy as np
import json

# Simulation parameters
TIME_STEP = 5
POPULATION_SIZE = 10
PARENTS_KEEP = 2
#INPUT = 5
#HIDDEN = 4
#OUTPUT = 2
#GENOME_SIZE = (1+INPUT)*HIDDEN  + (HIDDEN+1)*OUTPUT
GENERATIONS = 10
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05
EVALUATION_TIME = 60  # Simulated seconds per individual
RANGE = 2


def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return (0, 0, 1, angle)

def random_position(min_radius, max_radius, z):
    radius = np.random.uniform(min_radius, max_radius)
    angle = random_orientation()
    x = radius * np.cos(angle[3])
    y = radius * np.sin(angle[3])
    return (x, y, z)

class Evolution:
    def __init__(self, input, hidden, output):
        self.genome_size = 6#(1+input)*hidden+ (hidden+1)*output
        
        
        self.evaluation_start_time = 0
        self.collision = False

        # Supervisor to reset robot position
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
   
        self.robot_node = self.supervisor.getFromDef("ROBOT") 
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep()*TIME_STEP)
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')

        self.__ir_0 = self.supervisor.getDevice('prox.horizontal.0')
        self.__ir_1 = self.supervisor.getDevice('prox.horizontal.1')
        self.__ir_2 = self.supervisor.getDevice('prox.horizontal.2')
        self.__ir_3 = self.supervisor.getDevice('prox.horizontal.3')
        self.__ir_4 = self.supervisor.getDevice('prox.horizontal.4')
        self.__ir_5 = self.supervisor.getDevice('prox.horizontal.5')
        self.__ir_6 = self.supervisor.getDevice('prox.horizontal.6')
        self.__ir_7 = self.supervisor.getDevice('prox.ground.0')
        self.__ir_8 = self.supervisor.getDevice('prox.ground.1')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.__ir_0.enable(self.timestep)
        self.__ir_1.enable(self.timestep)
        self.__ir_2.enable(self.timestep)
        self.__ir_3.enable(self.timestep)
        self.__ir_4.enable(self.timestep)
        self.__ir_5.enable(self.timestep)
        self.__ir_6.enable(self.timestep)
        self.__ir_7.enable(self.timestep)
        self.__ir_8.enable(self.timestep)

        self.sensors = [self.__ir_0,self.__ir_2,self.__ir_4]
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]

        self.__n = 0
        self.prev_position = self.supervisor.getSelf().getPosition()
        

    def reset(self, seed=42, options=None):
        
        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.supervisor.getFromDef('ROBOT').getField('rotation').setSFRotation(random_rotation)
        self.supervisor.getFromDef('ROBOT').getField('translation').setSFVec3f([0, 0, 0])
        
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        

    def runStep(self, weights):
        
        self.collision = bool(
                self.__n > 10 and
                (self.__ir_0.getValue()>4300 or 
                self.__ir_1.getValue()>4300 or
                self.__ir_2.getValue()>4300 or
                self.__ir_3.getValue()>4300 or
                self.__ir_4.getValue()>4300 or
                self.__ir_5.getValue()>4300 or
                self.__ir_6.getValue()>4300)
            )
        
        ground_sensor_left = (self.ground_sensors[0].getValue()/1023 - .6)/.2>.3
        ground_sensor_right = (self.ground_sensors[1].getValue()/1023 - .6)/.2>.3
        #print(f"Ground Sensors: Left={ground_sensor_left}, Right={ground_sensor_right}")

        left_speed =  ground_sensor_left * weights[0] + ground_sensor_right * weights[1] + weights[2]
        right_speed = ground_sensor_left * weights[3] + ground_sensor_right * weights[4] + weights[5]
        
        self.left_motor.setVelocity(max(min(left_speed, 9), -9))
        self.right_motor.setVelocity(max(min(right_speed, 9), -9))

        self.supervisor.step(self.timestep)

        return self.get_step_fitness()

    def create_population(self):
        return [
            np.random.uniform(-RANGE, RANGE, self.genome_size)
            for _ in range(POPULATION_SIZE)
        ]
    
    def evaluate_individual(self, weights):
        self.reset()
    
        fitness = 0
        self.collision = False
        self.__n = 0
    
        start_time = self.supervisor.getTime()
        self.prev_position = self.robot_node.getPosition()
    
        while self.supervisor.getTime() - start_time < EVALUATION_TIME and not self.collision:
            #print(self.timestep)
            step_fitness = self.runStep(weights)
            fitness += step_fitness
    
        #print(f'Fitness: {fitness}')
        return fitness
   
    def run(self):
        self.evaluation_start_time = self.supervisor.getTime()
        for generation in range(GENERATIONS):
            population = self.create_population()
            fitnesses = [self.evaluate_individual(ind) for ind in population]
            best_fitness = max(fitnesses)
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            self.save_best_individual(population[np.argmax(fitnesses)], best_fitness)
            # Select parents 
            parents_indices = np.argsort(fitnesses)[-PARENTS_KEEP:]
            parents = [population[i] for i in parents_indices]
            
            # Create next generation
            next_population = parents.copy()
            while len(next_population) < POPULATION_SIZE:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)
            
            population = next_population
    

    # POSSIBLE TODO STUFF: random cut point, fixed, mudar o ponto por geração, etc
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.genome_size - 1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:])) 
        return child
    

    # POSSIBLE TODO STUFF: uniform mutation, gaussian mutation, a mutação pode variar ao longo das gerações (para funcionar como fine tune)
    def mutate(self, genome):
        for i in range(len(genome)):
            if random.random() < MUTATION_RATE:
                genome[i] += np.random.uniform(-MUTATION_SIZE, MUTATION_SIZE)
                genome[i] = max(min(genome[i], RANGE), -RANGE)
        return genome
    
    
    def get_step_fitness(self):
        # Fitness: penalizar quando o sensor está a true e premiar quando está a false, para incentivar o robô a evitar obstáculos e a seguir a linha. O peso de cada sensor pode ser ajustado para equilibrar a importância de cada um.
        fitness = 0
        ground_sensor_left = (self.ground_sensors[0].getValue()/1023 - .6)/.2>.3
        ground_sensor_right = (self.ground_sensors[1].getValue()/1023 - .6)/.2>.3

        if ground_sensor_left or ground_sensor_right:
            #print("gray detected!")
            fitness -= 0.01
        else:            
            fitness += 1

        if self.collision:
            print("Collision detected!")
            fitness -= 10
        
        return fitness
    

    def save_best_individual(self, genome, fitness):
        # Save the best individual's genome and its fitness to a json file in json formate with weights and biases (if there are biases)
        
        best_individual = {
            "genome": genome.tolist(),
            "fitness": fitness      
        }
        with open("best_individual.json", "w") as f:
            json.dump(best_individual, f, indent=4)
        

# metodo para carregar o melhor individuo de um ficheiro json e correr a simulação com ele, para ver o comportamento do melhor individuo encontrado
def test_best_individual():
    import json
    with open("best_individual.json", "r") as f:
        best_individual = json.load(f)
    
    genome = np.array(best_individual["genome"])
    fitness = best_individual["fitness"]
    print(f"Testing Best Individual: Fitness = {fitness}")
    
    controller = Evolution(2, 0, 2)
    controller.reset()
    
    # Run the simulation with the best individual's genome
    while True:
        controller.runStep(genome)


# Main evolutionary loop
def main():
    # Run the evolutionary algorithm
    controller = Evolution(2, 0, 2)
    controller.run()
    test_best_individual()
if __name__ == "__main__":
    main()

