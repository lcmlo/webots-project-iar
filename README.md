# Thymio II — Evolutionary Controllers

Training a **Thymio II mobile robot** to follow a black line while maximizing the distance travelled and avoiding collisions.

The project compares three different controllers evolved using a **genetic algorithm**:

* **Braitenberg controller** — a reactive controller using only the two ground sensors.
* **Simple ANN** — a feedforward neural network using the two ground sensors.
* **Advanced ANN** — a larger neural network combining the two ground sensors with three proximity sensors, allowing the robot to follow the line while navigating obstacles.

The main objective was to investigate how different controller architectures perform when evolving robot behaviour in increasingly complex environments.

## Controllers

### Braitenberg

The Braitenberg controller receives the two ground sensors and directly determines the motor velocities.

It uses only **6 evolved parameters**, making it the simplest and most lightweight controller of the three.

### Simple ANN

The Simple ANN is a feedforward neural network with the following architecture:

```text
2 → 4 → 2
```

The two ground sensors are used as inputs and the network outputs the velocities of the two motors.

The network contains **22 parameters**.

### Advanced ANN

The Advanced ANN extends the sensor input with three proximity sensors:

```text
5 → 64 → 32 → 2
```

This allows the robot to use information about both the line and nearby obstacles.

The network contains **2530 parameters**.

Both neural networks use `tanh` activation functions in their hidden and output layers.

## Evolutionary Algorithm

The controllers are evolved using a **genetic algorithm** with:

* K-tournament selection
* Elitism
* K-point crossover
* Random mutation
* Early stopping based on fitness stagnation

### Main parameters

| Parameter           |          Value |
| ------------------- | -------------: |
| Population size     |             50 |
| Generations         |             50 |
| Elitism             |              5 |
| Tournament size     |              3 |
| K-point crossover   |              3 |
| Mutation rate       |           0.15 |
| Mutation size       |           0.05 |
| Evaluation time     |          100 s |
| Evaluation runs     |              3 |
| Maximum speed       |           9.53 |
| Stagnation limit    | 10 generations |
| Minimum improvement |         0.005% |
| Timestep            |            6.4 |

A random seed is generated for each execution and recorded so that experiments can be reproduced.

## Experimental Methodology

To make the comparison between controllers fair, the same fitness parameters, rewards and penalties were used throughout the experiments.

The robot's initial position and orientation were randomized inside a region around the centre of the arena.

Each individual was evaluated **three times per generation**, with all individuals in the same generation being tested in exactly the same scenarios.

New scenarios were generated for the following generation.

For the Advanced ANN, the obstacle configuration was also randomized between generations. This encourages the controller to evolve behaviour that generalizes to different environments rather than simply memorizing a particular configuration.

## Fitness Function

The fitness function primarily rewards the distance travelled while following the line.

Distance is considered most valuable when **both ground sensors are on the line**. Having only one sensor on the line provides a much smaller reward, allowing the robot to recover and discover the line without making this behaviour as valuable as properly following it.

Several penalties were introduced to discourage undesirable strategies:

* Line revisitation
* Excessive exploration of the same area
* Remaining stuck on the same cell
* Losing the line for an extended period
* Collisions
* Prolonged backwards movement

A buffer of **300 cells**, with each cell representing 2 cm of the line, was used to keep track of recently visited areas.

This corresponds to a tracked line distance of approximately **6 metres**.

## Results

Each controller was validated using **300 independent episodes**, with each episode lasting 100 seconds.

The main metrics were:

* Distance travelled with both sensors on the line
* Distance travelled with at least one sensor on the line
* Average number of collisions
* Percentage of episodes travelling more than 2 metres
* Percentage of episodes with at most 5 collisions

### Validation Results

| Controller   | Avg. Distance (2 sensors) | Avg. Distance (≥1 sensor) | Avg. Collisions | Distance Success (>2m) | Collision Success (≤5) |
| ------------ | ------------------------: | ------------------------: | --------------: | ---------------------: | ---------------------: |
| Braitenberg  |                    3.81 m |               **10.91 m** |           50.61 |              **95.0%** |                  95.0% |
| Simple ANN   |                **7.09 m** |                    7.71 m |           10.64 |                  72.7% |                  92.0% |
| Advanced ANN |                **7.14 m** |                    7.71 m |        **0.47** |                  83.0% |              **97.7%** |

## Results Analysis

### Braitenberg

The Braitenberg controller was able to consistently locate and follow the line despite having only six evolved parameters.

It achieved the highest distance when considering episodes where **at least one sensor** was on the line, reaching an average of 10.91 m.

However, its distance with both sensors on the line was considerably lower at 3.81 m. This indicates that the robot often followed the line in a decentralized position, with only one sensor detecting it.

Its main weakness was robustness after losing the line. In these situations, the robot tended to remain close to walls, resulting in a high average number of collisions.

### Simple ANN

The Simple ANN achieved the most precise effective line following.

The small difference between the distance measured with both sensors active (7.09 m) and with at least one sensor active (7.71 m) indicates that the robot remained relatively centred on the line for most of the evaluation.

It also significantly reduced the number of collisions compared with the Braitenberg controller.

### Advanced ANN

The Advanced ANN achieved similar line-following performance to the Simple ANN while introducing obstacle avoidance capabilities.

The combination of ground and proximity sensors allowed it to learn both **line following and obstacle navigation**.

Most notably, it achieved an average of only **0.47 collisions per episode**, considerably lower than the other controllers.

The more complex environment occasionally caused the robot to become blocked or unable to find an exit before the 100-second limit. Short backwards movements were also occasionally used as manoeuvres to escape obstacles.

## Controller Comparison

The experiments demonstrate an interesting trade-off between controller complexity and behaviour.

The **Braitenberg controller** required only 6 parameters and was nevertheless capable of producing effective line-following behaviour.

The **Simple ANN** produced significantly more centred line following, with a much smaller gap between its one-sensor and two-sensor distance measurements.

The **Advanced ANN** maintained similar line-following performance while being considerably more robust to collisions, demonstrating the advantage of additional sensory information and a more expressive neural architecture when the environment becomes more complex.

Overall:

```text
                 Simplicity    Line Following    Obstacle Avoidance
Braitenberg          ★★★★☆          ★★☆☆☆              ★☆☆☆☆
Simple ANN           ★★★☆☆          ★★★★☆              ★★☆☆☆
Advanced ANN         ★☆☆☆☆          ★★★★☆              ★★★★★
```

## Demonstrations

### Braitenberg

[Watch the Braitenberg controller](https://youtube.com/shorts/cl8mZL3IFVY?utm_source=chatgpt.com)

### Simple ANN

[Watch the Simple ANN controller](https://youtube.com/shorts/751m7WsukVE?utm_source=chatgpt.com)

### Advanced ANN

[Watch the Advanced ANN controller](https://youtube.com/shorts/diN6rfTi0bo?utm_source=chatgpt.com)

## Technologies & Concepts

* Python 
* Thymio II
* Artificial Intelligence
* Evolutionary Algorithms
* Genetic Algorithms
* Artificial Neural Networks
* Braitenberg Vehicles
* Robot Control
* Line Following
* Obstacle Avoidance
* Fitness Functions
* Behaviour Evolution

## Project Context

This project was developed for the **Artificial Intelligence for Robotics** course during the 2025/2026 academic year.

The work focused on evolving robot controllers and experimentally comparing reactive and neural approaches to autonomous robot control.
