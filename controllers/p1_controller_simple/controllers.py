import numpy as np


class BaseController:
    GENOME_SIZE = None
    MAX_SPEED = 9.0

    def __init__(self, genome):
        self.genome = np.array(genome, dtype=float)

        if self.GENOME_SIZE is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define GENOME_SIZE."
            )

        if len(self.genome) != self.GENOME_SIZE:
            raise ValueError(
                f"{self.__class__.__name__} expected genome size "
                f"{self.GENOME_SIZE}, but got {len(self.genome)}."
            )

    def compute_speeds(self, sensors):
        raise NotImplementedError

    @staticmethod
    def _clip(value, min_value, max_value):
        return float(np.clip(value, min_value, max_value))

    def _clip_speed(self, speed):
        return self._clip(speed, -self.MAX_SPEED, self.MAX_SPEED)

    def _ground_value(self, sensors, key):
        value = sensors[key]

        if isinstance(value, (bool, np.bool_)):
            return 1.0 if value else 0.0

        value = float(value)

        if value > 1.0:
            value = value / 1023.0

        return self._clip(value, 0.0, 1.0)

    def _proximity_value(self, sensors, key):
        value = float(sensors[key])

        if value > 1.0:
            value = value / 4500.0

        return self._clip(value, 0.0, 1.0)

    def _scale_ann_output_to_speed(self, output_value):
        return self._clip_speed(output_value * self.MAX_SPEED)
    

class BraitenbergController(BaseController):

    GENOME_SIZE = 6

    def compute_speeds(self, sensors):
        se = self._ground_value(sensors, "ground_left")
        sd = self._ground_value(sensors, "ground_right")

        p1e, p2e, p3e, p1d, p2d, p3d = self.genome

        left_speed = (p1e * se) + (p2e * sd) + p3e
        right_speed = (p1d * se) + (p2d * sd) + p3d

        left_speed = self._clip_speed(left_speed)
        right_speed = self._clip_speed(right_speed)

        return left_speed, right_speed

class ANN:

    def __init__(
            self,
            genome,
            layer_sizes,
            hidden_activation=np.tanh,
            output_activation=np.tanh,
    ):
        self.genome = genome
        self.layer_sizes = layer_sizes

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.weights = []
        self.biases = []

        self._decode_genome()

    def _decode_genome(self):
        index = 0

        for i in range(len(self.layer_sizes) - 1):
            inputs = self.layer_sizes[i]
            outputs = self.layer_sizes[i + 1]

            weight_count = inputs * outputs

            W = self.genome[
                index:index + weight_count
            ].reshape(inputs, outputs)

            index += weight_count

            b = self.genome[
                index:index + outputs
            ]

            index += outputs

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, inputs):

        x = np.asarray(inputs, dtype=float)

        for i in range(len(self.weights)):

            x = (
                    np.dot(x, self.weights[i])
                    + self.biases[i]
            )

            is_output_layer = (
                    i == len(self.weights) - 1
            )

            if is_output_layer:
                x = self.output_activation(x)
            else:
                x = self.hidden_activation(x)

        return x

    @staticmethod
    def calculate_genome_size(layer_sizes):
        size = 0

        for i in range(len(layer_sizes) - 1):
            inputs = layer_sizes[i]
            outputs = layer_sizes[i + 1]

            size += inputs * outputs
            size += outputs

        return size

class SimpleANNController(BaseController):
    LAYERS = [2, 4, 2]

    HIDDEN_ACTIVATION = np.tanh
    OUTPUT_ACTIVATION = np.tanh

    GENOME_SIZE = ANN.calculate_genome_size(
        LAYERS
    )

    def __init__(self, genome):
        super().__init__(genome)

        self.ann = ANN(
            genome=self.genome,
            layer_sizes=self.LAYERS,
            hidden_activation=self.HIDDEN_ACTIVATION,
            output_activation=self.OUTPUT_ACTIVATION,
        )

    def compute_speeds(self, sensors):
        ground_left = self._ground_value(
            sensors,
            "ground_left"
        )

        ground_right = self._ground_value(
            sensors,
            "ground_right"
        )

        outputs = self.ann.forward(
            [ground_left, ground_right]
        )

        left_speed = self._scale_ann_output_to_speed(
            outputs[0]
        )

        right_speed = self._scale_ann_output_to_speed(
            outputs[1]
        )

        return left_speed, right_speed

class AdvancedANNController(BaseController):

    LAYERS = [5, 32, 16, 2]

    HIDDEN_ACTIVATION = np.tanh
    OUTPUT_ACTIVATION = np.tanh

    GENOME_SIZE = ANN.calculate_genome_size(
        LAYERS
    )

    def __init__(self, genome):
        super().__init__(genome)

        self.ann = ANN(
            genome=self.genome,
            layer_sizes=self.LAYERS,
            hidden_activation=self.HIDDEN_ACTIVATION,
            output_activation=self.OUTPUT_ACTIVATION,
        )

    def compute_speeds(self, sensors):
        ground_left = self._ground_value(
            sensors,
            "ground_left"
        )

        ground_right = self._ground_value(
            sensors,
            "ground_right"
        )

        prox_left = self._proximity_value(
            sensors,
            "prox_left"
        )

        prox_center = self._proximity_value(
            sensors,
            "prox_center"
        )

        prox_right = self._proximity_value(
            sensors,
            "prox_right"
        )

        outputs = self.ann.forward(
            [
                ground_left,
                ground_right,
                prox_left,
                prox_center,
                prox_right,
            ]
        )

        left_speed = self._scale_ann_output_to_speed(
            outputs[0]
        )

        right_speed = self._scale_ann_output_to_speed(
            outputs[1]
        )

        return left_speed, right_speed