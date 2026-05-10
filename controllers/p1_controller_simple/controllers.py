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


class SimpleANNController(BaseController):
    INPUT_SIZE = 2
    HIDDEN_SIZE = 4
    OUTPUT_SIZE = 2

    GENOME_SIZE = (
        INPUT_SIZE * HIDDEN_SIZE +
        HIDDEN_SIZE * OUTPUT_SIZE +
        HIDDEN_SIZE +
        OUTPUT_SIZE
    )

    def __init__(self, genome):
        super().__init__(genome)
        self._decode_genome()

    def _decode_genome(self):
        index = 0

        input_hidden_size = self.INPUT_SIZE * self.HIDDEN_SIZE
        hidden_output_size = self.HIDDEN_SIZE * self.OUTPUT_SIZE
        hidden_bias_size = self.HIDDEN_SIZE
        output_bias_size = self.OUTPUT_SIZE

        self.w_input_hidden = self.genome[
            index:index + input_hidden_size
        ].reshape(self.INPUT_SIZE, self.HIDDEN_SIZE)

        index += input_hidden_size

        self.w_hidden_output = self.genome[
            index:index + hidden_output_size
        ].reshape(self.HIDDEN_SIZE, self.OUTPUT_SIZE)

        index += hidden_output_size

        self.b_hidden = self.genome[
            index:index + hidden_bias_size
        ]

        index += hidden_bias_size

        self.b_output = self.genome[
            index:index + output_bias_size
        ]

    def compute_speeds(self, sensors):
        ground_left = self._ground_value(sensors, "ground_left")
        ground_right = self._ground_value(sensors, "ground_right")

        inputs = np.array(
            [ground_left, ground_right],
            dtype=float
        )

        hidden = np.tanh(
            np.dot(inputs, self.w_input_hidden) + self.b_hidden
        )

        outputs = np.tanh(
            np.dot(hidden, self.w_hidden_output) + self.b_output
        )

        left_speed = self._scale_ann_output_to_speed(outputs[0])
        right_speed = self._scale_ann_output_to_speed(outputs[1])

        return left_speed, right_speed
    

class AdvancedANNController(BaseController):
 
    INPUT_SIZE = 5
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 2

    GENOME_SIZE = (
        INPUT_SIZE * HIDDEN_SIZE +
        HIDDEN_SIZE * OUTPUT_SIZE +
        HIDDEN_SIZE +
        OUTPUT_SIZE
    )

    def __init__(self, genome):
        super().__init__(genome)
        self._decode_genome()

    def _decode_genome(self):
        index = 0

        input_hidden_size = self.INPUT_SIZE * self.HIDDEN_SIZE
        hidden_output_size = self.HIDDEN_SIZE * self.OUTPUT_SIZE
        hidden_bias_size = self.HIDDEN_SIZE
        output_bias_size = self.OUTPUT_SIZE

        self.w_input_hidden = self.genome[
            index:index + input_hidden_size
        ].reshape(self.INPUT_SIZE, self.HIDDEN_SIZE)

        index += input_hidden_size

        self.w_hidden_output = self.genome[
            index:index + hidden_output_size
        ].reshape(self.HIDDEN_SIZE, self.OUTPUT_SIZE)

        index += hidden_output_size

        self.b_hidden = self.genome[
            index:index + hidden_bias_size
        ]

        index += hidden_bias_size

        self.b_output = self.genome[
            index:index + output_bias_size
        ]

    def compute_speeds(self, sensors):
        ground_left = self._ground_value(sensors, "ground_left")
        ground_right = self._ground_value(sensors, "ground_right")

        prox_left = self._proximity_value(sensors, "prox_left")
        prox_center = self._proximity_value(sensors, "prox_center")
        prox_right = self._proximity_value(sensors, "prox_right")

        inputs = np.array(
            [
                ground_left,
                ground_right,
                prox_left,
                prox_center,
                prox_right
            ],
            dtype=float
        )

        hidden = np.tanh(
            np.dot(inputs, self.w_input_hidden) + self.b_hidden
        )

        outputs = np.tanh(
            np.dot(hidden, self.w_hidden_output) + self.b_output
        )

        left_speed = self._scale_ann_output_to_speed(outputs[0])
        right_speed = self._scale_ann_output_to_speed(outputs[1])

        return left_speed, right_speed