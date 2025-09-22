import numpy as np


class NeuralNetwork(object):

    def __init__(self, layer_sizes: list[int]):
        self.layers_number = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights: list[np.matrix] = [
            np.random.randn(y, x)
            for x, y in zip(
                self.layer_sizes[:-1],
                self.layer_sizes[1:]
            )
        ]
        self.biases: list[np.ndarray] = [
            np.random.randn(y, 1) # Random vector of biases 
            for y in self.layer_sizes[1:] # exclude first layer
        ]

    def _sigmoid(z: float | np.ndarray) -> float | np.ndarray:
        return 1.0/(1.0 + np.exp(-z))

    def feedforward(self, in_values: np.ndarray) -> np.ndarray:
        '''The output of the network given an input

        :param in_values: The values fed as input to the first layer of the neural network
        :type in_values: numpy.ndarray
        :returns: The outputs of the last layer of the neural network
        :rtype: numpy.ndarray
        '''
        if not isinstance(in_values, np.ndarray):
            raise TypeError(f"Argument {in_values} for parameter" \
                " in_values is not of type numpy.ndarray")
        if in_values.size != self.layer_sizes[0]:
            raise ValueError(f"Argument {in_values} for parameter" \
                             f" in_values has {in_values.size} values," \
                             f" {self.layer_sizes[0]} expected by network")
        for weights, bias in zip(self.weights, self.biases): # Iterating over layers
            in_values = NeuralNetwork._sigmoid(weights.dot(in_values) + bias)
        return in_values
