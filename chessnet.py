#/bin/env python3
from __future__ import annotations

import os
import io
import sys
import time
import yaml
import pathlib
import shutil
import random
import kagglehub
from PIL import Image
import numpy as np


class NeuralNetwork(object):

    def __init__(self, layer_sizes: list[int]=None, *, decay: float=None):
        if layer_sizes is not None:
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
        self.decay = decay if decay is not None else 0.01

    @classmethod
    def load(cls, directory: pathlib.Path) -> NeuralNetwork:
        network = cls()
        network.weights = []
        for entry in sorted(os.listdir(directory / "weights")):
            if entry.endswith(".npy"):
                network.weights.append(np.load(directory / "weights" / entry))

        network.layer_sizes = [network.weights[0].shape[1]] # Number of rows of the first weights set
        for w in network.weights:
            network.layer_sizes.append(w.shape[0]) # Append the size of each layer
        network.layers_number = len(network.layer_sizes)
        network.biases = []
        for entry in sorted(os.listdir(directory / "biases")):
            if entry.endswith(".npy"):
                network.biases.append(np.load(directory / "biases" / entry))
        network.decay = DECAY
        return network

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        def _float_sigmoid(z_: float) -> float:
            if z_ > 0:
                return 1.0 / (1.0 + np.exp(-z_))
            else:
                tmp = np.exp(z_)
                return tmp / (1.0 + tmp)
        return np.vectorize(_float_sigmoid)(z)

    @staticmethod
    def _sigmoid_prime(z: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid function."""
        return NeuralNetwork._sigmoid(z)*(1-NeuralNetwork._sigmoid(z))

    @staticmethod
    def _array_from_image(img: Image.Image) -> np.ndarray:
        return (np.array(img).astype(np.float32) / 255.0).flatten().reshape(-1, 1)

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
            # Propagating forward to the next layer as σ(weights x in_values) + bias
            in_values = NeuralNetwork._sigmoid(weights.dot(in_values) + bias)
        return in_values

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, sorted_values=None, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        lrate = eta
        if not pathlib.Path("stats/learning-rates.csv").exists():
            with open("stats/learning-rates.csv", "w") as file:
                file.write(str(eta))
        with open("stats/learning-rates.csv", "a") as file:
            for j in range(epochs):
                random.shuffle(training_data)
                mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)
                ] # Divide into equal sized batches
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, lrate, sorted_values)
                lrate = eta * (1 / (1 + self.decay * j))
                file.write(f", {lrate}")
                if test_data:
                    print("Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(test_data), n_test
                    ))
                else:
                    print("\rEpoch {0} complete".format(j), end='')
        print()

    def update_mini_batch(self, mini_batch: list[tuple[np.ndarray, str]], eta, sorted_values=None):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # These two nabla_* vectors contain the derivatives of
        # the cost in respect of biases and weights: basically
        # the gradient we aim to follow; The term "nabla" (∇)
        # is a mathematical symbol commonly used to denote gradients.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: # Iterating over samples
            # x is the input layer, y is the correct guess
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, sorted_values)
            # Each element in the zip is a layer; we sum up
            # gradients with the deltas returned by the backpropagation
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Each sample in the mini batch influences the network
        # by a fraction of the "eta" learning factor
        mini_eta = eta/len(mini_batch)
        self.weights = [
            w - mini_eta*nw
            for w, nw in zip(self.weights, nabla_w)
        ] # Iteration happens over layers
        self.biases = [
            b - mini_eta*nb
            for b, nb in zip(self.biases, nabla_b)
        ] # Iteration happens over layers

    def backprop(self, x: np.ndarray, y: str, sorted_values: list[str]=None):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward - First we do "inference", so we can compute the cost
        activation = x # The activation in the first layer: the input
        activations: list[np.ndarray] = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b # z = weight x activation + bias
            zs.append(z)
            activation = NeuralNetwork._sigmoid(z)
            activations.append(activation)
        # Convert y to one-hot vector if needed
        if isinstance(y, str) and sorted_values is not None:
            y_index = sorted_values.index(y)
            y_vec = np.zeros((len(sorted_values), 1))
            y_vec[y_index] = 1.0
        else:
            y_vec = y # Where y_vec is the expected output (y in the book)
        # backward pass (https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)
        sp = NeuralNetwork._sigmoid_prime(zs[-1])
        if pathlib.Path("stats/cost.csv").exists():
            with open("stats/cost.csv", "a") as file:
                file.write(f", {NeuralNetwork.cost(activations[-1], y)}")
        else:
            with open("stats/cost.csv", "w") as file:
                file.write(f"{NeuralNetwork.cost(activations[-1], y)}")
        delta = NeuralNetwork.cost_derivative(activations[-1], y_vec) * sp
        nabla_b[-1] = delta # Where delta is just the derivative of the cost in respect of z
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html#numpy.ndarray.transpose
        # The .transpose() method simply "rotates" a column into a row
        # in order to enable a dot product between vectors
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.layers_number): # Iterating over layers...
            z = zs[-l] # ...kind of backwards
            sp = NeuralNetwork._sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta # Where delta is just the derivative of the cost in respect of z (includes the bias component)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            # Where nabla_w[-l] is just the derivative of the cost in respect of weight
        return (nabla_b, nabla_w) # Returns the [opposite of the] gradients that must be applied

    def evaluate(self, test_data: list[tuple[np.ndarray, str | int]], sorted_values: list[str]) -> int:
        """Return the number of test inputs for which the neural
        network outputs the correct result."""
        test_results = [
            (np.argmax(self.feedforward(x)), sorted_values.index(y) if isinstance(y, str) else y)
            for (x, y) in test_data
        ]
        for x, y in test_results:
            if x != y:
                print(f"Guessed {sorted_values[y]} but was {sorted_values[x]}!", file=sys.stderr)
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def costs(output: np.ndarray, correct: str | int, sorted_values: list[str]=None) -> list[int]:
        if isinstance(correct, str) and sorted_values is not None:
            correct: int = sorted_values.index(correct)
        return [(output[i] - (1 if i == correct else 0)) for i in range(len(output))]

    @staticmethod
    def cost(output: np.ndarray, correct: str | int, sorted_values: list[str]=None) -> float:
        costs = NeuralNetwork.costs(output, correct, sorted_values)
        return float(np.sum(np.square(costs)))

    @staticmethod
    def cost_derivative(output_activations, y):
        """Return the vector of partial derivatives
         of cost in respect of activations."""
        return 2 * (output_activations - y)

    def save(self, path: pathlib.Path) -> None:
        weights_dir = path / "weights"
        biases_dir = path / "biases"
        os.makedirs(str(weights_dir), exist_ok=True)
        os.makedirs(str(biases_dir), exist_ok=True)
        # https://numpy.org/doc/stable/reference/generated/numpy.save.html#numpy.save
        for index, layer in zip(range(1, self.layers_number), self.biases):
            np.save(biases_dir / f"{index}.npy", layer, allow_pickle=False)
            np.savetxt(biases_dir / f"{index}.csv", layer)
        for index, layer in enumerate(self.weights):
            np.save(weights_dir / f"{index}.npy", layer, allow_pickle=False)
            np.savetxt(weights_dir / f"{index}.csv", layer)

DATA_PATH = pathlib.Path(r"/home/flak-zoso/.cache/kagglehub/datasets/s4lman/chess-pieces-dataset-85x85/versions/2/data")
DATA_DESTINATION = pathlib.Path(r"./data")
SAVE_PATH = pathlib.Path(r"testNet")

SPLIT = 0.8
BATCH = 20
ETA = 0.5
DECAY = 0.01

INPUT_SIZE_X = 85
INPUT_SIZE_Y = 85
INNER_LAYER_SIZES = [10, 10]
OUTPUT_SIZE = 6

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    if not DATA_PATH.exists():
        DATA_PATH = pathlib.Path(kagglehub.dataset_download("s4lman/chess-pieces-dataset-85x85")) / 'data'

    with open("chessnet.yml", "r") as file:
        config = yaml.safe_load(file)
        EPOCHS = config["epochs"]
        SPLIT = config["split"]
        BATCH = config["batch"]
        ETA = config["eta"]
        DECAY = config["decay"]
        INPUT_SIZE_X: int = config["input-size"]["x"] if "input-size" in config else INPUT_SIZE_X
        INPUT_SIZE_Y: int = config["input-size"]["y"] if "input-size" in config else INPUT_SIZE_Y
        INNER_LAYER_SIZES: list[int] = config["inner-layer-sizes"] if "inner-layer-sizes" in config else INNER_LAYER_SIZES
        OUTPUT_SIZE: int = config["output-size"] if "output-size" in config else OUTPUT_SIZE

    pieces: dict[str, list[bytes]] = {}
    training: dict[str, list[bytes]] = {}
    testing: dict[str, list[bytes]] = {}
    try:
        shutil.copytree(str(DATA_PATH), str(DATA_DESTINATION))
    except FileExistsError:
        pass
    PIECE_NAMES = sorted([*os.listdir(DATA_DESTINATION)])
    for piece in PIECE_NAMES:
        piece_path = DATA_DESTINATION / piece
        pieces[piece] = []
        for image in os.listdir(piece_path):
            with open(str(piece_path / image), "rb") as file:
                image = Image.open(io.BytesIO(file.read()))
            image = image.convert("L")
            image.format = "PNG"
            pieces[piece].append(image)
        testing[piece] = pieces[piece][0:len(pieces[piece]):(SPLIT*10).__ceil__()]
        training[piece] = [value for value in pieces[piece] if value not in testing[piece]]
        print(f"The data for {piece}s consists of {len(testing[piece])} samples for testing and {len(training[piece])} for training")

    testing_data: list[tuple[np.ndarray, str]] = []
    for piece in PIECE_NAMES:
        for image in testing[piece]:
            testing_data.append((NeuralNetwork._array_from_image(image), piece))

    if pathlib.Path(SAVE_PATH).exists():
        print("Loading the saved neural network...")
        chessnet = NeuralNetwork.load(SAVE_PATH)
    else:
        print("Generating random neural network...")
        chessnet = NeuralNetwork(
            [INPUT_SIZE_X * INPUT_SIZE_Y] + INNER_LAYER_SIZES + [OUTPUT_SIZE], decay=DECAY
        )

    evaluate = chessnet.evaluate(testing_data, PIECE_NAMES)
    percentage = evaluate / len(testing_data) * 100
    print(f"Test data evaluation: {evaluate} / {len(testing_data)} = {percentage:.2f}%")
    if pathlib.Path("stats/results.csv").exists():
        with open("stats/results.csv", "a") as file:
            file.write(f", {percentage}")
    else:
        with open("stats/results.csv", "w") as file:
            file.write(str(percentage))

    training_data: list[tuple[np.ndarray, str]] = []
    for piece in PIECE_NAMES:
        for image in training[piece]:
            training_data.append((NeuralNetwork._array_from_image(image), piece))
    chessnet.stochastic_gradient_descent(training_data, EPOCHS, BATCH, ETA, PIECE_NAMES)

    evaluate = chessnet.evaluate(testing_data, PIECE_NAMES)
    percentage = evaluate / len(testing_data) * 100
    print(f"Test data evaluation: {evaluate} / {len(testing_data)} = {percentage:.2f}%")

    chessnet.save(SAVE_PATH)
    shutil.make_archive(f"stats/{percentage:.2f}%", "zip", root_dir=str(SAVE_PATH))
