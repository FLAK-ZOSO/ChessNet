#/bin/env python3
from __future__ import annotations

import os
import io
import pathlib
import shutil
import kagglehub
from PIL import Image
import matplotlib.pyplot as plt
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

    @classmethod
    def load(cls, directory: str) -> NeuralNetwork:
        ...

    def _sigmoid(z: np.ndarray) -> np.ndarray:
        def _float_sigmoid(z_: float) -> float:
            if z_ > 0:
                return 1.0 / (1.0 + np.exp(-z_))
            else:
                tmp = np.exp(z_)
                return tmp /(1.0 + tmp)
        return [_float_sigmoid(z_) for z_ in z]

    def _array_from_image(img: Image.Image) -> np.ndarray:
        return np.array(img).flatten().reshape(-1, 1)

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

    def save(self, path: str) -> None:
        directory = pathlib.Path(path)
        weights_dir = directory / "weights"
        biases_dir = directory / "biases"
        os.makedirs(str(weights_dir), exist_ok=True)
        os.makedirs(str(biases_dir), exist_ok=True)
        # https://numpy.org/doc/stable/reference/generated/numpy.save.html#numpy.save
        for index, layer in zip(range(1, self.layers_number), self.biases):
            np.save(weights_dir / f"{index}.npy", layer, allow_pickle=False)
        for index, layer in enumerate(self.weights):
            np.save(biases_dir / f"{index}.npy", layer, allow_pickle=False)

DATA_PATH = pathlib.Path(r"/home/flak-zoso/.cache/kagglehub/datasets/s4lman/chess-pieces-dataset-85x85/versions/2/data")
DATA_DESTINATION = pathlib.Path(r"./data")
SPLIT = 0.8


if __name__ == "__main__":
    if not DATA_PATH.exists():
        DATA_PATH = pathlib.Path(kagglehub.dataset_download("s4lman/chess-pieces-dataset-85x85")) / 'data'
    pieces: dict[str, list[bytes]] = {}
    training: dict[str, list[bytes]] = {}
    testing: dict[str, list[bytes]] = {}
    try:
        shutil.copytree(str(DATA_PATH), str(DATA_DESTINATION))
    except FileExistsError:
        pass
    for piece in os.listdir(DATA_DESTINATION):
        piece_path = DATA_DESTINATION / piece
        pieces[piece] = []
        for image in os.listdir(piece_path):
            with open(str(piece_path / image), "rb") as file:
                image = Image.open(io.BytesIO(file.read()))
            image = image.convert("L")
            image.format = "PNG"
            pieces[piece].append(image)
        training[piece] = pieces[piece][:int(SPLIT*len(pieces[piece]))]
        testing[piece] = pieces[piece][:int((1 - SPLIT)*len(pieces[piece]))]
    print(NeuralNetwork._array_from_image(testing["king"][0]))

    chessnet = NeuralNetwork([85*85, 15, 15, 6])
    array = NeuralNetwork._array_from_image(testing["bishop"][0])
    output = chessnet.feedforward(array)
    print(*zip(pieces.keys(), output))

    chessnet.save("testNet")
