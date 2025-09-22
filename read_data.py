import os
import pathlib
from PIL import Image


DATA_PATH = pathlib.Path(r"/home/flak-zoso/.cache/kagglehub/datasets/s4lman/chess-pieces-dataset-85x85/versions/2/data")

pieces_files: dict[str, list[str]] = {}
for piece in os.listdir(DATA_PATH):
    piece_path = DATA_PATH / piece
    pieces_files[piece] = []
    for image in os.listdir(piece_path):
        pieces_files[piece].append(str(piece_path / image))

print(pieces_files)
