from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
import numpy as np
import yaml

data = np.loadtxt("results.csv", dtype=float, delimiter=',')
print(data)

EPOCHS = 30
SPLIT = 0.8
BATCH = 20
ETA = 0.5

INPUT_SIZE_X = 85
INPUT_SIZE_Y = 85
INNER_LAYER_SIZES = [10, 10]
OUTPUT_SIZE = 6

with open("../chessnet.yml", "r") as file:
    config: dict = yaml.safe_load(file)
    EPOCHS: int = config["epochs"] if "epochs" in config else EPOCHS
    SPLIT: float = config["split"] if "split" in config else SPLIT
    BATCH: int = config["batch"] if "batch" in config else BATCH
    ETA: float = config["eta"] if "eta" in config else ETA
    INPUT_SIZE_X: int = config["input-size"]["x"] if "input-size" in config else INPUT_SIZE_X
    INPUT_SIZE_Y: int = config["input-size"]["y"] if "input-size" in config else INPUT_SIZE_Y
    INNER_LAYER_SIZES: list[int] = config["inner-layer-sizes"] if "inner-layer-sizes" in config else INNER_LAYER_SIZES
    OUTPUT_SIZE: int = config["output-size"] if "output-size" in config else OUTPUT_SIZE

plt.figure(figsize=(10, 6))
# Set y axis to display percentages and the x axis to represent epochs (scalar)
plt.gca().yaxis.set_major_formatter(tkr.PercentFormatter())
plt.gca().xaxis.set_major_formatter(tkr.ScalarFormatter())
plt.gca().xaxis.set_major_locator(tkr.MultipleLocator(EPOCHS))
# plt.gca().xaxis.set_minor_locator(tkr.MultipleLocator(10))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(f"Training Accuracy Over Time ([{INPUT_SIZE_X}*{INPUT_SIZE_Y}, {', '.join(map(str, INNER_LAYER_SIZES))}, {OUTPUT_SIZE}], ({EPOCHS}, {BATCH}, {ETA}))")
# Plot data, setting x values to be epochs (every EPOCHS data points)
plt.plot([*range(0, len(data) * EPOCHS, EPOCHS)], data)
plt.savefig("results.png")
plt.show()


PREFIX = 1000
data = np.loadtxt("cost.csv", dtype=float, delimiter=',')
data = data[:PREFIX]
print(data)

plt.figure(figsize=(10, 6))
# Set y axis to display percentages and the x axis to represent epochs (scalar)
plt.gca().yaxis.set_major_formatter(tkr.ScalarFormatter())
plt.gca().xaxis.set_major_formatter(tkr.ScalarFormatter())
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title(f"Inference Cost Over Time ([{INPUT_SIZE_X}*{INPUT_SIZE_Y}, {', '.join(map(str, INNER_LAYER_SIZES))}, {OUTPUT_SIZE}], ({EPOCHS}, {BATCH}, {ETA}))")
# Plot data, setting x values to be epochs (every 1 data points)
plt.plot([*range(0, len(data) * 1, 1)], data)
plt.savefig("cost.png")
plt.show()

data = np.loadtxt("learning-rates.csv", dtype=float, delimiter=',')
data = data[:PREFIX]
print(data)

plt.figure(figsize=(10, 6))
# Set y axis to display percentages and the x axis to represent epochs (scalar)
plt.gca().yaxis.set_major_formatter(tkr.ScalarFormatter())
plt.gca().xaxis.set_major_formatter(tkr.ScalarFormatter())
plt.xlabel("Epochs")
plt.ylabel("Learning Rate η")
plt.title(f"Inference Learning Rate η Over Time ([{INPUT_SIZE_X}*{INPUT_SIZE_Y}, {', '.join(map(str, INNER_LAYER_SIZES))}, {OUTPUT_SIZE}], ({EPOCHS}, {BATCH}, {ETA}))")
# Plot data, setting x values to be epochs (every 1 data points)
plt.plot([*range(0, len(data) * 1, 1)], data)
plt.savefig("learning-rates.png")
plt.show()
