from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
import numpy as np

data = np.loadtxt("results.csv", dtype=float, delimiter=',')
print(data)

EPOCHS = 30

plt.figure(figsize=(10, 6))
# Set y axis to display percentages and the x axis to represent epochs (scalar)
plt.gca().yaxis.set_major_formatter(tkr.PercentFormatter())
plt.gca().xaxis.set_major_formatter(tkr.ScalarFormatter())
plt.gca().xaxis.set_major_locator(tkr.MultipleLocator(EPOCHS))
plt.gca().xaxis.set_minor_locator(tkr.MultipleLocator(10))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(f"Training Accuracy Over Time ([85*85, 25, 25, 6], ({EPOCHS}, 20, 0.5))")
# Plot data, setting x values to be epochs (every EPOCHS data points)
plt.plot([*range(0, len(data) * EPOCHS, EPOCHS)], data)
plt.savefig("results.png")
plt.show()
