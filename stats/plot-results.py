from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
import numpy as np

data = np.loadtxt("results.csv", dtype=int, delimiter=',')
print(data)

plt.figure(figsize=(10, 6))
# Set y axis to display percentages and the x axis to represent epochs (scalar)
plt.gca().yaxis.set_major_formatter(tkr.PercentFormatter())
plt.gca().xaxis.set_major_formatter(tkr.ScalarFormatter())
plt.gca().xaxis.set_major_locator(tkr.MultipleLocator(30))
plt.gca().xaxis.set_minor_locator(tkr.MultipleLocator(10))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Time ([85*85, 20, 25, 6], (30, 20, 0.5))")
# Plot data, setting x values to be epochs (every 30 data points)
plt.plot([*range(0, len(data) * 30, 30)], data)
plt.savefig("results.png")
plt.show()
