import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Path to the .npy file containing numbers
evaluator_differences_path = "./logs/state_evaluator_differences_02.05-17:26.npy"

# Load the values from the file
values = np.load(evaluator_differences_path)

# Adjust sigma to control the degree of smoothing.
smoothed_values = gaussian_filter1d(values, sigma=500)

# Plot both the original and smoothed data
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, len(values), len(values)), values, label="Original Data", alpha=0.5)
plt.plot(np.linspace(0, len(values), len(values)), smoothed_values, label="Smoothed Data", linewidth=2)

plt.xlabel("step")
plt.ylabel("real - predicted difference")
plt.title("prediction error over time")
plt.legend()
plt.grid(True)
plt.show()