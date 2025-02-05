import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Path to the .npy file containing numbers
evaluator_differences_path = "./logs/state_evaluator_differences_time.npy"

# Load the values from the file
values = np.load(evaluator_differences_path)

# Create a time axis where each index corresponds to a timestamp
time = np.arange(len(values))

# Smooth the data using a Gaussian filter.
# Adjust sigma to control the degree of smoothing.
smoothed_values = gaussian_filter1d(values, sigma=5)

# Plot both the original and smoothed data
plt.figure(figsize=(10, 6))
plt.plot(time, values, label="Original Data", alpha=0.5)
plt.plot(time, smoothed_values, label="Smoothed Data", linewidth=2)

plt.xlabel("Time (Index)")
plt.ylabel("Value")
plt.title("Smoothed Value-Time Graph")
plt.legend()
plt.grid(True)
plt.show()