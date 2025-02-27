import minari
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
dataset = minari.load_dataset("xarm_push_3d_action_space_closer_1k-v0")

# Set plot limits
x_min, x_max = -1, 1
y_min, y_max = -1, 1

# Iterate through all episodes
for episode in dataset.iterate_episodes():
    observations = episode.observations
    ee_positions = np.array([obs[-3:-1] for obs in observations]) # ignore z by -1

    # Extract x, y (ignoring z)
    x_vals = ee_positions[:, 0]
    y_vals = ee_positions[:, 1]

    # Generate a color gradient from blue to red
    colors = np.linspace(0, 1, len(x_vals))  # 0 (blue) to 1 (red)

    # Plot the trajectory
    plt.scatter(x_vals, y_vals, c=colors, cmap='coolwarm', edgecolors='k', s=3, linewidths=0.3)

# Configure the plot
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("End-Effector Trajectories")
plt.colorbar(label="Progress from Start (Blue) to End (Red)")
plt.show()