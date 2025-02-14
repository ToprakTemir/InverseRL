import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch
from StateEvaluator import StateEvaluator
import minari
import random


def plot_evaluator_training_info():
    """
    Loads training data from a NumPy file and plots:
      1) The predicted and actual values as lines versus steps.
      2) The smoothed differences (between predicted and actual) as a line,
         using a Gaussian filter with the provided sigma.

    Parameters:
        sigma (float): The standard deviation for the Gaussian kernel used to smooth the differences.
    """
    # Path to the .npy file containing numbers.
    evaluator_differences_path = "./logs/state_evaluator_differences_02.12-01:28.npy"
    training_data = np.load(evaluator_differences_path, allow_pickle=True)
    training_data = [item for item in training_data if item is not None]

    # Extract columns: assuming the columns are [steps, differences, predicted, actual].

    l = len(training_data)
    steps = np.zeros(l)
    differences = np.zeros(l)
    predicted = np.zeros(l)
    actual = np.zeros(l)
    for i in range(l):
        steps[i] = training_data[i]["step"]
        differences[i] = training_data[i]["difference"]
        predicted[i] = training_data[i]["predicted"]
        actual[i] = training_data[i]["actual"]

    # -------------------------------------------
    # Plot: Smoothed Differences (Predicted - Actual)
    # -------------------------------------------

    # use exponential moving average for smoothing
    alpha = 0.001

    smoothed_diff = np.zeros_like(differences)
    smoothed_diff[0] = differences[0]
    for i in range(1, len(differences)):
        smoothed_diff[i] = alpha * differences[i] + (1 - alpha) * smoothed_diff[i - 1]


    plt.figure(figsize=(10, 6))
    plt.plot(steps, smoothed_diff, label=f"Smoothed Differences", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Difference")
    plt.title("Smoothed Differences between Predicted and Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_initial_policy_training_info():
    losses_path = "./logs/initial_policy_differences_02.13-20:40.npy"
    training_data = np.load(losses_path, allow_pickle=True)
    training_data = [item for item in training_data if item is not None]

    # Extract columns: assuming the columns are [step, loss].
    steps = np.array([item["step"] for item in training_data])
    losses = np.array([item["loss"] for item in training_data])

    losses_smoothed = gaussian_filter1d(losses, sigma=5000)

    # Plot the losses over time
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses_smoothed, label="Loss", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Initial Policy Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_evaluator_guesses_compared_to_real_timestamps():
    # Load dataset, environment, and the state evaluator model
    state_evaluator_path = "./models/state_evaluators/state_evaluator_02.12-01:28.pth"
    dataset = minari.load_dataset("xarm_push_only_successful_5k-v0")

    state_evaluator = StateEvaluator(3)
    state_evaluator.load_state_dict(torch.load(state_evaluator_path))

    # ===== CONFIGURATION =====
    num_episodes_to_plot = 5  # Change this to plot more episodes at once
    plot_style = "scatter"  # Options: "scatter" or "line"
    colors = plt.cm.get_cmap('tab10', num_episodes_to_plot)  # Get a colormap for distinct colors
    # =========================

    episodes = list(dataset.iterate_episodes())
    episodes_to_plot = random.sample(episodes, num_episodes_to_plot)

    plt.figure(figsize=(12, 8))  # Create a single figure for all episodes
    for ep_num, episode in enumerate(episodes_to_plot, start=1):
        total_steps = len(episode.observations)
        actual_timestamps = []
        predicted_timestamps = []

        # Collect timestamps for each step in the episode
        for step_idx, obs in enumerate(episode.observations):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            obs_tensor_only_object = obs_tensor[0:3]  # IMPORTANT: indexes dependent on environment
            predicted = state_evaluator(obs_tensor_only_object)
            predicted_value = predicted.item()  # Convert tensor to scalar

            # Compute the actual timestamp (linearly increasing)
            actual = step_idx / total_steps
            actual_timestamps.append(actual)
            predicted_timestamps.append(predicted_value)

        # Plot each episode with a different color and label
        if plot_style == "scatter":
            plt.scatter(actual_timestamps, predicted_timestamps, label=f"Episode {ep_num}",
                        s=5, color=colors(ep_num - 1))  # Smaller size for better clarity
        elif plot_style == "line":
            plt.plot(actual_timestamps, predicted_timestamps, marker="o", label=f"Episode {ep_num}",
                     linewidth=2, markersize=2, color=colors(ep_num - 1))
        else:
            raise ValueError(f"Unknown plot_style: {plot_style}")

    # Plot the diagonal line (perfect prediction) for reference
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")

    plt.xlabel("Actual Timestamp")
    plt.ylabel("Predicted Timestamp")
    plt.title("Predicted vs Actual Timestamps (All Episodes)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)  # Move legend outside the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ppo_evaluations():
    # Path to the .npy file containing numbers
    ppo_evaluations_path = "./models/inverse_model_logs/evaluations.npz"

    # Load the values from the file
    data = np.load(ppo_evaluations_path)
    timesteps = data["timesteps"]
    results = data["results"]
    results = [np.mean(results[i]) for i in range(len(results))]

    smoothed_results = gaussian_filter1d(results, sigma=100)

    # Plot both the original and smoothed data
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, results, label="Evaluation Results", linewidth=2)
    plt.plot(timesteps, smoothed_results, label="Smoothed Evaluation Results", linewidth=2)

    plt.xlabel("timesteps")
    plt.ylabel("evaluation results")
    plt.title("evaluation results over time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    # plot_evaluator_training_info()
    plot_initial_policy_training_info()
    # plot_evaluator_guesses_compared_to_real_timestamps()
    # plot_ppo_evaluations()