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
    evaluator_differences_path = "./logs/state_evaluator_differences_02.09-21:50.npy"
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

    thin_coeff = 10000
    steps_thinned = steps[::thin_coeff]
    predicted_thinned = predicted[::thin_coeff]
    actual_thinned = actual[::thin_coeff]

    predictions_smoothed = gaussian_filter1d(predicted, sigma=1_000)
    actual_smoothed = gaussian_filter1d(actual, sigma=1_000)

    # # ----------------------------
    # # Plot 1: Predicted vs Actual
    # # ----------------------------
    # plt.figure(figsize=(10, 6))
    # plt.plot(steps, predictions_smoothed, label="Predicted", linewidth=2)
    # plt.plot(steps, actual_smoothed, label="Actual", linewidth=2)
    # plt.xlabel("Steps")
    # plt.ylabel("Value")
    # plt.title("Predicted vs Actual")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # -------------------------------------------
    # Plot 2: Smoothed Differences (Predicted - Actual)
    # -------------------------------------------
    smoothed_diff = gaussian_filter1d(differences, sigma=1_000)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, smoothed_diff, label=f"Smoothed Differences", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Difference")
    plt.title("Smoothed Differences between Predicted and Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_evaluator_guesses_compared_to_real_timestamps():
    # Load dataset, environment, and the state evaluator model
    state_evaluator_path = "./models/state_evaluators/state_evaluator_02.09-21:50.pth"
    dataset = minari.load_dataset("xarm_push_only_successful_1k-v0")
    env = dataset.recover_environment()

    state_evaluator = StateEvaluator(3)
    state_evaluator.load_state_dict(torch.load(state_evaluator_path))

    # ===== CONFIGURATION =====
    num_episodes_to_plot = 1  # Change this to plot more episodes at once
    plot_style = "scatter"  # Options: "scatter" or "line"
    # =========================

    # Convert the generator to a list to allow random sampling
    episodes = list(dataset.iterate_episodes())

    # Randomly sample episodes to plot
    episodes_to_plot = random.sample(episodes, num_episodes_to_plot)

    for ep_num, episode in enumerate(episodes_to_plot, start=1):
        total_steps = len(episode.observations)
        actual_timestamps = []
        predicted_timestamps = []

        # Collect timestamps for each step in the episode
        for step_idx, obs in enumerate(episode.observations):
            # Convert observation to a float tensor (adjust dtype if needed)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            obs_tensor_only_object = obs_tensor[0:3]  # IMPORTANT: indexes dependent on environment
            predicted = state_evaluator(obs_tensor_only_object)
            predicted_value = predicted.item()  # convert tensor to scalar

            # Compute the actual timestamp (linearly increasing)
            actual = step_idx / total_steps

            actual_timestamps.append(actual)
            predicted_timestamps.append(predicted_value)

        # Create a new figure for each episode
        plt.figure()
        if plot_style == "scatter":
            plt.scatter(actual_timestamps, predicted_timestamps, label="Predicted")
        elif plot_style == "line":
            plt.plot(actual_timestamps, predicted_timestamps, marker="o", label="Predicted")
        else:
            raise ValueError(f"Unknown plot_style: {plot_style}")

        # Optionally, you can plot the diagonal line (perfect prediction) for reference
        plt.plot([0, 1], [0, 1], "k--", label="Ideal")

        plt.xlabel("Actual Timestamp")
        plt.ylabel("Predicted Timestamp")
        plt.title(f"Episode {ep_num}")
        plt.legend()
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

    plot_evaluator_training_info()
    # plot_evaluator_guesses_compared_to_real_timestamps()
    # plot_ppo_evaluations()