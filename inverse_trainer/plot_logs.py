import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch
from StateEvaluator import StateEvaluator
import minari
import random


def plot_evaluator_differences():
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

def plot_evaluator_guesses_compared_to_real_timestamps():
    # Load dataset, environment, and the state evaluator model
    state_evaluator_path = "./models/state_evaluators/state_evaluator_02.05-17:45.pth"
    dataset = minari.load_dataset("xarm_push_5k_300steps-v0")
    env = dataset.recover_environment()

    state_evaluator = StateEvaluator(env.observation_space.shape[0])
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
            predicted = state_evaluator(obs_tensor)
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

    plot_evaluator_guesses_compared_to_real_timestamps()
    # plot_evaluator_differences()
    # plot_ppo_evaluations()