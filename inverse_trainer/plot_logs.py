import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch
from torch.backends.cudnn import deterministic

from StateEvaluator import StateEvaluator
import minari
import random

from CustomPolicy import CustomPolicy


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
    evaluator_differences_path = "./logs/state_evaluator_differences_02.19-14:43.npy"
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


def plot_evaluator_guesses_compared_to_real_timestamps(state_evaluator_path=None):
    # Load dataset, environment, and the state evaluator model
    if state_evaluator_path is None:
        state_evaluator_path = "models/state_evaluators/state_evaluator_02.19-14:43.pth"
    dataset = minari.load_dataset("xarm_push_3d_action_space_1k-v0")

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

def plot_evaluator_guesses_in_2d_plane(state_evaluator_path=None):
    """
    Visualizes the state evaluator's predictions over a 2D grid of object locations.
    Uses a smooth colormap to enhance distinguishability in abrupt value changes.
    """
    # Load the state evaluator model
    if state_evaluator_path is None:
        state_evaluator_path = "models/state_evaluators/state_evaluator_02.21-13:14.pth"

    state_evaluator = StateEvaluator(3)
    state_evaluator.load_state_dict(torch.load(state_evaluator_path))
    state_evaluator.eval()  # Set to evaluation mode

    # Define the 2D grid limits (assuming x-y plane for table space)
    x_min, x_max = -1, 1  # Adjust based on actual workspace dimensions
    y_min, y_max = -1, 1  # Adjust as needed
    grid_resolution = 100  # Increase resolution for smoother color transitions

    # Generate a grid of object positions
    x_vals = np.linspace(x_min, x_max, grid_resolution)
    y_vals = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Flatten the grid for processing
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    predictions = np.zeros(len(grid_points))

    # Evaluate the state evaluator for each point in the grid
    with torch.no_grad():
        for i, (x, y) in enumerate(grid_points):
            obj_position = torch.tensor([x, y, 0.0], dtype=torch.float32)  # Assume z=0 for 2D plot
            predictions[i] = state_evaluator(obj_position).item()

    # Reshape predictions to match the grid
    predictions = predictions.reshape(X.shape)

    # Normalize predictions for better contrast
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

    # Plot the predictions with a smooth interpolation
    plt.figure(figsize=(10, 8))
    plt.imshow(predictions, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="turbo", interpolation="bilinear")
    plt.colorbar(label="State Evaluator Prediction")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("State Evaluator Predictions in 2D Plane")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()


def plot_initial_policy_training_info():
    losses_path = "./logs/initial_policy_differences_02.19-18:58.npy"
    training_data = np.load(losses_path, allow_pickle=True)
    training_data = [item for item in training_data if item is not None]

    # Extract columns: assuming the columns are [step, loss].
    steps = np.array([item["step"] for item in training_data])
    losses = np.array([item["loss"] for item in training_data])

    losses_smoothed = gaussian_filter1d(losses, sigma=500)

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

def plot_initial_policy_guesses_compared_to_reverse_trajectory(path=None, mode="ee_pos", start_from_end=False):
    if path is None:
        initial_policy_path = "models/initial_policies/best_initial_policy_02.20-19:27_LP.pth"
    else:
        initial_policy_path = path

    dataset = minari.load_dataset("xarm_push_3d_action_space_1k-v0")

    initial_policy = CustomPolicy(dataset.observation_space, dataset.action_space)
    pretrained_weights = torch.load(initial_policy_path, map_location=torch.device('cpu'))
    initial_policy.load_pretrained_weights(pretrained_weights)

    if mode == "ee_pos":
        dim = 3
    else:
        dim = 7

    # ===== CONFIGURATION =====
    plot_style = "line"  # Options: "scatter" or "line"
    colors = plt.cm.get_cmap('tab10', 7)  # Get a colormap for distinct colors
    deterministic = True
    # =========================

    sampled_episode = dataset.sample_episodes(1)[0]
    observations = sampled_episode.observations
    observations_rewound = observations[::-1]

    total_steps = len(sampled_episode.observations)
    actual_joint_angles = np.zeros((total_steps, dim))
    predicted_joint_angles = np.zeros((total_steps, dim))

    for step_idx, obs in enumerate(observations):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        predicted, _, _ = initial_policy(obs_tensor, deterministic=deterministic)
        predicted = predicted.squeeze(0)
        predicted = predicted[:dim]
        predicted_joint_angles[step_idx] = predicted.detach().numpy()

        if mode == "ee_pos":
            actual_joint_angles[step_idx] = obs[-3:]
        else:
            actual_joint_angles[step_idx] = obs[3:10]

    # actual_joint_angles = actual_joint_angles[::-1]


    for joint_idx in range(dim):
        if plot_style == "scatter":
            plt.scatter(range(total_steps), predicted_joint_angles[:, joint_idx], label=f"Predicted Joint {joint_idx}",
                        s=5, color=colors(joint_idx))
        elif plot_style == "line":
            plt.plot(range(total_steps), predicted_joint_angles[:, joint_idx], label=f"Predicted Joint {joint_idx}",
                     linewidth=2, color=colors(joint_idx))
        else:
            raise ValueError(f"Unknown plot_style: {plot_style}")

    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 10))]
    for joint_idx in range(dim):
        plt.plot(range(total_steps), actual_joint_angles[:, joint_idx], label=f"Actual Joint {joint_idx}", linewidth=2,
                 color="black", linestyle=styles[joint_idx])


    plt.xlabel("Steps")
    plt.ylabel("Joint Angles")
    plt.title("Predicted vs Actual Joint Angles")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)  # Move legend outside the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_ppo_evaluations():

    time = "02.21-01:54"
    ppo_evaluations_path = f"./models/inverse_model_logs/{time}/evaluations.npz"

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

def show_difference_between_datasets(dataset1, dataset2, num_plots=5):

    # sample a random episode from the 2 datasets and plot them, create num_plots different plots

    dataset1 = minari.load_dataset(dataset1)
    dataset2 = minari.load_dataset(dataset2)

    for i in range(num_plots):
        episode1 = list(dataset1.sample_episodes(1))[0]
        episode2 = list(dataset2.sample_episodes(1))[0]

        observations1 = episode1.observations
        observations2 = episode2.observations

        total_steps = max(len(observations1), len(observations2))

        actual_joint_angles1 = np.zeros((total_steps, 7))
        actual_joint_angles2 = np.zeros((total_steps, 7))

        for step_idx, obs in enumerate(observations1):
            actual_joint_angles1[step_idx] = obs[3:10]

        for step_idx, obs in enumerate(observations2):
            actual_joint_angles2[step_idx] = obs[3:10]

        for joint_idx in range(7):
            plt.plot(range(total_steps), actual_joint_angles1[:, joint_idx], label=f"Dataset 1 Joint {joint_idx}", linewidth=2)
            plt.plot(range(total_steps), actual_joint_angles2[:, joint_idx], label=f"Dataset 2 Joint {joint_idx}", linewidth=2)

        plt.xlabel("Steps")
        plt.ylabel("Joint Angles")
        plt.title("Dataset 1 vs Dataset 2 Joint Angles")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":

    # STATE EVALUATOR TESTS
    # plot_evaluator_training_info()
    state_evaluator_path = "models/state_evaluators/state_evaluator_02.21-03:27.pth"
    plot_evaluator_guesses_compared_to_real_timestamps(state_evaluator_path)
    plot_evaluator_guesses_in_2d_plane(state_evaluator_path)

    # INITIAL POLICY TESTS
    # plot_initial_policy_training_info()
    # plot_initial_policy_guesses_compared_to_reverse_trajectory(start_from_end=True)

    # show_difference_between_datasets(dataset1="xarm_synthetic_push_50-v0", dataset2="xarm_synthetic_push_1k-v0")

    # FINAL MODEL TEST
    # plot_ppo_evaluations()