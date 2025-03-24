import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch
from torch.backends.cudnn import deterministic

from StateEvaluator import StateEvaluator
import minari
import random

from CustomPPOPolicy import CustomPolicy


def plot_evaluator_training_info(evaluator_differences_path=None):
    """
    Loads training data from a NumPy file and plots:
      1) The predicted and actual values as lines versus steps.
      2) The smoothed differences (between predicted and actual) as a line,
         using a Gaussian filter with the provided sigma.

    Parameters:
        sigma (float): The standard deviation for the Gaussian kernel used to smooth the differences.
    """
    # Path to the .npy file containing numbers.
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


def plot_evaluator_guesses_compared_to_real_timestamps(state_evaluator_path=None, title=None):
    # Load dataset, environment, and the state evaluator model
    if state_evaluator_path is None:
        state_evaluator_path = "models/state_evaluators/state_evaluator_02.19-14:43.pth"
    dataset = minari.load_dataset("xarm_push_3d_action_space_closer_1k-v0")

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

    plt.xlabel("Actual Timestamp", fontsize=14)
    plt.ylabel("Predicted Timestamp", fontsize=14)
    if title is None:
        title = "Predicted vs Actual Timestamps (All Episodes)"
    # plt.title(title)
    # plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=15)  # Move legend outside the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_evaluator_guesses_in_2d_plane(state_evaluator_path=None, title=None):
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
    # plt.xlabel("X Position (m)")
    # plt.ylabel("Y Position (m)")

    if title is None:
        title = "State Evaluator Predictions in 2D Plane"

    # plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()


def plot_initial_policy_training_info(losses_path = None):
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

def plot_initial_policy_guesses_compared_to_reverse_trajectory(path=None, mode="ee_pos"):
    if path is None:
        raise ValueError("No initial policy provided")
    else:
        initial_policy_path = path

    dataset = minari.load_dataset("xarm_push_3d_action_space_closer_1k-v0")

    initial_policy = CustomPolicy(dataset.observation_space, dataset.action_space)
    pretrained_weights = torch.load(initial_policy_path, map_location=torch.device('cpu'))
    initial_policy.load_state_dict(pretrained_weights)

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

    for step_idx, obs in enumerate(observations_rewound):
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



def plot_ppo_evaluations(path=None):

    # Load the values from the file
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]
    results = [np.max(results[i]) for i in range(len(results))]

    smoothed_results = gaussian_filter1d(results, sigma=40)

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

def make_cool_state_eval_plot():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow

    # Generate time values
    t = np.linspace(0, 1, 100)

    # Generate multiple demonstration curves
    np.random.seed(42)
    demonstrations = [np.sin(2 * np.pi * t) * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1) for _ in
                      range(10)]

    # Chosen demonstration for training
    chosen_demo = demonstrations[3]

    # Simulate timestamps as labels (normalized time for training)
    timestamps = np.linspace(0, 1, len(chosen_demo))  # Target for the model

    def plot_demonstrations():
        """Plots multiple demonstrations (D)."""
        plt.figure(figsize=(6, 4))

        for demo in demonstrations:
            plt.plot(t, demo, 'k', alpha=0.7)

        plt.title("Demonstrations (D)")
        plt.xlabel("time")
        plt.ylabel(r"$SM(t)$")

        plt.show()

    def plot_training_diagram():
        """Plots the chosen demonstration, highlighting timestamps as training targets."""
        plt.figure(figsize=(6, 4))

        # Plot chosen demonstration
        plt.plot(t, chosen_demo, 'k', label="Seçilen Gösterim")

        # Scatter plot with color gradient indicating timestamps
        sc = plt.scatter(t, chosen_demo, c=timestamps, cmap="coolwarm", edgecolors="black", label="Eğitim Veri Noktaları")

        # Add colorbar to show timestamp intensity
        cbar = plt.colorbar(sc)
        cbar.set_label("Zaman İndexi (Eğitim Hedefi)")

        # Labels and formatting
        plt.title("Çevre Gözlemi -> Zaman İndeksi Eğitimi")
        plt.xlabel("Zaman")
        plt.ylabel(r"Gözlem")
        plt.legend()

        plt.show()

    # Call functions to generate separate plots
    plot_demonstrations()
    plot_training_diagram()



if __name__ == "__main__":
    # make_cool_state_eval_plot()

    # STATE EVALUATOR TESTS
    evaluator_differences_path = "./logs/state_evaluator_differences_03.04-03:31.npy"
    state_evaluator_path = "models/state_evaluators/best_state_evaluator_03.04-03:31.pth"
    plot_evaluator_training_info(evaluator_differences_path)
    plot_evaluator_guesses_compared_to_real_timestamps(state_evaluator_path)
    plot_evaluator_guesses_in_2d_plane(state_evaluator_path)

    # PLOTTING 4 DIFFERENT TVL VALUED EVALUATOR
    # id1 = "models/state_evaluators/state_evaluator_03.02-23:29_no_TVL.pth"
    # id2 = "models/state_evaluators/state_evaluator_03.04-03:09_TVL_01.pth"
    # id3 = "models/state_evaluators/state_evaluator_03.04-03:20_TVL_02.pth"
    # id4 = "models/state_evaluators/state_evaluator_03.04-03:31_TVL_015.pth"
    #
    # notes = ["No TVL", "TVL=0.1", "TVL=0.2", "TVL=0.15"]
    # i = 0
    # for id in [id1, id2, id4, id3]:
    #     plot_evaluator_guesses_compared_to_real_timestamps(id, title=notes[i])
    #     plot_evaluator_guesses_in_2d_plane(id, title=notes[i])
    #     i += 1

    # INITIAL POLICY TESTS
    # losses_path = "./logs/initial_policy_differences_03.02-23:35.npy"
    # initial_policy_path = "models/initial_policies/best_initial_policy_log_prob_03.02-23:35.pth"
    # plot_initial_policy_training_info(losses_path)
    # plot_initial_policy_guesses_compared_to_reverse_trajectory(path=initial_policy_path)

    # show_difference_between_datasets(dataset1="xarm_synthetic_push_50-v0", dataset2="xarm_synthetic_push_1k-v0")

    # FINAL MODEL TEST
    # ppo_evaluations_path = f"./models/inverse_model_logs/03.06-03:42/evaluations.npz"
    # plot_ppo_evaluations(path=ppo_evaluations_path)

    # Pure PPO rewards
    # path = f"./models/pure_PPOs/03.07-01:48/evaluations.npz"
    # plot_ppo_evaluations(path)

