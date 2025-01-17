import minari
import torch
import numpy as np

from StateEvaluator import StateEvaluator
from StateCreator import StateCreator

if __name__ == "__main__":
    state_dim = 23

    state_evaluator = StateEvaluator(state_dim)
    state_evaluator.load_state_dict(torch.load("./models/state_evaluators/state_evaluator_01.16-06:22.pth"))

    state_creator = StateCreator(state_dim)
    state_creator.load_state_dict(torch.load("./models/state_creators/state_creator_01.16-06:22.pth"))


    dataset = minari.load_dataset("pusher_demo_R08_large-v0")
    # dataset = minari.load_dataset("pusher_demo_large-v0")

    num_episodes = dataset.total_episodes

    initial_states = np.zeros((num_episodes, state_dim))
    uncertain_states = np.zeros((2 * num_episodes, state_dim))
    final_states = np.zeros((num_episodes, state_dim))

    episodes = dataset.iterate_episodes()
    i = 0
    for episode in episodes:
        demo = episode.observations
        initial_states[i] = demo[0]
        final_states[i] = demo[-1]

        j1 = int(np.random.uniform(len(demo) // 10, 9 * len(demo) // 10))
        j2 = int(np.random.uniform(len(demo) // 10, 9 * len(demo) // 10))

        uncertain_states[2 * i] = demo[j1]
        uncertain_states[2 * i + 1] = demo[j2]
        i += 1


    initial_states = np.concatenate((initial_states, -np.ones((num_episodes, 1))), axis=1)
    uncertain_states = np.concatenate((uncertain_states, np.zeros((2*num_episodes, 1))), axis=1)
    final_states = np.concatenate((final_states, np.ones((num_episodes, 1))), axis=1)
    labeled_states = np.concatenate((initial_states, uncertain_states, final_states))
    np.random.shuffle(labeled_states)

    correct_guesses = 0
    with torch.no_grad():
        test_states = torch.tensor(labeled_states[:, :-1], dtype=torch.float32)
        test_labels = torch.tensor(labeled_states[:, -1], dtype=torch.float32)
        test_labels = test_labels.unsqueeze(1)

        predictions = state_evaluator(test_states)
        for i in range(len(predictions)):
            prediction = predictions[i]
            print(f"prediction: {float(prediction)}, real label: {test_labels[i]}, distance = {float(abs(prediction - test_labels[i]))}")
            if abs(prediction - test_labels[i]) < 0.5:
                correct_guesses += 1

        generated_states = state_creator(test_labels)
        distance_sum = 0
        for i in range(len(generated_states)):
            generated_state = generated_states[i]
            # TODO: update the code to take into account that generated_state is a mean and variance, not a state anymore

            distance_sum += float(torch.norm(generated_state - test_states[i]))
            print(f"generated_state: {generated_state}, \n real state: {test_states[i]}, \n distance = {float(torch.norm(generated_state - test_states[i]))}")

        mean_distance = distance_sum / len(generated_states)
        print(f"mean_distance: {mean_distance}")




        accuracy = correct_guesses / len(predictions)
        print(f"StateEvaluator Accuracy: {accuracy}")
