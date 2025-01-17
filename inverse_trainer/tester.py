from stable_baselines3 import PPO
import gymnasium as gym
import torch

from StateCreator import StateCreator
from StateEvaluator import StateEvaluator

from gymnasium.envs.registration import register
register(
    id="CustomPuller-v0",
    entry_point="environments.CustomPullerEnv:CustomPullerEnv",
    max_episode_steps=300,
)
register(
    id="InverseTrainerEnvironment-v0",
    entry_point="inverse_trainer.InverseTrainerEnv:InverseTrainerEnv",
    max_episode_steps=300,
)

# model_path = "/Users/toprak/InverseRL/inverse_trainer/models/inverse_model_logs/best_model.zip"
model_path = "/Users/toprak/InverseRL/inverse_trainer/models/inverse_model_01.16-13:25_trained_500k"
model = PPO.load(model_path)

state_creator_path = "/Users/toprak/InverseRL/inverse_trainer/models/state_creators/state_creator_01.16-06:22.pth"
state_evaluator_path = "/Users/toprak/InverseRL/inverse_trainer/models/state_evaluators/state_evaluator_01.16-06:22.pth"
state_creator = StateCreator(23)
state_evaluator = StateEvaluator(23)
state_creator.load_state_dict(torch.load(state_creator_path))
state_evaluator.load_state_dict(torch.load(state_evaluator_path))

env = gym.make("InverseTrainerEnvironment-v0", render_mode="human", state_evaluator=state_evaluator, state_creator=state_creator)
# env = gym.make("CustomPuller-v0", render_mode="human")

while True:
    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, reward_terms = env.step(action)
        episode_over = terminated or truncated


