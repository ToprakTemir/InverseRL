import minari
import time
import sys

def run_demo(episode_idx=0, sleep_time=0.05):
    # Load the Minari dataset and recover the underlying MuJoCo environment
    dataset = minari.load_dataset("xarm_push_1k_300steps-v0")
    env = dataset.recover_environment().unwrapped

    # IMPORTANT: this doesn't look like the real demonstration since the environment is reset and the object is placed at a different location than it was in the original demonstration, but the robot executes the same movements.
    for episode in dataset.iterate_episodes():
        actions = episode.actions

        print(f"Running demonstration episode {episode_idx} with {len(actions)} steps.")

        # Reset the environment.
        # (Note: Depending on the recovered environment, resetting may not set the state to match the demonstration.
        # If you need exact state playback and your environment supports it, you may need to set the state manually.)
        obs = env.reset()
        env.render_mode = "human"

        for idx, action in enumerate(actions):
            env.render()  # Render the environment; this opens a MuJoCo viewer window.
            obs, reward, done, truncated, info = env.step(action)
            time.sleep(sleep_time)
            if done or truncated:
                print(f"Episode finished after {idx+1} steps.")
                break

    # Close the rendering window (if supported by your environment)
    env.close()

if __name__ == '__main__':
    # Optionally pass the episode index as a command-line argument
    run_demo()