# CartPole Reinforcement Learning Visualization with Stable-Baselines3 & Hugging Face

This repository contains a Python script for downloading a pre-trained **PPO (Proximal Policy Optimization)** agent from the **Hugging Face Hub**, running it in the **Gymnasium CartPole-v1** environment, and generating a side-by-side animated visualization of the simulation frames and cumulative rewards using `Matplotlib`.

## 📋 Overview of the Pipeline

1. **Model Loading from Hugging Face Hub (`huggingface_sb3`)**:
   * Downloads a pre-trained PPO agent checkpoint (`sb3/ppo-CartPole-v1`) using `load_from_hub`.
   * Restores the model with custom configuration objects (`learning_rate`, `clip_range`).

2. **Environment Simulation & Rollout (`Gymnasium`)**:
   * Initializes the `CartPole-v1` environment with RGB array rendering mode.
   * Runs a rollout sequence of specified length ($150$ steps), tracking cumulative rewards and rendering frames.
   * Automatically handles episode resets upon termination or truncation.

3. **Dynamic Animation & Visualization (`Matplotlib`)**:
   * Builds a side-by-side matplotlib figure featuring the agent's rendered environment frame on the left and the real-time cumulative reward curve on the right.
   * Compiles the frames into an interactive HTML animation via `animation.FuncAnimation` and `IPython.display.HTML`.

---

## 🛠️ Code Implementation

```python
import gymnasium as gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import random

repo_id = "sb3/ppo-CartPole-v1"
filename = "ppo-CartPole-v1.zip"

checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
custom_objects = {
    "learning_rate": 0.0003,
    "lr_schedule": lambda _: 0.0003,
    "clip_range": lambda _: 0.2,
}

model = PPO.load(checkpoint, custom_objects=custom_objects)

env_id = "CartPole-v1"
env = gym.make(env_id, render_mode="rgb_array")

obs, info = env.reset(seed=42)
images = []
total_reward = 0
cumulative_rewards = []
sequence_length = 150

for _ in range(sequence_length):
  if random.random() < 1:
    action = env.action_space.sample()
  else:
    action, _states = model.predict(obs, deterministic=True)

  obs, reward, terminated, truncated, info = env.step(action)

  total_reward += reward
  cumulative_rewards.append(total_reward)

  img = env.render()
  images.append((img, total_reward))

  done = terminated or truncated
  if done:
    obs, info = env.reset(seed=42)
    total_reward = 0
env.close()

plt.close('all')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

def animate(i):
    img, _ = images[i]
    ax1.clear()
    ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"Frame {i + 1}", fontsize=14)

    ax2.clear()
    ax2.plot(cumulative_rewards[:i + 1], color='blue', lw=2)
    ax2.set_title("Earned rewards", fontsize=14)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward")
    ax2.grid(True)

ani = animation.FuncAnimation(fig, animate, frames=len(images), interval=50)
plt.close(fig)
HTML(ani.to_jshtml())
```

---

## 📊 Features

* **Hub Integration**: Seamlessly downloads pre-trained reinforcement learning models directly from the Hugging Face Hub.
* **Interactive Animation**: Generates browser-compatible inline animations displaying synchronous frame execution and reward accumulation.