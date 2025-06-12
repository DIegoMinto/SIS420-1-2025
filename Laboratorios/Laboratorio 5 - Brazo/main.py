import torch
import torch.optim as optim
from environment import PyBulletRobotArmEnv
from agent import PPOAgent
from model import ActorCritic
from config import get_config
import os
import numpy as np


def train():
    config = get_config()

    env = PyBulletRobotArmEnv(
        render_mode='human',
        initial_arm_angles=config.initial_arm_angles,
        cup_position=np.array(config.cup_position),
        time_limit=config.episode_time_limit
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim, config.hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    agent = PPOAgent(
        model=model,
        optimizer=optimizer,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_epsilon=config.clip_epsilon,
        ppo_epochs=config.ppo_epochs,
        value_coeff=config.value_coeff,
        entropy_coeff=config.entropy_coeff,
        max_grad_norm=config.max_grad_norm
    )

    print("Comenzando entrenamiento...")
    best_reward = -float('inf')
    for episode in range(config.num_episodes):
        state, info = env.reset()

        done = False
        truncated = False
        episode_reward = 0

        episode_log_probs = []
        episode_values = []
        episode_rewards = []
        episode_masks = []
        episode_states = []
        episode_actions = []

        while not (done or truncated):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(
                action.cpu().numpy())

            episode_log_probs.append(log_prob.item())
            episode_values.append(value.item())
            episode_rewards.append(reward)
            episode_masks.append(1.0 - float(done or truncated))
            episode_states.append(state)
            episode_actions.append(action.cpu().numpy())

            state = next_state
            episode_reward += reward

        _, _, last_value_tensor = agent.select_action(state)
        last_value = last_value_tensor.item()

        agent.update(
            torch.tensor(np.array(episode_states), dtype=torch.float32),
            torch.tensor(np.array(episode_actions), dtype=torch.float32),
            torch.tensor(np.array(episode_log_probs), dtype=torch.float32),
            torch.tensor(np.array(episode_values), dtype=torch.float32),
            torch.tensor(np.array(episode_rewards), dtype=torch.float32),
            torch.tensor(np.array(episode_masks), dtype=torch.float32),
            torch.tensor(last_value, dtype=torch.float32)
        )

        print(f"Episodio {episode + 1}: Recompensa = {episode_reward:.2f}")
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_path = os.path.join(config.model_save_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"Nuevo mejor modelo guardado con recompensa: {best_reward:.2f}")

    print("Entrenamiento completado.")


if __name__ == "__main__":
    train()
