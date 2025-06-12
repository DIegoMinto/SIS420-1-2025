import os


class Config:
    def __init__(self):
        self.initial_arm_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cup_position = [0.5, 0.0, 0.0]
        self.action_space_limit = 0.1
        self.episode_time_limit = 200

        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 10
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        self.max_grad_norm = 0.5

        self.hidden_size = 256

        self.num_episodes = 5000
        self.log_interval = 10
        self.model_save_dir = "models"

        os.makedirs(self.model_save_dir, exist_ok=True)


def get_config():
    return Config()
