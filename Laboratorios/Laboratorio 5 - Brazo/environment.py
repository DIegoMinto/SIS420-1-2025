import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import os


class PyBulletRobotArmEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode='human', initial_arm_angles=None, cup_position=None, time_limit=200):
        super(PyBulletRobotArmEnv, self).__init__()

        self.render_mode = render_mode
        self.time_limit = time_limit
        self.current_step = 0
        self.robot_id = None
        self.cup_id = None
        self.plane_id = None

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        if initial_arm_angles is None:
            self.initial_arm_angles = np.zeros(7, dtype=np.float32)
        else:
            self.initial_arm_angles = np.array(
                initial_arm_angles, dtype=np.float32)

        self.num_joints = 7
        self.joint_indices = list(range(self.num_joints))

        self.joint_limits_low = np.array(
            [-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054], dtype=np.float32)
        self.joint_limits_high = np.array(
            [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054], dtype=np.float32)
        self.end_effector_link_index = 6

        self.cup_radius = 0.03
        self.cup_height = 0.1
        self.cup_mass = 0.1

        if cup_position is None:
            self.initial_cup_position = np.array(
                [0.5, 0.0, self.cup_height / 2.0], dtype=np.float32)
        else:
            self.initial_cup_position = np.array(
                cup_position, dtype=np.float32)
            self.initial_cup_position[2] += self.cup_height / 2.0

        state_dim = self.num_joints + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        action_dim = self.num_joints + 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def _get_obs(self):
        if self.robot_id is None or self.cup_id is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        joint_states = p.getJointStates(
            self.robot_id, self.joint_indices, physicsClientId=self.client)
        current_joint_angles = np.array(
            [state[0] for state in joint_states], dtype=np.float32)

        end_effector_position = np.array([0, 0, 0], dtype=np.float32)
        if self.end_effector_link_index != -1:
            ee_state = p.getLinkState(
                self.robot_id, self.end_effector_link_index, physicsClientId=self.client)
            end_effector_position = np.array(ee_state[0], dtype=np.float32)

        cup_pos, cup_orn = p.getBasePositionAndOrientation(
            self.cup_id, physicsClientId=self.client)
        cup_pos = np.array(cup_pos, dtype=np.float32)

        observation = np.concatenate(
            [current_joint_angles, end_effector_position, cup_pos])
        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        print("\n--- INICIANDO RESET ---")
        p.stepSimulation(physicsClientId=self.client)

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client)

        kuka_urdf_path = os.path.join(
            pybullet_data.getDataPath(), "kuka_iiwa", "model.urdf")
        self.robot_id = p.loadURDF(kuka_urdf_path, basePosition=[
            0, 0, 0], useFixedBase=True, physicsClientId=self.client)
        print(f"Robot cargado con ID: {self.robot_id}")

        plane_urdf_path = os.path.join(
            pybullet_data.getDataPath(), "plane.urdf")
        self.plane_id = p.loadURDF(
            plane_urdf_path, physicsClientId=self.client)
        print(f"Plano cargado con ID: {self.plane_id}")

        cup_urdf_path = os.path.join(os.path.dirname(__file__), "cup.urdf")

        self.cup_id = p.loadURDF(cup_urdf_path,
                                 basePosition=self.initial_cup_position,
                                 physicsClientId=self.client)
        print(
            f"Vaso cargado (desde URDF) con ID: {self.cup_id} en posición: {self.initial_cup_position}")

        if self.render_mode == 'human':
            p.stepSimulation(physicsClientId=self.client)
            time.sleep(0.5)

        self.joint_indices = []
        self.joint_limits_low = []
        self.joint_limits_high = []
        self.end_effector_link_index = -1

        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(
                self.robot_id, i, physicsClientId=self.client)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            joint_lower_limit = info[8]
            joint_upper_limit = info[9]

            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_limits_low.append(joint_lower_limit)
                self.joint_limits_high.append(joint_upper_limit)
                if joint_name == "lbr_iiwa_joint_7":
                    self.end_effector_link_index = i

        self.joint_limits_low = np.array(
            self.joint_limits_low, dtype=np.float32)
        self.joint_limits_high = np.array(
            self.joint_limits_high, dtype=np.float32)

        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx,
                              self.initial_arm_angles[i], physicsClientId=self.client)

        observation = self._get_obs()
        print(
            f"Posición del vaso después del reset y primer obs: {p.getBasePositionAndOrientation(self.cup_id, physicsClientId=self.client)[0]}")

        if self.render_mode == 'human':
            cup_pos_current, _ = p.getBasePositionAndOrientation(
                self.cup_id, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=cup_pos_current,
                physicsClientId=self.client
            )

        print("--- FIN DEL RESET ---")

        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1

        joint_deltas_normalized = action[:-1]
        push_action_strength_normalized = action[-1]

        joint_states = p.getJointStates(
            self.robot_id, self.joint_indices, physicsClientId=self.client)
        current_joint_angles = np.array(
            [state[0] for state in joint_states], dtype=np.float32)

        action_scale = 0.1
        target_joint_angles = current_joint_angles + \
            joint_deltas_normalized * action_scale
        target_joint_angles = np.clip(
            target_joint_angles, self.joint_limits_low, self.joint_limits_high)

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_angles,
            physicsClientId=self.client
        )

        if self.end_effector_link_index != -1 and push_action_strength_normalized > 0.5:
            ee_state = p.getLinkState(
                self.robot_id, self.end_effector_link_index, physicsClientId=self.client)
            end_effector_position = ee_state[0]

            cup_pos, _ = p.getBasePositionAndOrientation(
                self.cup_id, physicsClientId=self.client)

            direction = np.array([cup_pos[0] - end_effector_position[0],
                                  cup_pos[1] - end_effector_position[1],
                                  0.0])

            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)

            force_magnitude = abs(push_action_strength_normalized) * 50
            force_vector = force_magnitude * direction

            p.applyExternalForce(self.robot_id,
                                 self.end_effector_link_index,
                                 forceObj=force_vector,
                                 posObj=end_effector_position,
                                 flags=p.WORLD_FRAME,
                                 physicsClientId=self.client)

        p.stepSimulation(physicsClientId=self.client)
        if self.render_mode == 'human':
            time.sleep(1./self.metadata['render_fps'])

        reward = 0.0
        done = False
        truncated = False

        cup_pos_current, cup_orn_current = p.getBasePositionAndOrientation(
            self.cup_id, physicsClientId=self.client)
        cup_euler = p.getEulerFromQuaternion(cup_orn_current)

        ee_state_after_step = p.getLinkState(
            self.robot_id, self.end_effector_link_index, physicsClientId=self.client)
        end_effector_position_after_step = np.array(
            ee_state_after_step[0], dtype=np.float32)
        distance_to_cup = np.linalg.norm(
            end_effector_position_after_step - np.array(cup_pos_current, dtype=np.float32))

        reward_approach = -1.0 * distance_to_cup
        reward += reward_approach

        knocked_over_angle_threshold = np.pi / 4

        if abs(cup_euler[0]) > knocked_over_angle_threshold or \
           abs(cup_euler[1]) > knocked_over_angle_threshold:
            reward += 500.0
            done = True
            print("¡Vaso DERRIBADO exitosamente en PyBullet!")
            reward -= reward_approach

        reward -= 0.01

        if self.current_step >= self.time_limit:
            done = True
            truncated = True
            if not done:
                reward -= 50.0
            print("Límite de tiempo alcanzado.")

        observation = self._get_obs()
        info = {}

        return observation, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.client)


if __name__ == '__main__':
    from config import get_config
    config = get_config()

    cup_pos_config = np.array(config.cup_position)
    initial_arm_angles_config = np.array(config.initial_arm_angles)

    env = PyBulletRobotArmEnv(
        render_mode='human',
        initial_arm_angles=initial_arm_angles_config,
        cup_position=cup_pos_config,
        time_limit=config.episode_time_limit
    )

    obs, info = env.reset()
    print(f"Estado inicial (obs) después del primer reset: {obs}")

    num_test_episodes = 5
    for episode in range(num_test_episodes):
        print(f"\n--- Iniciando Episodio {episode + 1} ---")
        obs, info = env.reset()
        episode_reward = 0

        for _ in range(config.episode_time_limit + 50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                print(
                    f"Episodio {episode + 1} terminado. Recompensa total: {episode_reward:.2f}")
                break
        else:
            print(
                f"Episodio {episode + 1} terminado por límite de tiempo. Recompensa total: {episode_reward:.2f}")

    env.close()
