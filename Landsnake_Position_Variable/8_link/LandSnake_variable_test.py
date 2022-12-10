import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

INCL_HEAD_POS = True
INCL_OPT_ENCODER_VELOCITIES = True
INCL_TORQUES = True
INCL_ANG_VELS = False
INCL_HEAD_ANGLE = True
FRAME_SKIP = 1
RESET_NOISE_SCALE = 0

env = gym.make('landsnake_variable-v0',
                xml_file = 'landsnake_waypoint_8links.xml',
                frame_skip=FRAME_SKIP,
                NLINKS = 8,
                FIXED_TARGET = True,
                TARGET = np.array([5, 0]),
                INCL_HEAD_POS=INCL_HEAD_POS,
                INCL_HEAD_ANGLE = INCL_HEAD_ANGLE,
                INCL_OPT_ENCODER_VELOCITIES = INCL_OPT_ENCODER_VELOCITIES,
                INCL_ANG_VELS = INCL_ANG_VELS,
                INCL_TORQUES = INCL_TORQUES,
                reset_noise_scale = RESET_NOISE_SCALE)

models_dir = "November_models/PPO"
timestamp = "11-15-2022,18:18:27"
iteration = "1000000"
model_path = f"{models_dir}/{timestamp}/{iteration}.zip"

model = SAC.load(model_path, env = env)

episodes = 10

for ep in range(episodes):
    env.set_new_target(np.random.uniform(low = -4, high = 4, size = 2))
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()
