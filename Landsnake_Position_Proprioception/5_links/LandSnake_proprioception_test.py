import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC



INCL_RELATIVE_VELOCITIES = True
INCL_GLOBAL_ANGLE = True
INCL_BODY_COM = True
INCL_TORQUES = True
FRAME_SKIP = 1
RESET_NOISE_SCALE = 0

env = gym.make('landsnake_variable-v0',
                xml_file = 'landsnake_waypoint_obstacles_9_30_22.xml',
                frame_skip=FRAME_SKIP,
                NLINKS = 5,
                TRAIN = False,
                TARGET = np.array([-3, -3]),
                INCL_GLOBAL_ANGLE=INCL_GLOBAL_ANGLE,
                INCL_BODY_COM = INCL_BODY_COM,
                INCL_RELATIVE_VELOCITIES = INCL_RELATIVE_VELOCITIES,
                INCL_TORQUES = INCL_TORQUES,
                reset_noise_scale = RESET_NOISE_SCALE)

models_dir = "models/SAC_Curriculum"
timestamp = "09-29-2022,16:25:13"
iteration = "800000"
model_path = f"{models_dir}/{timestamp}/{iteration}.zip"

model = SAC.load(model_path, env = env)

episodes = 10

for ep in range(episodes):
    env.set_new_target(np.array([np.random.uniform(low=-2, high=2), np.random.uniform(low=0, high=4)]))
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()
