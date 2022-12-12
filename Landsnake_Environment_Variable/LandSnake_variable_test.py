import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from landsnake_env_variable import LandSnakeEnv_VariableEnv

TARGET = np.array([3,0])
INCL_HEAD_POS = True
INCL_BODY_CENTER = False
INCL_OPT_ENCODER_VELOCITIES = True
INCL_ANG_VELS = False
INCL_HEAD_ANGLE = True
INCL_TORQUES = True
FRAME_SKIP = 1
RESET_NOISE_SCALE = 0
XML_list = ['landsnake_waypoint_obstacles_12-11_1.xml']

env = LandSnakeEnv_VariableEnv(
                xml_file_list = XML_list,
                frame_skip=FRAME_SKIP,
                NLINKS = 5,
                TARGET = TARGET,
                INCL_HEAD_ANGLE = INCL_HEAD_ANGLE,
                INCL_HEAD_POS = INCL_HEAD_POS,
                INCL_BODY_CENTER = INCL_BODY_CENTER,
                INCL_ANG_VELS = INCL_ANG_VELS,
                INCL_TORQUES = INCL_TORQUES,
                INCL_OPT_ENCODER_VELOCITIES = INCL_OPT_ENCODER_VELOCITIES,
                reset_noise_scale = RESET_NOISE_SCALE)

models_dir = "December_models/SAC"
timestamp = "12-09-2022,22:26:07"
iteration = "2000000"
model_path = f"{models_dir}/{timestamp}/{iteration}.zip"

model = SAC.load(model_path, env = env)

episodes = 1

for ep in range(episodes):

    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
    
env.close()


