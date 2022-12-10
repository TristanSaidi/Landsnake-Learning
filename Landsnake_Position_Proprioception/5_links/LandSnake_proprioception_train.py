import os
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import torch as th
import random
import numpy as np

GAMMA = 0.99
MAX_EP_LEN = 750
LEARNING_RATE = 3e-4
BATCH_SIZE = 1024
INCL_BODY_COM = True
INCL_RELATIVE_VELOCITIES = True
INCL_GLOBAL_ANGLE = True
INCL_TORQUES = True
FRAME_SKIP = 5
RESET_NOISE_SCALE = 0
NOTES = "Both networks have [256,256,256] neurons. Activation function is tanh. SAC algorithm used"
TARGET_RANGE_X = np.array([-2,2])
TARGET_RANGE_Y = np.array([0,4])
XML = 'landsnake_waypoint_obstacles_9_27_22.xml'

policy_kwargs = dict(net_arch=dict(pi=[256,256,256], qf=[256,256,256]))

env = gym.make('landsnake_variable-v0',
                xml_file = XML,
                frame_skip=FRAME_SKIP,
                NLINKS = 5,
                TRAIN = True,
                INCL_GLOBAL_ANGLE = INCL_GLOBAL_ANGLE,
                INCL_BODY_COM = INCL_BODY_COM,
                INCL_RELATIVE_VELOCITIES = INCL_RELATIVE_VELOCITIES,
                INCL_TORQUES = INCL_TORQUES,
                TARGET_RANGE_X = TARGET_RANGE_X,
                TARGET_RANGE_Y = TARGET_RANGE_Y,
                reset_noise_scale = RESET_NOISE_SCALE)

SEED = random.randint(0,100)
print(f"Random Seed: {SEED}")

date_time = datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
models_dir = f"models/SAC_Curriculum/{date_time}"
logdir = "logs_curriculum"

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

info = open(f"{models_dir}/info.txt","w")
info.write(f"GAMMA = {GAMMA}\n")
info.write(f"MAX_EP_LEN = {MAX_EP_LEN}\n")
info.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
info.write(f"INCL_BODY_COM = {INCL_BODY_COM}\n")
info.write(f"INCL_GLOBAL_ANGLE = {INCL_GLOBAL_ANGLE}\n")
info.write(f"INCL_RELATIVE_VELOCITIES = {INCL_RELATIVE_VELOCITIES}\n")
info.write(f"INCL_TORQUES = {INCL_TORQUES}\n")
info.write(f"FRAME_SKIP = {FRAME_SKIP}\n")
info.write(f"RESET_NOISE_SCALE = {RESET_NOISE_SCALE}\n")
info.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
info.write(f"NOTES = {NOTES}\n")
info.write(f"TARGET_RANGE_X = {TARGET_RANGE_X}\n")
info.write(f"TARGET_RANGE_Y = {TARGET_RANGE_Y}\n")
info.write(f"XML = {XML}\n")

info.close()

env = Monitor(env)
model = SAC("MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose = 1,
            gamma = GAMMA,
            tensorboard_log=logdir,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            seed=SEED)

TIMESTEPS = 50000
for i in range(20):

    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"SAC_{date_time}")
    model.save(f"{models_dir}/{TIMESTEPS*i}.zip")

env.close()
