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
GAE_LAMBDA = 0.95
MAX_EP_LEN = 2000
LEARNING_RATE = 3e-4
N_EPOCHS = 3
BATCH_SIZE = 1024
INCL_HEAD_POS = True
INCL_BODY_CENTER = False
INCL_TARGET_SNAKE_DISPLACEMENT_VECTOR = False
INCL_OPT_ENCODER_VELOCITIES = True
INCL_ANG_VELS = False
INCL_HEAD_ANGLE = True
INCL_TORQUES = True
OPT_ENC_PENALTY = 0.001
FRAME_SKIP = 5
RESET_NOISE_SCALE = 0
CTRL_COST = 0
NOTES = "Both networks have [256,256,256] neurons. Activation function is tanh. SAC algorithm used"
XML = 'landsnake_waypoint_8links.xml'


policy_kwargs = dict(net_arch=dict(pi=[256,256,256], qf=[256,256,256]))

env = gym.make('landsnake_variable-v0',
                xml_file = XML,
                frame_skip=FRAME_SKIP,
                NLINKS = 8,
                TARGET = np.array([3,0]),
                FIXED_TARGET = False,
                INCL_HEAD_ANGLE = INCL_HEAD_ANGLE,
                INCL_HEAD_POS = INCL_HEAD_POS,
                INCL_BODY_CENTER = INCL_BODY_CENTER,
                INCL_TARGET_SNAKE_DISPLACEMENT_VECTOR = INCL_TARGET_SNAKE_DISPLACEMENT_VECTOR,
                INCL_ANG_VELS = INCL_ANG_VELS,
                INCL_TORQUES = INCL_TORQUES,
                INCL_OPT_ENCODER_VELOCITIES = INCL_OPT_ENCODER_VELOCITIES,
                ctrl_cost_weight = CTRL_COST,
                OPT_ENC_PENALTY = OPT_ENC_PENALTY,
                reset_noise_scale = RESET_NOISE_SCALE)

SEED = random.randint(0,100)
print(f"Random Seed: {SEED}")

date_time = datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
models_dir = f"November_models/PPO/{date_time}"
logdir = "November_logs"

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

info = open(f"{models_dir}/info.txt","w")
info.write(f"GAMMA = {GAMMA}\n")
#info.write(f"GAE_LAM = {GAE_LAMBDA}\n")
info.write(f"MAX_EP_LEN = {MAX_EP_LEN}\n")
info.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
info.write(f"INCL_HEAD_POS = {INCL_HEAD_POS}\n")
info.write(f"INCL_BODY_CENTER = {INCL_BODY_CENTER}\n")
info.write(f"INCL_HEAD_ANGLE = {INCL_HEAD_ANGLE}\n")
info.write(f"INCL_OPT_ENCODER_VELOCITIES = {INCL_OPT_ENCODER_VELOCITIES}\n")
info.write(f"INCL_ANG_VELS = {INCL_ANG_VELS}\n")
info.write(f"INCL_TORQUES = {INCL_TORQUES}\n")
info.write(f"INCL_TARGET_SNAKE_DISPLACEMENT_VECTOR = {INCL_TARGET_SNAKE_DISPLACEMENT_VECTOR}\n")
info.write(f"FRAME_SKIP = {FRAME_SKIP}\n")
info.write(f"RESET_NOISE_SCALE = {RESET_NOISE_SCALE}\n")
info.write(f"XML = {XML}\n")
info.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
info.write(f"CTRL_COST = {CTRL_COST}\n")
info.write(f"OPT_ENC_PENALTY = {OPT_ENC_PENALTY}\n")
info.write(f"NOTES = {NOTES}\n")


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
for i in range(200):

    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_{date_time}")
    model.save(f"{models_dir}/{TIMESTEPS*i}.zip")

env.close()
