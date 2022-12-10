import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

DISP_POWER = True

INCL_HEAD_POS = True
INCL_OPT_ENCODER_VELOCITIES = True
INCL_TORQUES = False
INCL_ANG_VELS = False
INCL_HEAD_ANGLE = True
FRAME_SKIP = 1
RESET_NOISE_SCALE = 0

env = gym.make('landsnake_variable-v0',
                xml_file = 'landsnake_waypoint_compliant_10_7_22.xml',
                frame_skip=FRAME_SKIP,
                NLINKS = 5,
                FIXED_TARGET = True,
                TARGET = np.array([5, 0]),
                INCL_HEAD_POS=INCL_HEAD_POS,
                INCL_HEAD_ANGLE = INCL_HEAD_ANGLE,
                INCL_OPT_ENCODER_VELOCITIES = INCL_OPT_ENCODER_VELOCITIES,
                INCL_ANG_VELS = INCL_ANG_VELS,
                INCL_TORQUES = INCL_TORQUES,
                reset_noise_scale = RESET_NOISE_SCALE)

models_dir = "October_models/PPO"
timestamp = "10-19-2022,20:25:08"
iteration = "950000"
model_path = f"{models_dir}/{timestamp}/{iteration}.zip"

model = SAC.load(model_path, env = env)

episodes = 1

power_cons = []

for ep in range(episodes):
    #env.set_new_target(np.random.uniform(low = -4, high = 4, size = 2))
    env.set_new_target(np.array([2,2]))
    obs = env.reset()
    done = False
    while not done:
        #obtain power cons
        torques = env.sim.data.qfrc_actuator[env.NLINKS-1:]
        rpm_vec = []
        for i in range(env.NLINKS-1):
            rpm_vec.append(env.sim.data.get_joint_qvel("rot"+str(i+1))/(2*np.pi)) #revolutions per second
        power = np.dot(rpm_vec, torques)
        power_cons.append(power)
        
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()

if(DISP_POWER):
    timesteps = np.arange(len(power_cons))
    plt.plot(timesteps, power_cons)
    plt.plot(timesteps, np.mean(power_cons)*np.ones(len(power_cons)), 'r--')
    plt.title("Power consumption of policy trained using friction coefficients of 0.1,0.2")
    plt.xlabel("timesteps (0.01 sec each)")
    plt.ylabel("Power (Joules)")
    plt.legend(["Instantaneous Power", "Average Power"])
    print("Average power:", np.mean(power_cons))
    plt.show()

