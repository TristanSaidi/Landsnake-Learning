import gym
import time
import numpy as np
import matplotlib.pyplot as plt

INCL_HEAD_POS = False
INCL_OPT_ENCODER_VELOCITIES = False
INCL_TORQUES = False
FRAME_SKIP = 1
RESET_NOISE_SCALE = 0

env = gym.make('landsnake_variable-v0',
                xml_file = 'landsnake_waypoint_unrestricted_torque.xml',
                FIXED_TARGET = True,
                TARGET = np.array([0,5]),
                NLINKS = 5,
                INCL_HEAD_POS=INCL_HEAD_POS,
                frame_skip=FRAME_SKIP,
                reset_noise_scale = RESET_NOISE_SCALE,
                INCL_OPT_ENCODER_VELOCITIES = INCL_OPT_ENCODER_VELOCITIES)
env.reset()

STARTTIME = time.time()

AMPLITUDE = 0.9
BETA = 4.39822971502571
FREQUENCY = 9.5

ITER = 700
NET_REWARD = 0

torques = np.zeros((ITER,4))

for i in range(ITER):
    timenow = time.time() - STARTTIME

    a1 = AMPLITUDE*np.sin((FREQUENCY*timenow)+1*(BETA))
    a2 = AMPLITUDE*np.sin((FREQUENCY*timenow)+2*(BETA))
    a3 = AMPLITUDE*np.sin((FREQUENCY*timenow)+3*(BETA))
    a4 = AMPLITUDE*np.sin((FREQUENCY*timenow)+4*(BETA))

    env.render()
    observation, reward, done, info = env.step(np.array([a1,a2,a3,a4]))
    
    torques_cur = env.sim.data.qfrc_actuator[4:]
    torques[i,:] = torques_cur
    

env.close()

time = np.linspace(start = 0, stop = 700, num = 700)
plt.plot(time,torques[:,0], 'r')
plt.xlabel("Timesteps (0.01 sec each)")
plt.ylabel("Torque (N*m)")
plt.show()

