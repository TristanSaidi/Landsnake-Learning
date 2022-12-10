import numpy as np
import gym
from gym.envs.mujoco import mujoco_env
from gym import utils

NLINKS = 5

DEFAULT_CAMERA_CONFIG = {}


class LandSnakeEnv_VariableEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file_list,
        frame_skip,
        NLINKS,
        TARGET = np.array([0,0]),
        INCL_HEAD_ANGLE=False,
        INCL_OPT_ENCODER_VELOCITIES=False,
        INCL_HEAD_POS=False,
        INCL_BODY_CENTER = False,
        INCL_TORQUES=False,
        INCL_ANG_VELS=False,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    ):
   
        utils.EzPickle.__init__(**locals())
       
        self.NLINKS = NLINKS
        self.INCL_HEAD_ANGLE = INCL_HEAD_ANGLE
        self.INCL_OPT_ENCODER_VELOCITIES = INCL_OPT_ENCODER_VELOCITIES
        self.INCL_HEAD_POS = INCL_HEAD_POS
        self.INCL_TORQUES = INCL_TORQUES
        self.INCL_BODY_CENTER = INCL_BODY_CENTER
        self.INCL_ANG_VELS = INCL_ANG_VELS
        
        self.TARGET = TARGET
        
        #allowing reset() function to access simulation parameters 
        self.xml_file_list = xml_file_list
        self.frame_skip = frame_skip

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        
        self.target_sid = -1
        #randomly select xml file to load
        random_idx = np.random.randint(0,len(xml_file_list))
        #load initial env
        mujoco_env.MujocoEnv.__init__(self, xml_file_list[random_idx], frame_skip=frame_skip,mujoco_bindings="mujoco_py")
        self.target_sid = self.model.site_name2id("target")

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

       
    def calculate_forward_reward(self,x_velocity,y_velocity,xy_position_after):
        if((self.TARGET == [0,0]).all()):
        	return 0
        if(self.INCL_BODY_CENTER == True):
            position = np.array([self.sim.data.get_joint_qpos("slider1"), self.sim.data.get_joint_qpos("slider2")])
        else:
            position = self.sim.data.get_geom_xpos("link1")[0:2]
        vec = self.TARGET - position
        reward = -0.01*np.linalg.norm(vec)
        return reward
        
    def step(self, action):
        xy_position_before = self.sim.data.qpos[0:2].copy()
       
        #select controller based on env parameters
        self.do_simulation(action, self.frame_skip)
   
        xy_position_after = self.sim.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        #select reward based on env parameters
        forward_reward = self.calculate_forward_reward(x_velocity, y_velocity, xy_position_after)

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost

        done = False
        if(forward_reward > -0.005): # reward threshold for termination
            done = True
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()

        #construct angular positions and angular velocities vectors
        angles = np.zeros(self.NLINKS-1)
        angular_velocities = np.zeros(self.NLINKS-1)
        for i in range(self.NLINKS-1):
            angles[i] = self.sim.data.get_joint_qpos("rot"+str(i+1))
            angular_velocities[i] = self.sim.data.get_joint_qvel("rot"+str(i+1))

        #construct relative velocity readings
        current_link_angle = position[3] #angle of link1 relative to x axis
        relative_velocities = np.zeros(self.NLINKS*2)
        global_angles = np.zeros(self.NLINKS)
        global_angles[0] = current_link_angle
        for i in range(self.NLINKS):
            #fetch global x and y velocity of current link
            link_global_velocity = self.sim.data.get_geom_xvelp("link"+str(i+1))[0:2]
            #calculate relative velocity using angle of current link relative to x axis
            link_relative_velocity = link_global_velocity*np.cos(current_link_angle)
            relative_velocities[i*2] = link_relative_velocity[0]
            relative_velocities[i*2+1] = link_relative_velocity[1]
            #calculate angle of next link relative to x axis by adding offset from current to next link
            if(i!=self.NLINKS-1): 
            	current_link_angle += angles[i]
            	global_angles[i+1] = current_link_angle
            	
        observation = angles.copy()
        
        if(self.INCL_HEAD_POS):
            head_pos = self.sim.data.get_geom_xpos("link1")[0:2]
            observation = np.concatenate((head_pos, observation)).ravel()
        if(self.INCL_BODY_CENTER):
            body_center = np.array([self.sim.data.get_joint_qpos("slider1"), self.sim.data.get_joint_qpos("slider2")])
            observation = np.concatenate((body_center, observation)).ravel()
        if(self.INCL_OPT_ENCODER_VELOCITIES):
            observation = np.concatenate((observation, relative_velocities)).ravel()
        if(self.INCL_ANG_VELS):
            observation = np.concatenate((observation, angular_velocities))
        if(self.INCL_HEAD_ANGLE):
            observation = np.insert(observation, 0, position[3])
        if(self.INCL_TORQUES):
            observation = np.concatenate((observation,self.sim.data.qfrc_actuator[self.NLINKS-1:]))     
        
        observation = np.concatenate((observation, self.TARGET)).ravel()
        return observation
        
    def set_new_target(self, new_target):
        self.TARGET = new_target

    def reset_model(self):
        print("Resetting Episode ...")
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        #randomly select xml file to load
        print("...and generating new environment ...")
        random_idx = np.random.randint(0,len(self.xml_file_list))
        mujoco_env.MujocoEnv.__init__(self, self.xml_file_list[random_idx], frame_skip=self.frame_skip,mujoco_bindings="mujoco_py")
        
        qpos = self.init_qpos
        qpos[4:] = qpos[4:] + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq - 4)
        qvel = np.zeros(self.model.nv)

        self.set_state(qpos, qvel)
        	
        self.model.site_pos[self.target_sid] = np.append(self.TARGET,0)
        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
                
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_EP_LEN = 2000
LEARNING_RATE = 3e-4
N_EPOCHS = 3
BATCH_SIZE = 1024
TARGET = np.array([3,0])
INCL_HEAD_POS = True
INCL_BODY_CENTER = False
INCL_OPT_ENCODER_VELOCITIES = True
INCL_ANG_VELS = False
INCL_HEAD_ANGLE = True
INCL_TORQUES = False
FRAME_SKIP = 5
RESET_NOISE_SCALE = 0
NOTES = "Both networks have [256,256,256] neurons. Activation function is tanh. SAC algorithm used"
XML_list = ['landsnake_waypoint_obstacles_12-06_1.xml','landsnake_waypoint_obstacles_12-06_2.xml','landsnake_waypoint_obstacles_12-06_3.xml']


policy_kwargs = dict(net_arch=dict(pi=[256,256,256], qf=[256,256,256]))

env = gym.make('landsnake_env_variable-v0',
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

for ep in range(1):
    obs = env.reset()
    done = False
    while not done:

        env.render()
        obs, reward, done, info = env.step(np.random.uniform(size=4))

env.close()

