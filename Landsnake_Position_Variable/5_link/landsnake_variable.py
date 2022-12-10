import numpy as np
import gym
from gym.envs.mujoco import mujoco_env
from gym import utils

NLINKS = 5

DEFAULT_CAMERA_CONFIG = {}


class LandSnakeEnv_Variable(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file,
        frame_skip,
        NLINKS,
        FIXED_TARGET,
        TARGET = np.array([0,0]),
        INCL_HEAD_ANGLE=False,
        INCL_OPT_ENCODER_VELOCITIES=False,
        INCL_HEAD_POS=False,
        INCL_TARGET_SNAKE_DISPLACEMENT_VECTOR = False,
        INCL_BODY_CENTER = False,
        INCL_TORQUES=False,
        INCL_ANG_VELS=False,
        TARGET_RANGE_X = np.array([-4,4]),
        TARGET_RANGE_Y = np.array([-4,4]),
        OPT_ENC_PENALTY = 0,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
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
        self.INCL_TARGET_SNAKE_DISPLACEMENT_VECTOR = INCL_TARGET_SNAKE_DISPLACEMENT_VECTOR
        
        assert not (FIXED_TARGET == True and (TARGET == [0,0]).all()), "No waypoint provided"
        
        self.FIXED_TARGET = FIXED_TARGET
        self.TARGET = TARGET
        self.TARGET_LOW_X = TARGET_RANGE_X[0]
        self.TARGET_HIGH_X = TARGET_RANGE_X[1]
        self.TARGET_LOW_Y = TARGET_RANGE_Y[0]
        self.TARGET_HIGH_Y = TARGET_RANGE_Y[1]
        self.OPT_ENC_PENALTY = OPT_ENC_PENALTY
        self.OPT_ENC_X_VELOCITIES = np.zeros((self.NLINKS, 1))

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        
        self.target_sid = -1
        mujoco_env.MujocoEnv.__init__(self, xml_file, frame_skip=frame_skip,mujoco_bindings="mujoco_py")
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
        reward = forward_reward - ctrl_cost - self.OPT_ENC_PENALTY*np.linalg.norm(self.OPT_ENC_X_VELOCITIES)

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
            self.OPT_ENC_X_VELOCITIES[i] = link_relative_velocity[0] #update horizontal velocity of current link for penalty calculation
            #calculate angle of next link relative to x axis by adding offset from current to next link
            if(i!=self.NLINKS-1): 
            	current_link_angle += angles[i]
            	global_angles[i+1] = current_link_angle

        if(self.INCL_HEAD_POS):
            head_pos = self.sim.data.get_geom_xpos("link1")[0:2]
            observation = np.concatenate((head_pos, angles)).ravel()
        else:
            observation = angles.copy()
        
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
        if(self.INCL_TARGET_SNAKE_DISPLACEMENT_VECTOR):
            displacement = self.TARGET - self.sim.data.get_geom_xpos("link1")[0:2]
            print("observed displacement:", displacement)      
        else:
            observation = np.concatenate((observation, self.TARGET)).ravel()
        return observation
        
    def set_new_target(self, new_target):
        self.TARGET = new_target

    def reset_model(self):
        print("Resetting Episode ...")
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos
        qpos[4:] = qpos[4:] + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq - 4)
        qvel = np.zeros(self.model.nv)

        self.set_state(qpos, qvel)
        
        if(self.FIXED_TARGET == False):
        	target = np.array([np.random.uniform(low=self.TARGET_LOW_X, high=self.TARGET_HIGH_X),
        				np.random.uniform(low=self.TARGET_LOW_Y, high=self.TARGET_HIGH_Y)])
        	print("New target: ", target)
        	self.TARGET = target
        	
        self.model.site_pos[self.target_sid] = np.append(self.TARGET,0)
        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

