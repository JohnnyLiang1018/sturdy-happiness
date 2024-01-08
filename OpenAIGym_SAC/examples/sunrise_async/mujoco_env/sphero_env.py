import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import time

#from Sphero import Sphero
#from Environment import Environment


class SpheroEnv(gym.Env):

	def __init__(self, agent):
		# Configurations
		self.xml_path = 'mujoco/field.xml' #xml file (assumes this is in the same folder as this file)
		self.simend = 40 #simulation time
		self.print_camera_config = 0 #set to 1 to print camera config
						
		#this is useful for initializing view of the model)
		# self.observation_space = spaces.Dict({
		# 	"agent": spaces.Box(0,15.3, (2,)),
		# 	"target": spaces.Box(0, 15.3, (2,))
		# })
		self.observation_space = spaces.Box(low=np.array([-10,-10, -180, -10, -10]), high=np.array([160, 160, 180, 160, 160]), shape=(5,))
		self.action_space = spaces.Box(low=np.array([-180]), high=np.array([180]))

		# For callback functions
		self.button_left = False
		self.button_middle = False
		self.button_right = False
		self.lastx = 0
		self.lasty = 0

		# Agent
		self.agent = agent
		self.agent_x = None
		self.agent_y = None
		self.forward = None
		self.current_action = 0
		self.action_list = [0.75,0.75]

		# Target
		self.target_x = None
		self.target_y = None

		# get the full path
		self.dirname = os.path.dirname(__file__)
		self.abspath = os.path.join(self.dirname + "/" + self.xml_path)
		self.xml_path = self.abspath

		# MuJoCo data structures
		self.model = mj.MjModel.from_xml_path(self.xml_path)  # MuJoCo model
		self.data = mj.MjData(self.model)                		# MuJoCo data
		self.cam = mj.MjvCamera()                        # Abstract camera
		self.opt = mj.MjvOption()                        # visualization options

	def move_and_rotate(self, current_coords, action):
		angle = (action + 1) * 180 
		self.forward = angle
		rad = math.radians(angle)
		x, y, z = current_coords
		# print(angle)
		x_prime = math.cos(rad)
		y_prime = -math.sin(rad)
		z_prime = 0

		return [x_prime, y_prime, z_prime]

	def Is_ball_touched(self):
		for i in range(len(self.data.contact.geom1)):
			if (data.geom(self.data.contact.geom1[i]).name == "ball_g" and self.data.geom(self.data.contact.geom2[i]).name == "sphero1") or (data.geom(data.contact.geom2[i]).name == "ball_g" and data.geom(data.contact.geom1[i]).name == "sphero1"):
				return 100

	def distance_bw_ball_n_sphero(self):
		return np.linalg.norm(self.data.xpos[2] - self.data.xpos[3])

	def compute_reward(self):
		distance = self.distance_bw_ball_n_sphero()
		return -distance

	def get_done(self):
		done = False
		if np.linalg.norm(np.array([self.agent_x, self.agent_y]) - np.array([self.target_x, self.target_y])) < 1.5:
			done = True
		return done


	def render_it(self):
		# self.reset()
		# mj.mj_resetData(self.model, self.data)
		# mj.mj_forward(self.model, self.data)
		#start_agent_x=random.uniform(-45, 45)
		#start_agent_y=random.uniform(-30, 30)
		#start_ball_x=random.uniform(-45, 45)
		#start_ball_y=random.uniform(-30, 30)
		# self.agent_x=-0.15
		# self.agent_y=-2.0
		# self.target_x=0
		# self.target_y=0
		# self.data.qpos[:2]=[self.agent_x, self.agent_y]
		# self.data.qpos[7:9]=[self.target_x, self.target_y]
		# Init GLFW, create window, make OpenGL context current, request v-sync
		glfw.init()
		window = glfw.create_window(1200, 900, 'RL Team - Soccer Game', None, None)
		glfw.make_context_current(window)
		glfw.swap_interval(1)

		# state =np.array([start_agent_x, start_agent_y, 0, 0, 0, start_ball_x, start_ball_y, 0, 0])
		state = np.array([self.agent_x, self.agent_y, 0, self.target_x, self.target_y])
		self.forward = 0

		# initialize visualization data structures
		mj.mjv_defaultCamera(self.cam)
		mj.mjv_defaultOption(self.opt)
		scene = mj.MjvScene(self.model, maxgeom=10000)
		context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

		# Callback functions
		def keyboard(window, key, scancode, act, mods):
			if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
				mj.mj_resetData(self.model, self.data)
				mj.mj_forward(self.model, self.data)
		
		def mouse_button(window, button, act, mods): 
			# update button state
			global button_left
			global button_middle
			global button_right

			button_left = (glfw.get_mouse_button(
				window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
			button_middle = (glfw.get_mouse_button(
				window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
			button_right = (glfw.get_mouse_button(
				window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

			# update mouse position
			glfw.get_cursor_pos(window)

		def mouse_move(window, xpos, ypos):
			# compute mouse displacement, save
			global lastx
			global lasty
			global button_left
			global button_middle
			global button_right

			dx = xpos - lastx
			dy = ypos - lasty
			lastx = xpos
			lasty = ypos

			# no buttons down: nothing to do
			if (not button_left) and (not button_middle) and (not button_right):
				return

			# get current window size
			width, height = glfw.get_window_size(window)

			# get shift key state
			PRESS_LEFT_SHIFT = glfw.get_key(
				window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
			PRESS_RIGHT_SHIFT = glfw.get_key(
				window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
			mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

			# determine action based on mouse button
			if button_right:
				if mod_shift:
					action = mj.mjtMouse.mjMOUSE_MOVE_H
				else:
					action = mj.mjtMouse.mjMOUSE_MOVE_V
			elif button_left:
				if mod_shift:
					action = mj.mjtMouse.mjMOUSE_ROTATE_H
				else:
					action = mj.mjtMouse.mjMOUSE_ROTATE_V
			else:
				action = mj.mjtMouse.mjMOUSE_ZOOM

			mj.mjv_moveCamera(self.model, action, dx/height,
						dy/height, scene, cam)

		def scroll(window, xoffset, yoffset):
			action = mj.mjtMouse.mjMOUSE_ZOOM
			mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, scene, self.cam)

		# install GLFW mouse and keyboard callbacks
		glfw.set_key_callback(window, keyboard)
		glfw.set_cursor_pos_callback(window, mouse_move)
		glfw.set_mouse_button_callback(window, mouse_button)
		glfw.set_scroll_callback(window, scroll)

		self.cam.azimuth = 90.38092929594274
		self.cam.elevation = -70.15643645584721
		self.cam.distance =  109.83430075014073
		self.cam.lookat =np.array([ 0.33268787911150655 , -2.0371257758709908e-17 , -2.6127905178878716 ])

		score=[]
		actions = []
		states = []
		rewards = []
		done = []
		next_states = []

		command_time = time.time() + 1
		while not glfw.window_should_close(window):
			a_pos, b_pos= self.data.xpos[2], self.data.xpos[3]
			agent_x, agent_y, agent_z = a_pos
			# ball_x, ball_y, ball_z = b_pos
			#print(data.qvel)
			# agent_vx, agent_vy= self.data.qvel[:2]
			# ball_vx, ball_vy= self.data.qvel[7:9]
			# Select an action using the agent's policy
			# TODO
			# action = self.agent.act(state)[0]
			# states.append(state) ##
			# actions.append(action) ##
			# angle, speed=action

			
			# direction= self.move_and_rotate(self.data.xpos[2], angle)
			# direction = np.array(direction[:2])
			# direction /= np.linalg.norm(direction)  # normalize the velocity vector
			print(self.agent_x, self.agent_y)
			if time.time() > command_time:
				action = float(input("Press to continue"))
			# action = (np.random.random() - 0.5) * 2
			# action = self.action_list[self.current_action % 2]
			# self.current_action += 1
				command_time = time.time() + 1
				obs, reward, done, _ = self.step([action, 0])

				next_state = obs
				next_states.append(next_state) ##
				score.append(reward)
				state=next_state

				if done == True:
					print("Reach Target")
					break
			# print(direction)
			# self.data.qvel[:2] = 5 * direction

			# print("state: ", state)
			# print("action: ", action)
			# print("reward: ", reward)

			# time_prev = self.data.time
			# while (self.data.time - time_prev < 1/200):
			# 	mj.mj_step(self.model, self.data)
				#angle, speed = env.action_space.sample()



				# if goal or foul:
				# 	break
			
				# if goal or foul:
				# 	break

				# End simulation based on time
				# if (self.data.time>=simend):
				# 	break
			
			# time_prev = self.data.time
			# self.data.qvel[:2] = 0
			# while (self.data.time - time_prev < 1/100):
			# 	mj.mj_step(self.model, self.data)
			
			# agent_x_ , agent_y_, _= self.data.xpos[2]
			# target_x_, target_y_, _ = self.data.xpos[3]
			# reward = self.compute_reward()
			# rewards.append(reward) ##
			# next_state = np.array([agent_x, agent_y, agent_vx, agent_vy, forward, ball_x, ball_y, ball_vx, ball_vy])
			# next_state = np.array([agent_x_, agent_y_, self.forward, target_x_, target_y_])


			# print("Move x distance", agent_x_ - agent_x)
			# print("Move y distance", agent_y_ - agent_y)

			# print("Used time", self.data.time-time_prev)
			# get framebuffer viewport
			viewport_width, viewport_height = glfw.get_framebuffer_size(window)
			viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

			#print camera configuration (help to initialize the view)
			if (print_camera_config==1):
				print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
				print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

			# Update scene and render
			mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
			mj.mjr_render(viewport, scene, context)

			# swap OpenGL buffers (blocking call due to v-sync)
			glfw.swap_buffers(window)

			# process pending GUI events, call GLFW callbacks
			glfw.poll_events()
		glfw.terminate()

		return states, actions, rewards, next_states
		# return sum(score), score

    # Callback functions
	def keyboard(self, window, key, scancode, act, mods):
		if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
			mj.mj_resetData(self.model, self.data)
			mj.mj_forward(self.model, self.data)

	def mouse_button(window, button, act, mods):
		# update button state
		global button_left
		global button_middle
		global button_right

		button_left = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
		button_middle = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
		button_right = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

		# update mouse position
		glfw.get_cursor_pos(window)

	def step(self, action):
		direction= self.move_and_rotate(self.data.xpos[2], action[0])
		direction = np.array(direction[:2])
		direction /= np.linalg.norm(direction)  # normalize the velocity vector
		prev_reward = self.compute_reward()
		# print(self.agent_x, self.agent_y)
		self.data.qvel[:2] = 5 * direction
		start_time = self.data.time
		while(self.data.time - start_time < 0.7):
			mj.mj_step(self.model, self.data)
		# input("Wait for input")
		
		self.agent_x , self.agent_y, _ = self.data.xpos[2]
		self.target_x, self.target_y, _ = self.data.xpos[3]
		agent_x, agent_y, target_x, target_y = self.xy_normalize(self.agent_x, self.agent_y, self.target_x, self.target_y)
		obs = np.array([agent_x, agent_y, self.forward, target_x, target_y],dtype=np.float32)
		# print(agent_x, agent_y)
		# input("step completed")
		reward = self.compute_reward() - prev_reward
		# print("Obs", obs)
		# print("Reward", reward)
		done = self.get_done()
		# print("Done", done)

		return obs, reward*10, done, {}

	def reset(self):
		#start_agent_x=random.uniform(-45, 45)
		#start_agent_y=random.uniform(-30, 30)
		#start_ball_x=random.uniform(-45, 45)
		#start_ball_y=random.uniform(-30, 30)
		mj.mj_resetData(self.model, self.data)
		self.agent_x=random.uniform(-7.65, 7.65)
		self.agent_y=random.uniform(-7.65, 7.65)
		self.target_x=random.uniform(-7.65, 7.65)
		self.target_y=random.uniform(-7.65, 7.65)
		self.data.qpos[:2]=[self.agent_x, self.agent_y]
		self.data.qpos[7:9]=[self.target_x, self.target_y]
		mj.mj_forward(self.model, self.data)

		agent_x, agent_y, target_x, target_y = self.xy_normalize(self.agent_x, self.agent_y, self.target_x, self.target_y)
		obs = np.array([agent_x, agent_y, 0, target_x, target_y], dtype=np.float32)
		info = {}

		score=[]
		actions = []
		states = []
		rewards = []
		done = []
		next_states = []
		return obs
	
	def render(self):
		return self.render_it()

	def set_initial_position(self, agent_x, agent_y, target_x, target_y):
		self.agent_x = agent_x
		self.agent_y = agent_y
		self.target_x = target_x
		self.target_y = target_y
		# agent_x, agent_y, agent_z = self.data.xpos[2]
		# target_x, target_y, target_z = self.data.xpos[3]
		# self.data.xpos[2] = [self.agent_x, self.agent_y, agent_z]
		# self.data.xpos[3] = [self.target_x, self.target_y, target_z]
		self.data.qpos[:2] = [self.agent_x, self.agent_y]
		self.data.qpos[7:9] = [self.target_x, self.target_y]
		mj.mj_forward(self.model, self.data)


	def xy_normalize(self, agent_x, agent_y, target_x, target_y):
		agent_x += 7.65
		agent_y += 7.65
		target_x += 7.65
		target_y += 7.65
		return agent_x*10, agent_y*10, target_x*10, target_y*10 

class Soccer(gym.Env):


    def __init__(self):
        # Define the action space
        # The first action is the angle of rotation (-π to π)
        # The second action is the direction of movement (0: stop, 1: forward)
        self.action_space = spaces.Box(
                                        low=np.array([-np.pi, 0], dtype=np.float32),
                                        high=np.array([np.pi, 1], dtype=np.float32),
                                        dtype=np.float32
                                      )

        # Define the observation space
        # The observation space has 10 dimensions:
        # 1. Agent x-coordinate
        # 2. Agent y-coordinate
        # 3. Agent x-velocity
        # 4. Agent y-velocity
        # 5. Agent angle with respect to x-axis (-pi to pi)
        # 6. Ball x-coordinate
        # 7. Ball y-coordinate
        # 8. Ball x-velocity
        # 9. Ball y-velocity
        low = np.array([-45, -30, 0, -45, -30], dtype=np.float32)
        high = np.array([45, 30, 360, 45, 30], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

# Actor network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

# Critic network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Replay buffer
class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.max_size = BUFFER_SIZE
        self.ptr = 0
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.max_size
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward.reshape(-1, 1), next_state, done.reshape(-1, 1)

class DDPG:
    def __init__(self, state_size, action_size, hidden_size):
        self.actor = Actor(state_size, action_size, hidden_size).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(state_size, action_size, hidden_size).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.replay_buffer = ReplayBuffer()
        

    def act(self, state, epsilon=0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            
            action = self.actor(state).cpu().data.numpy()
        return action
    
    def train(self):
        if len(self.replay_buffer.buffer) < BATCH_SIZE:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # Update critic
        #print(action)
        Q = self.critic(state, action)
        next_action = self.target_actor(next_state)
        next_Q = self.target_critic(next_state, next_action.detach())
        target_Q = reward + GAMMA * next_Q * (1 - done)
        critic_loss = nn.MSELoss()(Q, target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(TAU * source.data + (1 - TAU) * target.data)
        for target, source in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(TAU * source.data + (1 - TAU) * target.data)
        
    def update_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.target_actor = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
        self.target_critic = copy.deepcopy(self.critic)



env = Soccer()



def Is_ball_touched():
	for i in range(len(data.contact.geom1)):
		if (data.geom(data.contact.geom1[i]).name == "ball_g" and data.geom(data.contact.geom2[i]).name == "sphero1") or (data.geom(data.contact.geom2[i]).name == "ball_g" and data.geom(data.contact.geom1[i]).name == "sphero1"):
			#print("touched_ball")
			return 100
	return 0
boundaries=( "Touch lines1", "Touch lines2", "Touch lines3", "Touch lines4", "Touch lines5", "Touch lines6",)
def Is_boundaries_touched():
	for i in range(len(data.contact.geom1)):
		if (data.geom(data.contact.geom1[i]).name == "sphero1" and data.geom(data.contact.geom2[i]).name in boundaries) or (data.geom(data.contact.geom2[i]).name == "sphero1" and data.geom(data.contact.geom1[i]).name in boundaries):
			#print("touched_boundary")
			#print(data.xpos[8])
			return -10000
	return 0
Goal=("Goal lines1", "Goal lines2")
def Is_goal():
	for i in range(len(data.contact.geom1)):
		if (data.geom(data.contact.geom1[i]).name == "ball_g" and data.geom(data.contact.geom2[i]).name in Goal) or (data.geom(data.contact.geom2[i]).name == "ball_g" and data.geom(data.contact.geom1[i]).name in Goal):
			#print("Goal!!!")
			return 1000
	return 0
def Is_goal_sphero():
	for i in range(len(data.contact.geom1)):
		if (data.geom(data.contact.geom1[i]).name == "sphero1" and data.geom(data.contact.geom2[i]).name in Goal) or (data.geom(data.contact.geom2[i]).name == "sphero1" and data.geom(data.contact.geom1[i]).name in Goal):
			#print("Sphero Goal!!!")
			return -100
	return 0
def distance_bw_goal1_n_ball():
		# define the line by two points a and b
		a = np.array([45, 5, 0])
		b = np.array([45, -5, 0])
		# define the point p
		p = data.xpos[2]
		# calculate the distance
		distance = np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(b - a)
		return distance
def distance_bw_goal2_n_ball():
		# define the line by two points a and b
		a = np.array([-45, 5, 0])
		b = np.array([-45, -5, 0])
		# define the point p
		p = data.xpos[2]
		# calculate the distance
		distance = np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(b - a)
		return distance
def distance_bw_ball_n_sphero():
    return np.linalg.norm(data.xpos[2] - data.xpos[3])


# Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
WEIGHT_DECAY = 0.0001

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#print(data.xpos[8])
#position=move_and_rotate(data.xpos[8], 0)
#position=move_and_rotate(data.xpos[8], -2.800366)#[-,-]
#position=move_and_rotate(data.xpos[8], 2.800366)#[-,+]
#position=move_and_rotate(data.xpos[8], 0.8782678)#[+,+]
#position=move_and_rotate(data.xpos[8], -0.8782678)#[+,-]
#print(position)

#direction = np.array(position[:2])
#direction /= np.linalg.norm(direction)  # normalize the velocity vector
#data.qvel[:2] =  direction

# Configurations
xml_path = 'mujoco/field.xml' #xml file (assumes this is in the same folder as this file)
simend = 40 #simulation time
print_camera_config = 0 #set to 1 to print camera config
						#this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0




# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                		# MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options
def render_it(env):
	mj.mj_resetData(model, data)
	mj.mj_forward(model, data)
	#start_agent_x=random.uniform(-45, 45)
	#start_agent_y=random.uniform(-30, 30)
	#start_ball_x=random.uniform(-45, 45)
	#start_ball_y=random.uniform(-30, 30)
	start_agent_x=0
	start_agent_y=0
	start_ball_x=5
	start_ball_y=0
	data.qpos[:2]=[start_agent_x, start_agent_y]
	data.qpos[7:9]=[start_ball_x, start_ball_y]
	# Init GLFW, create window, make OpenGL context current, request v-sync
	glfw.init()
	window = glfw.create_window(1200, 900, 'RL Team - Soccer Game', None, None)
	glfw.make_context_current(window)
	glfw.swap_interval(1)

	state=np.array([start_agent_x, start_agent_y, 0, 0, 0, start_ball_x, start_ball_y, 0, 0])

	# initialize visualization data structures
	mj.mjv_defaultCamera(cam)
	mj.mjv_defaultOption(opt)
	scene = mj.MjvScene(model, maxgeom=10000)
	context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    # Callback functions
	def keyboard(window, key, scancode, act, mods):
		if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
			mj.mj_resetData(model, data)
			mj.mj_forward(model, data)

	def mouse_button(window, button, act, mods):
		# update button state
		global button_left
		global button_middle
		global button_right

		button_left = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
		button_middle = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
		button_right = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

		# update mouse position
		glfw.get_cursor_pos(window)

	def mouse_move(window, xpos, ypos):
		# compute mouse displacement, save
		global lastx
		global lasty
		global button_left
		global button_middle
		global button_right

		dx = xpos - lastx
		dy = ypos - lasty
		lastx = xpos
		lasty = ypos


		# no buttons down: nothing to do
		if (not button_left) and (not button_middle) and (not button_right):
			return

		# get current window size
		width, height = glfw.get_window_size(window)

		# get shift key state
		PRESS_LEFT_SHIFT = glfw.get_key(
			window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
		PRESS_RIGHT_SHIFT = glfw.get_key(
			window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
		mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

		# determine action based on mouse button
		if button_right:
			if mod_shift:
				action = mj.mjtMouse.mjMOUSE_MOVE_H
			else:
				action = mj.mjtMouse.mjMOUSE_MOVE_V
		elif button_left:
			if mod_shift:
				action = mj.mjtMouse.mjMOUSE_ROTATE_H
			else:
				action = mj.mjtMouse.mjMOUSE_ROTATE_V
		else:
			action = mj.mjtMouse.mjMOUSE_ZOOM

		mj.mjv_moveCamera(model, action, dx/height,
						dy/height, scene, cam)

	def scroll(window, xoffset, yoffset):
		action = mj.mjtMouse.mjMOUSE_ZOOM
		mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

	# install GLFW mouse and keyboard callbacks
	glfw.set_key_callback(window, keyboard)
	glfw.set_cursor_pos_callback(window, mouse_move)
	glfw.set_mouse_button_callback(window, mouse_button)
	glfw.set_scroll_callback(window, scroll)

	cam.azimuth = 90.38092929594274
	cam.elevation = -70.15643645584721
	cam.distance =  109.83430075014073
	cam.lookat =np.array([ 0.33268787911150655 , -2.0371257758709908e-17 , -2.6127905178878716 ])
	score=[]
	actions = []
	states = []
	rewards = []
	done = []
	next_states = []
	
	while not glfw.window_should_close(window):
		time_prev = data.time

		while (data.time - time_prev < 1/60.0):
			forward=state[4]
			mj.mj_step(model, data)
			#angle, speed = env.action_space.sample()
			
			# Select an action using the agent's policy
			action = agent.act(state)[0]
			states.append(state) ##
			actions.append(action) ##
			#print(action)
			#print(action)
			angle, speed=action
			forward, direction=env.move_and_rotate(data.xpos[2], angle, forward)
			direction = np.array(direction[:2])
			direction /= np.linalg.norm(direction)  # normalize the velocity vector
			data.qvel[:2] = speed * direction
			reward = env.compute_reward()
			rewards.append(reward) ##

			a_pos, b_pos=data.xpos[2], data.xpos[3]
			agent_x, agent_y, agent_z = a_pos
			ball_x, ball_y, ball_z = b_pos
			#print(data.qvel)
			agent_vx, agent_vy=data.qvel[:2]
			ball_vx, ball_vy=data.qvel[7:9]
			next_state=np.array([agent_x, agent_y, agent_vx, agent_vy, forward, ball_x, ball_y, ball_vx, ball_vy])
			next_states.append(next_state) ##
			agent.update_replay_buffer(state, action, reward, next_state, 0.0)
			agent.train()
			score.append(reward)
			state=next_state
		# 	if goal or foul:
		# 		break
		
		# if goal or foul:
		# 	break

		# End simulation based on time
		if (data.time>=simend):
			break

		# get framebuffer viewport
		viewport_width, viewport_height = glfw.get_framebuffer_size(window)
		viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

		#print camera configuration (help to initialize the view)
		if (print_camera_config==1):
			print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
			print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

		# Update scene and render
		mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
		mj.mjr_render(viewport, scene, context)

		# swap OpenGL buffers (blocking call due to v-sync)
		glfw.swap_buffers(window)

		# process pending GUI events, call GLFW callbacks
		glfw.poll_events()
	glfw.terminate()

	return states, actions, rewards, next_states
	# return sum(score), score


# state_size = env.observation_space.shape[0]
# action_size = env.action_space.shape[0]
# hidden_size = 256
# agent = DDPG(state_size, action_size, hidden_size)
# sphero_env = SpheroEnv(agent)
# for i in range(1000):
# 	score, s_list = sphero_env.render_it()
# 	print(f"Episode {i+1} has the score of: {score} ")
# 	if i%50==0 and i!=0:
# 		agent.save(f"kicha_{i}")

if __name__ == "__main__":
	env = SpheroEnv("placehold")
	env.reset()
	env.set_initial_position(1, 0, -1, 0)
	env.render_it()
	# print("Qpos", env.data.qpos[:2], env.data.qpos[7:9])
	# print("Xpos", env.data.xpos[2], env.data.xpos[3])
	# obs, reward, done, _ = env.step([-1])
	# print(obs, reward)