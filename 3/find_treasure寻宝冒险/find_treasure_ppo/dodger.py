# ==============================================================================
# MODULES
# ==============================================================================
from torch.utils.tensorboard import SummaryWriter

import ppo
import torch
import numpy as np
try:
    from malmo import MalmoPython
except:
    import MalmoPython				# Malmo
import logging						# world debug prints
import time							# sleep for a few ticks every trial
import random						# random chance to choose actions
import world						# world observation
import sys							# max int tau
from collections import deque		# states/actions/rewards history
writer = SummaryWriter('runs')

damage_reward = -20
Complete_reward = 18
Avoid_reward = 4		# avoid the arrow, so scale up the reward
Waiting_reward = 0.9		# wait but wasting time, so scale down the reward
# ==============================================================================
# AI class
# ==============================================================================
class Dodger(object):
	def __init__(self, agent_host, num_arrow, alpha=0.001, gamma=.95, n=1):
		self.agent_host = agent_host	# init in main
		self.alpha = alpha				# learning rate
		self.gamma = gamma				# value decay rate
		self.n = n						# number of back steps to update
		self.epsilon = 0.15			# chance of taking a random action
		self.q_table = {}				
		self.start_pos = None			# init in world.refresh(...)
		self.dispenser_pos = None		# init in world.refresh(...)
		self.life = 0					# init in world.refresh(...)
		self.sleep_time	= 0.05			# time to sleep after action NOTE:0.05
		self.num_state = 1
		self.net = None
		self.start_z = -316.5
		# self.start_x = self.curr_x= 415.5
	# USE FOR 1 ARROW TESTING PURPOSES ONLY
	def print_1arrow_q_table(self, moveable_blocks, possible_arrow_x_pos):
		"""	prints a formatted q-table for an 1 arrow run
			args:	moveable blocks		(blocks the agent can walk on (rows))
					arrow x positions	(possible arrow x positions (columns))
		"""
		

	# USE FOR 1 ARROW HARD CODED RUN TESTING PURPOSES ONLY 
	def print_hc_wr_table(self, wait_block, possible_arrow_x_pos, wr_table):
		"""	prints a formatted win-rate table for a hard coded 1 arrow run
			args:	wait block			(block the agent can walk on (rows))
					arrow x positions	(possible arrow x positions (columns))
					win-rate table		(win-rates per possible arrow x pos)
		"""
		# print arrow x positions (x-axis)
		
		
		# print wait block (y-axis)
		
		
		# print win-rate for each arrow x position
		
		
	def update_q_table(self, tau, S, A, R, T):
		"""	performs relevant updates for state tau
		
			args:	tau				(integer state index to update)
					states deque	
					actions deque	
					rewards deque	
					term state index
		"""
		# upon terminating state, A is empty
		
		
		# calculate q value based on the most recent state/action/reward
		

	def get_reward(self, obs, prev_action):
		"""	get reward based on distance, life, action, and arrow avoidance
			args:	world observation	(use world.get_observations(...))
					prev_action			(use self.get_action(...))
			return:	reward value		(float)
					success flag		(True / False / None = still in progress)
		"""
		# reward = distance from start position * the following multipliers
		 
		
		# initialize reward multipliers and success flag
		
		# damaged: extremely low reward and success = False
		

		# complete: extremely high reward and success = True

		# waited: scale down reward
		
		
		# avoided arrow: scale up reward
		

	# def state2tensor(self, curr_state):
	# 	''' state :tuple(.., [.., ..])->tensor that canbe sent into net		
	# 	'''
	# 	state = [curr_state[0]]
	# 	for item in curr_state[1]:
	# 		state.append(item)
	# 	return torch.tensor(state)

	def get_action(self, possible_actions, policy):
		"""	get best action using epsilon greedy policy
			args:	current state		(use self.get_curr_state(obs))
					possible actions	(["move 1", "move 0"])
			return:	action				("move 1" or "move 0")
		"""
		# NOTE
		# new state
		# NOTE: maybe make it so agent always moves in a new state?
		if random.random() < self.epsilon:
			action = possible_actions[random.randint(0, 1)]
		else:
			p = policy.detach().numpy().tolist()
			action_num = int(np.random.choice(2, 1, p=[p[0],1-p[0]]))
			action = possible_actions[action_num]
		# chance to choose a random action
		# NOTE: maybe random chance to move instead?
		return action
			
		# get the best action based on the greatest q-val(s)
		

	def get_curr_state(self, obs):
		"""	get a simplified, integer-based version of the environment
			args:	world observations	(use world.get_observations(...))
			return:	state 				((curr z, arrow₁ x, arrow₂ x, ...))
		"""
		# get current z-position rounded down
		agent_pos = world.get_curr_pos(obs)
		print(float(agent_pos['z']))
		agent_cur_z = float(agent_pos['z']) - self.start_z
		self.curr_x = float(agent_pos['x'])
		# get arrow x-positions, ordered by increasing z-positions
		arrow_dic = world.get_arrow_pos(obs)
		# print('arrow_dic', arrow_dic)
		arrow_dic_keys = []
		# get agent_life
		agent_life = world.get_curr_life(obs)
		# arrow_dic_keys = [int(key) for key in arrow_dic.keys()].sort()
		for key in arrow_dic.keys():
			# print('key:',int(key))
			arrow_dic_keys.append(int(key))
			arrow_dic[int(key)] -= 451.5
		arrow_x_positions = []
		print('arrow_dic_keys:',arrow_dic_keys)
		# arrow_dic_keys.sort()
		# print('now agent"s z:', )
		state = [0.]*4
		state[0] = agent_cur_z
		for (index, item) in enumerate(self.dispenser_pos):
			# print("item", type(index))
			# print("int(item[int(index)])", item[])
			if int(item[2]) in arrow_dic_keys:
				state[index+1] = 1.0
		return state, agent_life
	
	# def is_damaged(self, state, next_state, arrow_dic, pre_life):
	def is_damaged(self, pre_life):
		# for key in arrow_dic.keys():
		# 	if (state[0] + self.start_z <= key and key <= next_state[0] + self.start_z \
		# 		and 450 <= arrow_dic[key] + 446 and arrow_dic[key] + 446 <= 453):	# 之前是451-452
		# 		return True
		# return False
		return self.life < pre_life
		# return self.curr_x != self.start_x

	def is_complete(self, next_state):
		return next_state[0] >= 8.0

	def get_distance(self, next_state):
		return abs(next_state[0])

	def step(self, next_state, action, pre_life):
		multireward = 1
		success = False
		done = False
		if action == 1:	# 向前进
			if self.is_damaged(pre_life):
				multireward *= damage_reward
				success = False
				done = True
			elif self.is_complete(next_state):
				multireward *= Complete_reward
				success = True
				done = True
			else:
				multireward *= Avoid_reward
				success = done = False

		else:		# waiting
			if self.is_damaged(pre_life):
				multireward *= damage_reward
				success = False
				done = True
			elif self.is_complete(next_state):
				multireward *= Complete_reward
				success = True
				done = True
			else:
				multireward *= Waiting_reward
				success = done = False
		reward = multireward * self.get_distance(next_state)
		return reward, done, success



	def run(self):
		"""	observations → state → act → reward ↩, and update q table
			return:	total reward		(cumulative int reward value of the run)
					success flag		(True / False)
		"""

		self.num_state = 1 + len(self.dispenser_pos)
		self.net = ppo.PPO(2, 1, self)
		MAX_STEP = 500
		done = False# 完成条件被箭射死或者到达终点
		total_loss = 0
		total_reward = 0
		success = False
		observations = world.get_observations(self.agent_host)
		state, self.life = self.get_curr_state(observations)
		pre_life = self.life
		while not done:
			for i in range(MAX_STEP):
				print('============rep============:', i)
				policy = self.net.PolicyNet(torch.tensor(state).float().reshape(1, -1), softmax_dim=1).squeeze()
				a = self.get_action(["move 0", "move 1"], policy)	# "move 1" or "move 0"
				self.agent_host.sendCommand(a)
				time.sleep(self.sleep_time)
				a = 1 if '1' in a else 0		# 1 or 0
				observations = world.get_observations(self.agent_host)
				next_state, self.life = self.get_curr_state(observations)
				r, done, success = self.step(next_state, a, pre_life)	# 获得奖励、完成情况
				total_reward += r
				if self.life == 0:
					done = True
					success = False
				# 另外实现一个判断damaged的办法：比较上次和现在的life值，如果减小则说明damaged
				print('action:', a)
				print('reward:', r)
				if done:
					if success:
						print('===========WIN===========')
					else:
						print('===========LOSE===========')
				data = [state, a, r, next_state, policy[a], done]
				self.net.put_data(data)
				state, pre_life = next_state, self.life
				if done:
					self.agent_host.sendCommand('move 0')
					break
				if i >= 3:
					loss=self.net.train_net()
					writer.add_scalar('loss', loss, i)
					total_loss += loss
		
		return total_reward, success,total_loss
		

		# history of states/actions/rewards

		# either you move or you don't
		
		# returns total reward and success flag
		
		# initialize terminating state 

		# run until damaged


			# get initial state/action/reward


			# continuously get observations
	
				# death or out of bounds glitching ends the run
				
					# episode finish: end state and get final reward
					
					
					# episode running: act and get state/action/reward
					
						# act (move or wait)
						
						
						# get reward and check if episode is finished 
						
						
						# get state/action
		
				# end of episode: update q table
		# return None, None			

	def hard_coded_run(self, wait_block, arrow_x_pos):
		"""	guarantee move when agent on wait_block and arrow on arrow_x_pos
			return:	success flag		(True / False)
		"""	
	

			# get initial state/action/reward
			
			# death or out of bounds glitching ends the run
			
			# act
			
			# win/lose condition

