# ==============================================================================
# MODULES
# ==============================================================================
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
import numpy

# ==============================================================================
# AI class
# ==============================================================================
class Dodger(object):
	def __init__(self, agent_host, alpha=0.4, gamma=.95, n=1):
		self.agent_host = agent_host	# init in main
		self.alpha = alpha				# learning rate
		self.gamma = gamma				# value decay rate
		self.n = n						# number of back steps to update
		self.epsilon = 0.1			# chance of taking a random action
		self.q_table = {}				
		self.start_pos = None			# init in world.refresh(...)
		self.dispenser_pos = None		# init in world.refresh(...)
		self.life = 0					# init in world.refresh(...)
		self.sleep_time	= 0.05			# time to sleep after action
		
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
		pass
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
		if len(A) == 0:
			A.append('move 0')
	# NOTE: maybe make it so agent always moves in a new state?
		# calculate q value based on the most recent state/action/reward
		current_A,current_S,current_R = A.popleft(),S.popleft(),R.popleft()
		TD_target= current_R + self.gamma * numpy.max([self.q_table[S[-1]][a] for a in ['move 0',  'move 1']])
		TD_delta = TD_target - self.q_table[current_S][current_A]
		self.q_table[current_S][current_A] += self.alpha * TD_delta

	def get_reward(self, obs, prev_action):
		"""	get reward based on distance, life, action, and arrow avoidance
			args:	world observation	(use world.get_observations(...))
					prev_action			(use self.get_action(...))
			return:	reward value		(float)
					success flag		(True / False / None = still in progress)
		"""

	# reward = distance from start position * the following multipliers
		current_position=world.get_curr_pos(obs)
		distance=current_position['z']-self.start_pos['z']
		cumulative_multiplier= 1

	# initialize reward multipliers and success flag
		DAMAGE=-300 # if agent is damaged
		COMPLETE =+100
		WAIT=0.9 #等待
		avoid_arrow=9.9 #避开箭
		success=None  #是否完成

	# damaged: extremely low reward and success = False
		if current_position['x'] != self.start_pos['x'] :
			success=False
			cumulative_multiplier*=DAMAGE

	# complete: extremely high reward and success = True
		view_ahead =obs.get(u'view_ahead', 0)
		if view_ahead[0]=='chest':
			success=True
			cumulative_multiplier*=COMPLETE  #若游戏视野第一个为宝箱，则成功！！！

	# waited: scale down reward
		if prev_action=='move 0':
			cumulative_multiplier*=WAIT

	# avoided arrow: scale up reward 要预判是否可能在箭可能射中的位置
		possible_arrow_z_pos= [dispenser[2] for dispenser in self.dispenser_pos] #TODO: discuss in here!
		if int(current_position['z'])-2 in possible_arrow_z_pos:
			cumulative_multiplier*=avoid_arrow

		return distance*cumulative_multiplier,success

	def get_action(self, curr_state, possible_actions):
		"""	get best action using epsilon greedy policy
			args:	current state		(use self.get_curr_state(obs))
					possible actions	(["move 1", "move 0"])
			return:	action				("move 1" or "move 0")
		"""

	# new state
	# NOTE: maybe make it so agent always moves in a new state?
		if curr_state not in self.q_table:
			self.q_table[curr_state]={}
			for action in possible_actions:
				self.q_table[curr_state][action]=0
	# chance to choose a random action
	# NOTE: maybe random chance to move instead?
		if random.random() < self.epsilon:
			random_action_i=random.randint(0,len(possible_actions)-1)
			return possible_actions[random_action_i]
		else:
			action_i = numpy.argmax([self.q_table[curr_state][action] for action in possible_actions])
			return possible_actions[action_i]

	# get the best action based on the greatest q-val(s) up

	def get_curr_state(self, obs):
		"""	get a simplified, integer-based version of the environment
			args:	world observations	(use world.get_observations(...))
			return:	state 				((curr z, arrow₁ x, arrow₂ x, ...))
		"""
		state=[]
	# get current z-position rounded down
		current_position=world.get_curr_pos(obs)
		state.append(int(current_position['z'])-1)

	# get arrow x-positions, ordered by increasing z-positions
		current_arrow_positions=world.get_arrow_pos(obs)
		for x,y,z in self.dispenser_pos:
			if int(z) in current_arrow_positions:
				state.append(current_arrow_positions[int(z)])

			else:
				state.append(None)

	# (curr_pos[z], arrow_pos[z₁] = x₁, arrow_pos[z₂] = x₂, ...)
		return tuple(state)

	def run(self):
		"""	observations → state → act → reward ↩, and update q table
			return:	total reward		(cumulative int reward value of the run)
					success_flag flag		(True / False)
		"""
		# history of states/actions/rewards
		S, A, R = deque(), deque(), deque()

		# either you move or you don't
		possible_actions = ['move 1', 'move 0']

		# returns total reward and success_flag flag
		totalR, success_flag = 0, None

		# initialize terminating state
		terminate_state = 'ENDDING'
		self.q_table[terminate_state] = {}
		for action in possible_actions:
			self.q_table[terminate_state][action] = 0

		# run until damaged
		running_flag = True
		while running_flag:

			# get initial state/action/reward
			obs = world.get_observations(self.agent_host)
			s0 = self.get_curr_state(obs)
			a0 = self.get_action(s0, possible_actions)
			r0 = 0
			S.append(s0)
			A.append(a0)
			R.append(r0)

			# continuously get observations
			T = sys.maxsize  # T是预设什么时候是episode的终点，初始化给定一个maxsize
			for t in range(sys.maxsize):  # t 在一个episode中采样到第几个状态
				obs = world.get_observations(self.agent_host)
				# death or out of bounds glitching ends the run
				curr_pos = world.get_curr_pos(obs)
				self.life = world.get_curr_life(obs)
				if curr_pos['z'] - self.start_pos['z'] > 10 or self.life == 0:
					# episode finish: end state and get final reward
					success_flag = False
					return totalR, success_flag

				if t < T:

					# episode running_flag: act and get state/action/reward
					if running_flag == False:
						# 说明当前episode已经采样结束了
						T = t + 1  # 当前episode的最大采样步数T就是t+1
						S.append(terminate_state)  # 说明当前的状态就是S_T+1，也就是terminate_s

					else:
						# act (move or wait)
						self.agent_host.sendCommand(A[-1])
						time.sleep(self.sleep_time)

						# get reward and check if episode is finished
						r, success_flag = self.get_reward(obs, A[-1])
						R.append(r)
						totalR += r
						if success_flag != None:
							if success_flag == True:
								print("You Find the Precious!")
							running_flag = False
							continue  # 到了终点或者死掉了就不再加入当前状态，而是加入end状态

						# get state/action
						s = self.get_curr_state(obs)
						action = self.get_action(s, possible_actions)
						S.append(s)
						A.append(action)
						print('s:', s, 'action:', action)
				# end of episode: update q table
				tau = t + 1 - self.n
				if S[0] == terminate_state:  # 排除只剩一个State的情况
					break
				if tau >= 0:
					self.update_q_table(tau, S, A, R, T)

		return totalR, success_flag



	def hard_coded_run(self, wait_block, arrow_x_pos):
		"""	guarantee move when agent on wait_block and arrow on arrow_x_pos
			return:	success flag		(True / False)
		"""
		pass
	

			# get initial state/action/reward
			
			# death or out of bounds glitching ends the run
			
			# act
			
			# win/lose condition

