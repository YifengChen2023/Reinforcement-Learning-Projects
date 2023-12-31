# ==============================================================================
# MODULES
# ==============================================================================
from torch.utils.tensorboard import SummaryWriter

from dodger import Dodger			# dodger ai
try:
    from malmo import MalmoPython
except:
    import MalmoPython				# Malmo
import logging						# world debug prints
import time							# sleep for a few ticks every trial
import os							# os.system.clear()
import random						# random start delay
import world						# world manipulation/observation functions


writer = SummaryWriter('runs')

# ==============================================================================
# CONFIGURATIONS
# ==============================================================================
# log to file / terminal
# logging.basicConfig(filename="log.txt", level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

#logging.getLogger().disabled = True

# external file info
mission_xml = "mission.xml"


# ==============================================================================
# MAIN FUNCTIONS
# ==============================================================================
def regular_run():
	# get agent_host
	agent_host = MalmoPython.AgentHost()
	world.handle_args(agent_host)
	
	# update the hard-coded mission xml world path for the unique user file path
	mission_xml_path = world.update_mission_xml(mission_xml)
	print('mission_xml_path', mission_xml_path)
	# create ai
	dodger = Dodger(agent_host, num_arrow=3)	# 1a:1 6:3
	
	# record win rate
	num_wins = 0
	num_runs = 0
	win_rate = 0
	wr_table = [0] * 20
	wr_i = 0
	
	# regular run test info
	#moveable_blocks = [-311, -313, -315]
	#moveable_blocks = [-313, -314]
	moveable_blocks = list(range(-308, -318, -1))	# Agent自己可以走的格子
	possible_arrow_x_pos = list(range(455, 446, -1))	# 哪些格子上面可能会发射弩箭
	possible_arrow_x_pos.append(None)
	
	# continuously repeat trials
	num_reps = 500
	for rep in range(num_reps):
		os.system("clear")
		
		# print q_table and win-rate
		# dodger.print_1arrow_q_table(moveable_blocks, possible_arrow_x_pos)
		logging.info("win-rate in " + str(num_runs) + ": " + str(win_rate))
		print(wr_table)	# win_rate table

		# death or first episode: hard refresh the world
		if dodger.life == 0 or rep == 0:
			agent_host.sendCommand("quit")
			world.start_mission(agent_host, mission_xml_path)
			time.sleep(1)
			world.refresh(agent_host, dodger)
			print("dodger.dispenser_pos", dodger.dispenser_pos)
			time.sleep(random.uniform(2, 3.5))
		logging.info("starting mission " + str(num_runs))	

		# run dodger ai to get total reward and success flag
		total_reward, success,loss = dodger.run()
		writer.add_scalar('reward', total_reward/(rep+1), num_runs)
		writer.add_scalar('loss', loss/(rep+1), num_runs)
		#todo: add writer
		
		# update win rate
		num_runs += 1
		num_wins += 1 if success == True else 0
		win_rate = round(num_wins/num_runs, 1)
		wr_table[wr_i] = win_rate
		if rep > 0 and rep % 25 == 0:
			wr_i += 1
			
		
		# damaged or completion: soft refresh the world
		logging.info("end mission " + str(num_runs) + ": " + str(total_reward))
		world.soft_refresh(agent_host, dodger)
		time.sleep(random.uniform(0.1, 0.8))
			
	logging.info("reached max reps")
	print(wr_table)

def hard_coded_run():
	# get agent_host
	agent_host = MalmoPython.AgentHost()
	world.handle_args(agent_host)
	
	# update the hard-coded mission xml world path for the unique user file path
	mission_xml_path = world.update_mission_xml(mission_xml)
	
	# create ai
	dodger = Dodger(agent_host)
	
	# record win rate
	num_wins = 0
	num_runs = 0
	win_rate = 0
	
	# hard coded run info
	wait_block = -313
	possible_arrow_x_pos = [455, 454, 453, 452, 451, 450, 449, 448, 447, None]
	arrow_x_pos_i = 0
	arrow_x_pos = possible_arrow_x_pos[arrow_x_pos_i]
	hc_wr_table = [None for x in possible_arrow_x_pos]
	
	# continuously repeat trials
	num_reps = 30000
	for rep in range(num_reps):
		os.system("clear")
		
		# refresh win rate and move_when_arrow_on every 100 runs on a position
		dodger.print_hc_wr_table(wait_block, possible_arrow_x_pos, hc_wr_table)
		if rep % 100 == 0 and rep != 0:
			num_wins = num_runs = win_rate = 0
			arrow_x_pos_i += 1
			if arrow_x_pos_i == len(possible_arrow_x_pos):
				logging.info("DONE!")
				exit(0)
			arrow_x_pos = possible_arrow_x_pos[arrow_x_pos_i]
		
		# death or first episode: hard refresh the world
		if dodger.life == 0 or rep == 0:
			agent_host.sendCommand("quit")
			world.start_mission(agent_host, mission_xml_path)
			time.sleep(1)
			world.refresh(agent_host, dodger)
			time.sleep(random.uniform(2, 3.5))
		logging.info("starting mission " + str(num_runs))	
		
		# run dodger ai to get success flag
		success = dodger.hard_coded_run(wait_block, arrow_x_pos)
		
		# update win rate
		num_runs += 1
		num_wins += 1 if success == True else 0
		win_rate = round(num_wins/num_runs, 3)
		hc_wr_table[arrow_x_pos_i] = win_rate
		
		# damaged or completion: soft refresh the world
		logging.info("end mission " + str(num_runs))
		world.soft_refresh(agent_host, dodger)
		time.sleep(random.uniform(0.1, 0.8))
			
	logging.info("reached max reps")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
	
	# perform a regular 1-arrow reinforced learning run,
	# and prints the q-table for relevant blocks and the overall win-rate

	
	regular_run()
	
	# perform a 1-arrow run where the agent stops before the arrow row, 
	# and prints win-rate for moving on each arrow x pos

#	hard_coded_run()

	# here is the q-table for a purely random run (epsilon = 1.0)
	# it is largely a 50/50 with a little bias towards good habits,
	# and a few wild exceptions