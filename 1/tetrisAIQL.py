from malmo import MalmoPython
import os
import sys
import time
import random
from random import randrange as rand
from collections import deque
from tetris_game import *
import pickle
import copy
import numpy
from numpy import zeros



rewards_map = {'inc_height': -20, 'clear_line': 100, 'holes': -20, 'top_height':-100}

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Tetris!</Summary>
        </About>
        <ServerSection>
                    <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/>
                <DrawingDecorator>
                    <DrawLine x1="2" y1="56" z1="22" x2="2" y2="72" z2="22" type="obsidian"/>
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
        </ServerSection>
        <AgentSection mode="Creative">
            <Name>MalmoTutorialBot</Name>
            <AgentStart>
                <Placement x="2.5" y="73" z="22.8" yaw="180"/>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromFullStats/>
                <ContinuousMovementCommands turnSpeedDegs="180"/>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''
    
def magic(X):
    return ''.join(str(i) for i in X)

QLfilename = 'maltrisQLTable.save'
QLgraphfilename = 'maltrisQLGraphData.save'

class TetrisAI:
    def __init__(self, game, alpha=0.3, gamma=1, n=1):
        self.epsilon = 0.3
        self.q_table = {}
        self.n, self.alpha, self.gamma = n, alpha, gamma
        self.gamesplayed = 0
        self.listGameLvl = []
        self.listClears = []
        self.game = game

    def saveQtable(self):
        f = open(QLfilename, 'wb')
        pickle.dump(self.gamesplayed, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        f2 = open(QLgraphfilename, 'wb')
        pickle.dump(self.gamesplayed, f2, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.listGameLvl, f2, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.listClears, f2, protocol=pickle.HIGHEST_PROTOCOL)
        f2.close()

    def loadQtable(self):
        f = open(QLfilename, 'rb')
        self.gamesplayed = pickle.load(f,encoding='iso-8859-1')
        self.q_table = pickle.load(f,encoding='iso-8859-1')
        f.close()
        f2 = open(QLgraphfilename, 'rb')
        self.gamesplayed = pickle.load(f2,encoding='iso-8859-1')
        self.listGameLvl = pickle.load(f2,encoding='iso-8859-1')
        self.listClears = pickle.load(f2,encoding='iso-8859-1')
        f2.close()

    def run(self, agent_host):
        states, actions, rewards = deque(), deque(), deque()
        curr_reward = 0
        done_update = False
        game_overs = 0
        self.loadQtable() #uncomment to load Q-table values
        while not done_update:
            init_state = self.get_curr_state()
            possible_actions = self.get_possible_actions()
            next_action = self.choose_action(init_state, possible_actions)
            states.append(init_state)
            actions.append(self.normalize(self.pred_insta_drop2(next_action)))
            rewards.append(0)

            T = sys.maxsize
            for t in range(sys.maxsize):
#                 time.sleep(0.1)
                if t < T:
                    curr_reward = self.act(next_action)
                    rewards.append(curr_reward)
                    
                    ########################################
                    if self.game.gameover == True:
                        game_overs += 1
                        self.gamesplayed += 1
                        self.listGameLvl.append(self.game.level)
                        self.listClears.append(self.game.line_clears)
                        print("这把agent到了第:", self.game.level, "关")
                        print("这把agent消去了", self.game.line_clears, "行")
                        self.game.start_game()

                        if game_overs == 10: # 每十局游戏打印目前战况
                            print("到目前为止,agent训练局数：", self.gamesplayed,
                                  "平均关数为", numpy.mean(self.listGameLvl),
                                  "平均消去行数为", numpy.mean(self.listClears))
                            self.saveQtable()
                            game_overs = 0

                    curr_state = self.get_curr_state()
                    states.append(curr_state)
                    possible_actions = self.get_possible_actions()
                    next_action = self.choose_action(curr_state, possible_actions)
                    actions.append(self.normalize(self.pred_insta_drop2(next_action)))
                    ########################################

                tau = t - self.n + 1
                if tau >= 0:
                    self.update_q_table(tau, states, actions, rewards, T)

                if tau == T - 1:
                    while len(states) > 1:
                        tau = tau + 1
                        self.update_q_table(tau, states, actions, rewards, T)
                    done_update = True
                    break

                if t%5000 == 0:
                    self.saveQtable()  # uncomment to save Q-table values
                    print("------------------Saving Qtable------------")
#                     time.sleep(0.1)
                
    def act(self, action):
        for i in range(action[1]):
            self.game.rotate_piece()
        self.game.move(action[0] - self.game.piece_x)
        m_score =  self.score(self.pred_insta_drop2(action))
        # self.game.insta_drop_no_draw() # uncomment to no draw
        self.game.insta_drop()
        return m_score

    def get_curr_state(self):
        board = self.game.board[-2::-1]
        for i, row in enumerate(board):
            if all(j == 0 for j in row):
                if i < 2:
                    new_state = board[0:2]
                    new_state = [[1 if x!= 0 else x for x in row]for row in new_state]
                    return new_state
                else:
                    new_state = board[i-2:i]
                    new_state = [[1 if x!= 0 else x for x in row]for row in new_state]
                    return new_state
                          
    def normalize(self, state):
        board = state[-2::-1]
        for i, row in enumerate(board):
            if all(j == 0 for j in row):
                if i < 2:
                    new_state = board[0:2]
                    new_state = [[1 if x!= 0 else x for x in row]for row in new_state]
                    return new_state
                else:
                    new_state = board[i-2:i]
                    new_state = [[1 if x!= 0 else x for x in row]for row in new_state]
                    return new_state
    
    def get_possible_actions(self):
        actions = []
        action = (0,0)    
        ########################################
        for i in range(4):
            piece_x = 0
            piece_y = self.game.piece_y

            while piece_x <= self.game.rlim - len(self.game.piece[0]):
                if not check_collision(self.game.board,
                                       self.game.piece,
                                       (piece_x, piece_y)):
                    if action not in actions:
                        actions.append(action)
                piece_x += 1
                action = (action[0] + 1, action[1])
                piece_y = self.game.piece_y
            self.game.rotate_piece()
            action = (0, action[1] + 1)
        ########################################
        
        return actions

    def score(self, board):
        current_r = 0
        complete_lines = 0
        highest = 0
        holes = 0

        ########################################
        new_board = board[-2::-1]

        # 检查是否有完整的一行
        for row in new_board:
            if 0 not in row:
                complete_lines += 1
        current_r += complete_lines * rewards_map['clear_line']

        # 检查使得已有的块总增高了多少
        heights = zeros((len(new_board[0]),), dtype=numpy.int)
        for i, row in enumerate(new_board):
            for j, col in enumerate(row):
                if col != 0:
                    heights[j] = i + 1
        current_r += sum(heights) * rewards_map['inc_height']

        # 检查最高的点增高了多少
        for i, row in enumerate(new_board):
            if all(j == 0 for j in row):
                highest = i
        current_r += (highest - 2) * rewards_map['top_height']

        # 检查空洞
        for i, row in enumerate(new_board):
            if 0 in row:
                row_indexes = [i for i, j in enumerate(row) if j == 0]
                for index in row_indexes:
                    if index == 0:
                        if new_board[i][1]:
                            holes += 1
                    elif index == len(row) - 1:
                        if new_board[i][index-1]:
                            holes += 1
                    elif new_board[i][index-1] and new_board[i][index+1]:
                        holes += 1
            if all(j ==0 for j in row):
                break
                    
        current_r += holes * rewards_map['holes']
        ########################################
        
        return current_r
    
    def rotate_piece(self, piece, piece_x, piece_y, board):
        new_piece = rotate_clockwise(piece)
        if not check_collision(board, new_piece, (piece_x, piece_y)):
            return new_piece
        else:
            return piece
            
    def pred_insta_drop(self, piece, piece_x, piece_y):
        new_board = copy.deepcopy(board)

        while not check_collision(new_board,
                           piece,
                           (piece_x, piece_y+1)):
            piece_y += 1
            
        piece_y += 1
        new_board = join_matrixes(
            new_board,
            piece,
            (piece_x, piece_y))

        return new_board

    def pred_insta_drop2(self, action):
        new_board = copy.deepcopy(self.game.board)
        new_piece = self.game.piece
        new_piece_x = self.game.piece_x
        new_piece_y = self.game.piece_y
        
        for i in range(action[1]):
            new_piece = self.rotate_piece(new_piece, new_piece_x, new_piece_y, new_board)

        new_piece_x = action[0] - new_piece_x + 1
        if new_piece_x < 0:
            new_piece_x = 0
        if new_piece_x > cols - len(new_piece[0]):
            new_piece_x = cols - len(new_piece[0])

        while not check_collision(new_board,
                           new_piece,
                           (new_piece_x, new_piece_y+1)):
            new_piece_y += 1
            
        new_piece_y += 1
        new_board = join_matrixes(
            new_board,
            new_piece,
            (new_piece_x, new_piece_y))

        return new_board
    
    def choose_action(self, state, possible_actions):
        curr_state = magic(state)
        #######################################
        if curr_state not in self.q_table:
            self.q_table[curr_state] = {}
        for action in possible_actions:
            next_state = magic(self.normalize(self.pred_insta_drop2(action)))
            if next_state not in self.q_table[curr_state]:
                self.q_table[curr_state][next_state] = 0

        rnd = random.random()
        if rnd <self.epsilon:
            a = random.randint(0, len(possible_actions) - 1)
            best_action = possible_actions[a]
        else:
            best_actions = [possible_actions[0]]
            best_next_state = magic(self.normalize(self.pred_insta_drop2(best_actions[0])))
            qvals = self.q_table[curr_state]
            for action in possible_actions:
                next_state = magic(self.normalize(self.pred_insta_drop2(action)))
                if qvals[next_state] > qvals[best_next_state]:
                    best_actions = [action]
                    best_next_state = next_state
                elif qvals[next_state] == qvals[best_next_state]:
                    best_actions.append(action)
            a = random.randint(0, len(best_actions) - 1)
            best_action = best_actions[a]
        ########################################
        return best_action

    def update_q_table(self, tau, S, A, R, T):
        curr_s, curr_a, curr_r = magic(S.popleft()), A.popleft(), R.popleft()
        ########################################
        next_s = magic(curr_a)
        G = sum([self.gamma ** i * R[i] for i in range(len(S))])
        if tau + self.n < T:
            G += self.gamma ** self.n * self.q_table[magic(S[-1])][magic(A[-1])]

        old_q = self.q_table[curr_s][next_s]
        self.q_table[curr_s][next_s] = old_q + self.alpha * (G - old_q)
        ########################################
    
if __name__ == "__main__":
    random.seed(0)

    #Initialize agent_host
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print("ERROR:",e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    #Initialize Mission
    mission = MalmoPython.MissionSpec(missionXML, True)
    mission.allowAllChatCommands()
    mission.forceWorldReset()
    mission_record = MalmoPython.MissionRecordSpec()

    #Build Tetris Board
    left_x, right_x = -1, -1+cols+1
    bottom_y, top_y = 68, 68+rows+1
    z_pos = 3
    mission.drawLine( left_x, bottom_y, z_pos, left_x, top_y, z_pos, "obsidian" )
    mission.drawLine( right_x, bottom_y, z_pos, right_x, top_y, z_pos, "obsidian" )
    mission.drawLine( left_x, bottom_y, z_pos, right_x, bottom_y, z_pos, "obsidian" )
    mission.drawLine( left_x, top_y, z_pos, right_x, top_y, z_pos, "obsidian" )
    for i in range(-1,cols):
        mission.drawLine(i, bottom_y, z_pos-1, i, bottom_y+rows, z_pos-1, "quartz_block")
    
    #Attempt to start Mission
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( mission, mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    #Loop until mission starts
    print("Waiting for the mission to start")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()
    print("Mission running")

    numIter = 1
    n = 1
    my_game = TetrisGame(agent_host)
    my_AI = TetrisAI(my_game)
    print("n=", n)
    for n in range(numIter):
        my_AI.run(agent_host)
    print(my_AI.q_table)
