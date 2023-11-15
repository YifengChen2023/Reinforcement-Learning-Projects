import configparser
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter()

config = configparser.ConfigParser()
config.read('config.ini')
epsilon = float(config.get('DEFAULT', 'EPSILON'))
epsilon_min = float(config.get('DEFAULT', 'EPSILON_MIN'))
epsilon_decay = float(config.get('DEFAULT', 'EPSILON_DECAY'))
batch_size = int(config.get('DEFAULT', 'BATCH_SIZE'))
gamma = float(config.get('DEFAULT', 'GAMMA'))
memory_capacity = int(config.get('DEFAULT', 'MEMORY_CAPACITY'))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#adevice= torch.device("cpu")

class duelingDQN(nn.Module):
    def __init__(self,input_size,output_size):
        super(duelingDQN, self).__init__()
        self.f1=nn.Linear(input_size,24)
        self.f1.weight.data.normal_(0,0.1)
        self.f2=nn.Linear(24,32)
        self.f2.weight.data.normal_(0,0.1)
        self.f3=nn.Linear(32,16)
        self.f3.weight.data.normal_(0,0.1)

        self.fc_A=nn.Linear(16,output_size)
        self.fc_V=nn.Linear(16,1)

    def forward(self,x):
        x=self.f1(x)
        x=F.leaky_relu(x)
        x=self.f2(x)
        x=F.leaky_relu(x)
        x=self.f3(x)
        x=F.leaky_relu(x)

        adv=self.fc_A(x)
        val=self.fc_V(x)
        adv_avg=torch.mean(adv,dim=1,keepdim=True)
        output=val+adv-adv_avg
        return output

    def sample_noise(self):
        pass #ToDO: implement noise sampling if time permits

class DQNAgent(object):
    def __init__(self,state_size,action_size) -> None:
        super(DQNAgent,self).__init__()
        self.state_size=state_size
        self.action_size=action_size
        self.device = device
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min
        self.batch_size=batch_size
        self.gamma=gamma

        self.learning_rate=0.1
        self.memory_capacity=memory_capacity
        self.mem_cnt=0
        self.memory= np.zeros((self.memory_capacity, self.state_size * 2 + 2+1))# state(7),net(s)=a,reward,nxt_state(7),done
        self.model=duelingDQN(self.state_size,self.action_size).to(device)
        self.target_model=duelingDQN(self.state_size,self.action_size).to(device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.loss=nn.MSELoss()
        self.step=0
        self.replace_step=int(config.get('DEFAULT', 'REPLACE_STEP'))

    def act(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        if np.random.uniform(low=0,high=1)<self.epsilon:
            action=np.random.randint(0,self.action_size)

        else:
            action=self.model(state).argmax().item()

        return action

    def max_q_value(self,state):
        #state=torch.unsqueeze(torch.FloatTensor(state),0).to(self.device)
        state=torch.tensor([state],dtype=torch.float).to(self.device)
        #return self.model(state).max().item()
        pass


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def remember(self,state,action,reward,nxt_state,done):
        index =len(self.memory)%self.memory_capacity
        member=np.append(state,action)
        member=np.append(member,reward)
        member=np.append(member,nxt_state)
        member=np.append(member,done)
        self.memory[index]=member
        if self.mem_cnt<self.memory_capacity:
            self.mem_cnt+=1

    def replay(self,batch_size):

        minibatch = np.random.choice(self.mem_cnt,batch_size)
        state_batch = torch.FloatTensor(self.memory[minibatch, 0:7]).to(self.device)
        action_batch =  torch.LongTensor(self.memory[minibatch, 7]).view(-1,1).to(self.device)
        reward_batch =  torch.FloatTensor(self.memory[minibatch, 8]).view(-1,1).to(self.device)
        next_state_batch = torch.FloatTensor(self.memory[minibatch, 9:16]).to(self.device)
        done_batch = torch.FloatTensor(self.memory[minibatch, 16]).view(-1,1).to(self.device)

        #torch.cuda.empty_cache()
        Q_eval=self.model(state_batch).gather(1,action_batch)
        with torch.no_grad():
            max_action=self.model(next_state_batch).max(1)[1].view(-1, 1)#Double DQN
            max_Q_value= self.target_model(next_state_batch).gather(1,max_action)
            Q_target=reward_batch+self.gamma*max_Q_value*(1-done_batch)#Dueling DQN

        loss=self.loss(Q_eval,Q_target)
        print("Loss: ",loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #if self.step%self.replace_step==0:
            #self.update_target_model()

        return loss.item()


    def save(self,nn_save):
        torch.save(self.model.state_dict(), nn_save)
        print("Model saved")


    def load(self,saved_model):
        self.model.load_state_dict(torch.load(saved_model))
        self.update_target_model()
        print("Model loaded")















