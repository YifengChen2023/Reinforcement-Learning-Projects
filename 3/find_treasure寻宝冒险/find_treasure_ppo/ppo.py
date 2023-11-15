import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# import torch.distributions as distributions
class PPO(nn.Module):
    def __init__(self,  Pidim, Vdim, Dorger):
        super(PPO, self).__init__()
        # some basic params
        self.data = []
        self.fc1 = nn.Linear(4, 512)
        self.fc_policy = nn.Linear(512, Pidim)
        self.fc_value = nn.Linear(512, Vdim)

        self.gamma = Dorger.gamma
        self.lr = Dorger.alpha
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    def put_data(self, transition):
        self.data.append(transition)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    '''    
    def choose_action(self, x):
        prob = self.PolicyNet(x)
        m = distributions.Categorical(prob)
        action = m.sample()
        return action.item(), m.log_prob(action)
    '''
    def PolicyNet(self, x, softmax_dim=0):
        x1=self.fc1(x)
        x = F.leaky_relu(x1)
        x = self.fc_policy(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def ValueNet(self, x):
        x = F.leaky_relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc_value(x)
        return x


    
    def Batch(self):
        s_lst, a_lst, r_lst, next_s_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for t in self.data:
            s, a, r, next_s, prob_a, done = t
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            next_s_lst.append(next_s)
            prob_a_lst.append(prob_a)
            done = 0 if done else 1
            done_lst.append([done])
            p_list_numpy = []
            for x in prob_a_lst:
                p_list_numpy.append(x.detach().numpy())


        s, a, r, next_s, prob_a, done = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                        torch.tensor(r_lst, dtype=torch.float), torch.tensor(next_s_lst, dtype=torch.float), \
                                        torch.tensor(np.array(p_list_numpy)), torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s, a, r, next_s, prob_a, done

    def train_net(self):
        s, a, r, next_s, prob_a, done = self.Batch()

        td_target = r + self.gamma * self.ValueNet(next_s) * done
        delta = td_target - self.ValueNet(s)
        delta = delta.detach().numpy()
        advantage_lst = []
        advantage = 0.0
        #GAE
        for delta_t in delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta_t[0]
            advantage_lst.append(advantage)
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)
        PI = self.PolicyNet(s, softmax_dim=1)
        PI_action = PI.gather(1, a.reshape(-1, 1))
        ratio = torch.exp(torch.log(PI_action) - torch.log(prob_a.reshape(-1, 1)))
        a2=torch.squeeze(ratio,1)

        surr1 = torch.dot(a2, advantage)

        surr2 = torch.dot(torch.clamp(a2, min=1-self.eps_clip, max=1+self.eps_clip), advantage)
        loss = -torch.min(surr1, surr2).mean() + F.mse_loss(self.ValueNet(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss