import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym

# delete cart velocity state observation
# made a standard cartpole env as POMDP!!!!!!!!!!!!!!!!!!!
STATE_DIM = 4-1;
ACTION_DIM = 2;
STEP = 5000;
SAMPLE_NUMS = 1000;
TIMESTEP = 8;
A_HIDDEN = 40;
C_HIDDEN = 40;


# actor using a LSTM + fc network architecture to estimate hidden states.
class ActorNetwork(nn.Module):

    def __init__(self,in_size,hidden_size,out_size):
        super(ActorNetwork, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size,out_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        x = F.log_softmax(x,2)
        return x, hidden

# critic using a LSTM + fc network architecture to estimate hidden states.
class ValueNetwork(nn.Module):

    def __init__(self,in_size,hidden_size,out_size):
        super(ValueNetwork, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size,out_size)

    def forward(self,x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

def roll_out(actor_network,task,sample_nums,value_network,init_state):
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
    c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
    c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);

    for j in range(sample_nums):
        states.append(state)
        log_softmax_action, (a_hx,a_cx) = actor_network(Variable(torch.Tensor([state]).unsqueeze(0)), (a_hx,a_cx))
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM,p=softmax_action.cpu().data.numpy()[0][0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state,reward,done,_ = task.step(action)
        next_state = np.delete(next_state, 1)
        #fix_reward = -10 if done else 1
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state = task.reset()
            state = np.delete(state,1)
            a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
            a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
            c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
            c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);

            #print score while training
            print(j+1)
            break
    if not is_done:
        c_out, (c_hx,c_cx) = value_network(Variable(torch.Tensor([final_state])), (c_hx,c_cx))
        final_r = c_out.cpu().data.numpy()
    return states,actions,rewards,final_r,state

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def main():
    # init a task generator for data fetching
    task = gym.make("CartPole-v0")
    init_state = task.reset()
    init_state = np.delete(init_state,1)
    

    # init value network
    value_network = ValueNetwork(in_size=STATE_DIM, hidden_size=C_HIDDEN, out_size=1)
    value_network_optim = torch.optim.Adam(value_network.parameters(),lr=0.01)

    # init actor network
    actor_network = ActorNetwork(STATE_DIM, A_HIDDEN, ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.001)

    steps =[]
    task_episodes =[]
    test_results =[]

    for step in range(STEP):
        states,actions,rewards,final_r,current_state = roll_out(actor_network,task,SAMPLE_NUMS,value_network,init_state)
        init_state = current_state
        actions_var = Variable(torch.Tensor(actions).view(-1,ACTION_DIM)).unsqueeze(0)
        states_var = Variable(torch.Tensor(states).view(-1,STATE_DIM)).unsqueeze(0)

        # train actor network
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
        actor_network_optim.zero_grad()
        # print(states_var.unsqueeze(0).size())
        log_softmax_actions, (a_hx,a_cx) = actor_network(states_var, (a_hx,a_cx))
        vs, (c_hx,c_cx) = value_network(states_var, (c_hx,c_cx))
        vs.detach()
        # calculate qs
        qs = Variable(torch.Tensor(discount_reward(rewards,0.99,final_r)))
        qs = qs.view(1, -1, 1)

        advantages = qs - vs
        actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
        actor_network_optim.step()

        # train value network
        value_network_optim.zero_grad()
        target_values = qs
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0);
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0);
        values, (c_hx,c_cx) = value_network(states_var, (c_hx,c_cx))

        criterion = nn.MSELoss()
        value_network_loss = criterion(values,target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(),0.5)
        value_network_optim.step()

        # # Testing
        # if (step + 1) % 50== 0:
        #         result = 0
        #         test_task = gym.make("CartPole-v0")
        #         for test_epi in range(10):
        #             state = test_task.reset()
        #             for test_step in range(200):
        #                 softmax_action = torch.exp(actor_network(Variable(torch.Tensor([state]))))
        #                 #print(softmax_action.data)
        #                 action = np.argmax(softmax_action.data.numpy()[0])
        #                 next_state,reward,done,_ = test_task.step(action)
        #                 result += reward
        #                 state = next_state
        #                 if done:
        #                     break
        #         print("step:",step+1,"test result:",result/10.0)
        #         steps.append(step+1)
        #         test_results.append(result/10)

if __name__ == '__main__':
    main()
