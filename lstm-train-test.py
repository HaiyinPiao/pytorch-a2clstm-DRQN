import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class rnn_wrapper(nn.Module):

    def __init__(self):
        super(rnn_wrapper, self).__init__()
        self.lstm = nn.LSTM(3, 3)  # 输入维度是3, 输出维度也是3
        self.fc = nn.Linear(3,1)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.fc(x))
        return x, hidden



torch.manual_seed(1)

inputs = [torch.randn(1, 3) for _ in range(10)] # 构造一个长度为10的序列
print('Inputs:',inputs)

# 初始化隐藏状态
hidden_init = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
hidden = hidden_init
# print('Hidden:',hidden)

rnn = rnn_wrapper()

loss_function=nn.MSELoss()
optimizer=optim.SGD(rnn.parameters(), lr=0.1)

target = torch.FloatTensor([i for i in range(10)])
# print(target)

for epoch in range(500):
    rnn.zero_grad()
    hidden = hidden_init

    q = []
    for i in inputs:
        # print(hidden[0])
        out, hidden = rnn(i.view(1, 1, -1), hidden)
        q.append(out)

    m = torch.stack(q, dim=1)
    
    m=m.view(10,-1)
    target=target.view(10,-1)
    print(m)

    loss=loss_function(m, target)
    print('Loss:',loss.item())
    loss.backward()
    
    
    # print( lstm.all_weights )
    optimizer.step()