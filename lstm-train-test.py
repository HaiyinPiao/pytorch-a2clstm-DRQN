import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

BATCH = 100;
TIMESTEP = 100;
HIDDEN = 40;
EPOCHS = 100;

class rnn_wrapper(nn.Module):

    def __init__(self):
        super(rnn_wrapper, self).__init__()
        self.lstm = nn.LSTM(input_size=HIDDEN, hidden_size=HIDDEN, batch_first = True)  # 输入维度是3, 输出维度也是3
        self.fc = nn.Linear(HIDDEN,1)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden



torch.manual_seed(1)

#---------------train procedure---------------------------
#---------------feed lstm by batch-------------------
hidden_init = (torch.randn(1, BATCH, HIDDEN), torch.randn(1, BATCH, HIDDEN))
hidden = hidden_init
# print('Hidden:',hidden)

rnn = rnn_wrapper()

loss_function=nn.MSELoss()
optimizer=optim.SGD(rnn.parameters(), lr=0.1)

# target construct
q = [(i*np.ones((BATCH,1))).tolist() for i in range(TIMESTEP)]
target = torch.FloatTensor(q).transpose(0,1)

# input construct
k = np.random.randn(TIMESTEP,HIDDEN).tolist()
inn = torch.FloatTensor([ k for i in range(BATCH)])

for epoch in range(EPOCHS):
    rnn.zero_grad()
    hidden = hidden_init

    out, hidden = rnn(inn, hidden)

    loss=loss_function(out, target)
    print('Loss:',loss.item())
    loss.backward()
    
    optimizer.step()

#---------------test procedure---------------------------
#---------------feed lstm per timestep-------------------
hx,cx = hidden
hx = hidden[0].select(1,0).unsqueeze(0);
cx = hidden[1].select(1,0).unsqueeze(0);
testin = inn.select(0,0);
print(testin)

for i in range(TIMESTEP):
    out, (hx,cx) = rnn(testin.select(0,i).view(1,1,-1), (hx,cx))
    print(out)