import torch
from torch.autograd import Variable

#from memnet1 import MemNet
from memnet1 import MemNet
from visualize_net import  make_dot

x = Variable(torch.randn(1,1,31,31))#change 12 to the channel number of network input
model = MemNet(1,64,65,6)
y = model(x)
g = make_dot(y)
g.view()
