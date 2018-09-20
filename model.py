import torch
import torch.nn as nn
import data
from tensorboardX import SummaryWriter
TRAIN = "train"
EVAL = "eval"
TEST = "test"
class mtModel(nn.Module):
    def forward(self, input, type):

        x = self.conv1(input)
        # print("9999 shape after conv1", x[:3])
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        # print("9999 shape after conv2", x[:3])
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv3(x)
        # print("9999 shape after conv3", x[:3])
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        # print("9999 shape after conv4", x[:3])
        x = self.bn4(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv5(x)
        # print("9999 shape after conv5", x[:3])
        x = self.bn5(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv6(x)
        # print("9999 shape after conv6", x[:3])
        x = self.bn6(x)
        x = self.relu(x)

        x = self.averagepool(x)

        x = self.dropout(x)

        # print("9999 shape before linear", x[:3])
        #forward中进行平铺,两种写法等价
        # x = x.view(x.size(0), -1)
        x = x.view(x.size(0), 7*7*128)

        x = self.linear1(x)
        # print("model - linear输出： "+str(x[:3]))

        if(type == TEST):
            x = self.sigmoid(x)
            # print("model - sigmoid输出： " + str(x[:3]))



        return x


    def __init__(self):
        super(mtModel,self).__init__()
        self.maxpool = nn.MaxPool2d([2, 2], stride=None, padding=0)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 16, (3, 3), stride=2, padding=1)  #输出(112,112)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)  #输出(112,112)
        self.bn2 = self.bn1
        self.conv3 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=1)  #输出(56,56) pooling负责减半
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)  #输出(56,56)
        self.bn4 = self.bn3
        self.conv5 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)  #输出(28,28) pooling负责减半
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)  #输出(14,14) pooling负责减半
        self.bn6 = nn.BatchNorm2d(128)

        self.averagepool = nn.AvgPool2d(2)  #输出(7,7) pooling负责减半
        self.dropout = nn.Dropout2d(0.5)    #不影响宽高

        #这里在forward里进行平铺

        self.linear1 = nn.Linear(7*7*128, 6941)

        self.sigmoid = nn.Sigmoid()



def drawGraph():
    with SummaryWriter(comment='xzy') as sw:
        sw.add_graph(mtModel)
