import torch
import torch.nn as nn
import data
# from tensorboardX import SummaryWriter
import torchvision.models as tvModels

TRAIN = "train"
EVAL = "eval"
TEST = "test"
LABEL_NUMS = 6941
# NormalizationNum = 255.0
NormalizationNum = 300.0
class mtModel(nn.Module):
    def forward(self, input, type):
        input = input/NormalizationNum

        x = self.conv1(input)
        # print("9999 shape after conv1", x[:3])
        x = self.bn1(x)
        # print("9999 shape after bn1", x[:3])

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


        # 如果要插入预训练模型，插到这里。

        x = self.averagepool(x)

        x = self.dropout(x)

        # print("9999 shape before linear", x[:3])
        #forward中进行平铺,两种写法等价
        # x = x.view(x.size(0), -1)
        x = x.view(x.size(0), 7*7*128)        #  这里暗含了已经把图片大小写死了为224，否则不能确定是7。

        x = self.linear1(x)
        # print("model - linear输出： "+str(x[:3]))

        if(type == TEST):
            x = self.sigmoid(x)
            # print("model - sigmoid输出： " + str(x[:3]))



        return x


    def __init__(self):
        super(mtModel,self).__init__()
        self.maxpool = nn.MaxPool2d([2, 2], stride=None, padding=0) #stride默认值等于kernelsize，所以这是2.
        self.relu = nn.ReLU()
                                                                    #输入(224,224)
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

        self.linear1 = nn.Linear(7*7*128, LABEL_NUMS)       #  这里暗含了已经把图片大小写死了为224，否则不能确定是7。

        self.sigmoid = nn.Sigmoid()



# def drawGraph():
#     with SummaryWriter(comment='xzy') as sw:
#         sw.add_graph(mtModel)



def loadPretrainModel(type):
    # res18 = tvModels.resnet18(pretrained=True)
    # res34 = tvModels.resnet34(pretrained=True)
    # res50 = tvModels.resnet50(pretrained=True)
    # res152 = tvModels.resnet152(pretrained=True)
    # print('res18 is : '+ str(res18))
    # print('res34 is : '+ str(res34))
    # print('res50 is : '+ str(res50))
    # print('res152 is : '+ str(res152))
    inceptionV3 = tvModels.inception_v3(pretrained=True)
    # print('inceptionV3 is : '+ str(inceptionV3))

    # for param in inceptionV3.parameters():
    #     param.requires_grad = False

    # inceptionV3.fc = nn.Linear(2048, LABEL_NUMS)   #简单全连接层
    # inceptionV3.fc = nn.Sequential(                 #复杂全连接层
    #     nn.Dropout(0.3),
    #     nn.Linear(2048, 2048),
    #     nn.ReLU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(2048, LABEL_NUMS),
    # )
    if type == "test" :
        inceptionV3.fc = nn.Sequential(
            nn.Linear(2048, LABEL_NUMS),  # 简单全连接层
            nn.Sigmoid()
        )
    else:
        inceptionV3.fc = nn.Linear(2048, LABEL_NUMS)  # 简单全连接层

    return inceptionV3



'''
    ##########inceptionV3的倒数两层：############
    
  (Mixed_7c): InceptionE(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(2048, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_1): BasicConv2d(
      (conv): Conv2d(2048, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_2a): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3_2b): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_1): BasicConv2d(
      (conv): Conv2d(2048, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_2): BasicConv2d(
      (conv): Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3a): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch3x3dbl_3b): BasicConv2d(
      (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    (branch_pool): BasicConv2d(
      (conv): Conv2d(2048, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
  
'''






