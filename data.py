import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from torch.autograd.variable import Variable
import torch
import torch.utils.data as data
import os
from tqdm import tqdm

# rootdict = "/Users/xzy/Documents/coder/ML/game/tinymindMultitag/MutitagData"
rootdict = "../MutitagData"
tainDataDict = "/train"
# testDataDict = "/valid"
testDataDict = "/test"     #决赛，更改了测试集
datasaveddict = "/dataSaved"
testDatasaveddict = "/testdataSaved"
xnpy = "/xSaved.npy"
ynpy = "/ySaved.npy"
testnpy = "/testDataSaved.npy"
testNamenpy = "/testNameSaved.npy"
bacth_size = 100
NEED_RENEW_DATA = False  #是否要重头开始读取图片文件
NEED_RENEW_TEST_DATA = False  #是否要重头开始读取测试图片文件
TRAIN = "train"
EVAL = "eval"

TRAIN_LOAD_IMG_NUM = 1000
TEST_LOAD_IMG_NUM = 50

# IMAGE_SIZE = 224
# IMAGE_SIZE = 500
IMAGE_SIZE = 299

# def main ():
    # checkCsv()
    # checkNpz()
    # onehots = np.random.randint(0,2,6941)
    # print(onehots)
    # onehot2strings(onehots)

    # loadTrainPic()


####################抄
# def hash_tag(filepath):
#     fo = open(filepath, "r", encoding='utf-8')
#     hash_tag = {}
#     i = 0
#     for line in fo.readlines():  # 依次读取每行
#         line = line.strip()  # 去掉每行头尾空白
#         hash_tag[i] = line
#         i += 1
#     return hash_tag
#
#
# def load_ytrain(filepath):
#     y_train = np.load(filepath)
#     y_train = y_train['tag_train']
#
#     return y_train
#
#
# def arr2tag(arr):
#     tags = []
#     for i in range(arr.shape[0]):
#         tag = []
#         index = np.where(arr[i] > 0.5)
#         index = index[0].tolist()
#         tag = [hash_tag[j] for j in index]
#
#         tags.append(tag)
#     return tags



####################抄











def label2dic():
    txtfile = open(rootdict+"/valid_tags.txt", "r", encoding="utf-8")
    txtdic = {}
    i = 0
    for line in txtfile.readlines():
        txtdic[i] = line
        i+=1
    # print(txtdic[3])
    return txtdic

#将输出预测值转化为文字，上传比赛时用
def onehot2strings(onehots):
    prectLabels = []
    onehotsLen = len(onehots)
    txtdic = label2dic()
    if(len(txtdic) != onehotsLen):
        # print("sth wrong")
        # print("txtdic length is: "+str(len(txtdic)))
        # print("onehotss length is: "+str(onehotsLen))
        return
    for i in range(onehotsLen):
        if onehots[i] == 1:                   #换更高级方法
            prectLabels.append(txtdic[i])
        # indexes = np.where(onehots[i] == 1) #indexes是元祖
        # indexes = indexes[0].tolist()
        # prectLabels.append( [txtdic[j] for j in indexes] )

    return prectLabels
    # print(str(prectLabels))
    # print(len(prectLabels))



#将输出概率预测值转化为文字，上传比赛时用
def sigmoid2strings(sigmoidPre):
    prectLabels = []
    # sigmoidsLen = len(sigmoidPre)
    sigmoidsLen = sigmoidPre.shape[0]   # 因为预测值是列向量
    txtdic = label2dic()
    if(len(txtdic) != sigmoidsLen):
        return
    for i in range(sigmoidsLen):
        if sigmoidPre[0][i] > 0.1:
            prectLabels.append(txtdic[i])
    return prectLabels


#csv文件用来拿到图片名，通过图片名再加载得到data
def checkCsv():
    print("begin read train csv file")
    dataFrame = pd.read_csv(rootdict+"/visual_china_train.csv")
    # print(dataFrame.head(5))
    # print(dataFrame.shape)

    labelSets = []
    for i in range(1,dataFrame.shape[0]):
        for j in dataFrame['tags'].iloc[i].split(','):
            labelSets.append(j)

    labelSets = set(labelSets)
    # print(len(labelSets))

    imagePath = dataFrame['img_path']
    # print("imagePath's type is: ", type(imagePath))
    imagePath = list(imagePath)
    # print(type(imagePath))
    # for i in range(5):
    #     print(imagePath[i])
    return imagePath


#npz文件用来拿到label
def checkNpz():
    npzFile = np.load(rootdict+"/tag_train.npz")
    # print("npz's tag is: "+str(npzFile.files))
    # print("npz file is: "+str(npzFile['tag_train']))
    # print("npz file shape is: "+str(npzFile['tag_train'].shape))
    # print("npz numpy : "+str(npzFile['tag_train'][:50]))

    return npzFile["tag_train"]


def loadTestPic():
    print("begin read test data")
    testPicList = os.listdir(rootdict + testDataDict + "/")
    # testPicList = testPicList[:TEST_LOAD_IMG_NUM] # 控制载入的测试图片数量
    testImages = np.zeros([len(testPicList), IMAGE_SIZE, IMAGE_SIZE, 3])
    i = 0
    for testPic in tqdm(testPicList):
        testImage = Image.open(rootdict + testDataDict + "/" + testPic)
        if(testImage.mode != "RGB"):
            testImage = testImage.convert("RGB")    #色彩空间
        testImage = testImage.resize((IMAGE_SIZE, IMAGE_SIZE))    #尺寸
        testImage = np.asarray(testImage)           #变数组1
    # testImages.append(testImage)
    # testImages = np.array(testImages)  #变数组2。这样变的数组，其每个元素（图片）没有第四维，没有batch。而搭建的
    # CNN网络需要每个数据有四维.（追评：该方法无法增加维度！！！！）
        testImages[i, :, :, :] = testImage
        i+=1

    if not os.path.exists(rootdict + testDatasaveddict):
        os.mkdir(rootdict + testDatasaveddict)

    np.save(rootdict + testDatasaveddict + testnpy, testImages )
    np.save(rootdict + testDatasaveddict + testNamenpy, testPicList)

    return testImages, testPicList


def loadTrainPic():
    imagePaths = checkCsv()
    # imagePaths = imagePaths[:TRAIN_LOAD_IMG_NUM]   # 控制载入的训练图片数量
    y = checkNpz()
    # y = y[:TRAIN_LOAD_IMG_NUM]                     # 控制载入的训练图片数量

    imageDatas = np.zeros((len(imagePaths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


    i = 0
    # for imageP in imagePaths[:bacth_size]:
    for imageP in tqdm(imagePaths):
        image = Image.open(rootdict + tainDataDict + "/" + imageP)
        # print('Image.open() return type is: ' + str(type(image)))

        if(image.mode != "RGB"):
            image = image.convert("RGB")    #色彩空间
        image = image.resize((IMAGE_SIZE,IMAGE_SIZE))     #尺寸
        image = np.asarray(image)           #变数组1
        imageDatas[i,:,:,:] = image         #加四维(没加成功)
        i+=1
    # print('imageDatas size is: '+str(len(imageDatas)))
    # 展示前9张图片
    # arg,axes = plt.subplots(3,3, figsize = (60,60))
    # p=0
    # q=0
    # for image in imageDatas[:9]:
    #     axes[p//3,q%3].imshow(image)
    #     p+=1
    #     q+=1
    # plt.show()


    x = imageDatas
    print("look x : ", x.shape)
    print("look y : ", y.shape)

    isExsit = os.path.exists(rootdict+datasaveddict)
    if not isExsit:
        os.mkdir(rootdict+datasaveddict)
        print("create a 'dataSaved' file ")

    np.save(rootdict+datasaveddict+xnpy, x)
    np.save(rootdict+datasaveddict+ynpy, y)



    return x, y


def processDataManual():    #with out 'data.Dataset'
    x,y = loadTrainPic()

    x_train,x_val,y_train,y_val = tts(x,y[:bacth_size],train_size=0.8)
    # print("x_train shape is"+str(x_train.shape))
    # print("x_val shape is"+str(x_val.shape))
    # print("y_train shape is"+str(y_train.shape))
    # print("y_val shape is"+str(y_val.shape))



    # tuple换成tensor

    # x_train = Variable(torch.from_numpy(x_train))
    # y_train = Variable(torch.from_numpy(y_train))
    # x_val = Variable(torch.from_numpy(x_val))
    # y_val = Variable(torch.from_numpy(y_val))
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)
    # pytorch 0.4.0 已经不需要在转为Variable变量才能输入网络


    x_train = x_train.view(80, 3, IMAGE_SIZE, IMAGE_SIZE)
    x_val = x_val.view(20, 3, IMAGE_SIZE, IMAGE_SIZE)
    # pytorch卷积网络要求通道数在宽高之前,而图片本身通道数在宽高之后。

    x_train = x_train.type(torch.FloatTensor)
    y_train = y_train.type(torch.FloatTensor)   #CrossEntropyLoss不需要，MultiLabelSoftMarginLoss需要。



class XzyData(data.Dataset):
    def __init__(self, type):

        # x, y = loadTrainPic() //改为存储数据到文件
        if os.path.exists(rootdict+datasaveddict+xnpy) \
                and not NEED_RENEW_DATA:
            x = np.load(rootdict+datasaveddict+xnpy)
            y = np.load(rootdict+datasaveddict+ynpy)
        else:
            x, y = loadTrainPic()

        # 划分验证集
        #这里没有加shuffle，有必要吗？
        index = np.arange(0, len(x), 1, dtype=np.int)
        if type == TRAIN:
            index = index[int(len(x)*0.1) : ]
        elif type == EVAL:
            index = index[ : int(len(x)*0.1)]


        self.x = x[index]
        self.y = y[index]


    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return torch.from_numpy(self.x[item]), torch.from_numpy(self.y[item])



class XzyTestData(data.Dataset):
    def __init__(self):

        if os.path.exists(rootdict+testDatasaveddict+testnpy) \
                and not NEED_RENEW_TEST_DATA:
            x = np.load(rootdict+testDatasaveddict+testnpy)
            y = np.load(rootdict+testDatasaveddict+testNamenpy)
        else:
            x, y = loadTestPic()

        self.x = x
        self.y = y


    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return torch.from_numpy(self.x[item]), self.y[item]




if(__name__ == '__main__'):
    checkNpz()
