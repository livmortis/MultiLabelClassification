
import torch
import data as datapy
import os
import numpy as np
import pandas as pd
import torch.utils.data as data

rootdict = "../MutitagData"
resultdict = "/submit"
resultCsv = "/result.csv"
modelsaveddict = "/modelSaved"
modelpkl = "/mySavedModel.pkl"
testDatasaveddict = "/testdataSaved"
testnpy = "/testDataSaved.npy"
TEST = "test"
IMAGE_SIZE = 299
batch_size = 128  #需要8的倍数

NEED_READ_ORIGION_PICTURE = False    #是否需要从头读取图片文件
testList = []
testPicNameList = []
predicts = []

def load_data():
    global testList

    global testLoader

    load_model()

def load_model():
    if os.path.exists(rootdict + modelsaveddict):
        mymodel = torch.load(rootdict + modelsaveddict + modelpkl)
        predict_with_model(mymodel)



def predict_with_model(mymodel):
    global testList
    global predicts
    global testLoader



    mymodel.eval()

    for index , (pre, name) in enumerate(testLoader):
        # pre = torch.from_numpy(pre)
        pre = pre.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        pre = pre.type(torch.FloatTensor) #一定要转换浮点数。

        # pre = pre.unsqueeze(dim=0)
        if torch.cuda.is_available():
            mymodel.cuda()
            pre = pre.cuda()
        # torch.unsqueeze(sample,dim=1)
        # print("the one sample shape before train is: " + str(sample.shape))
        # predict = mymodel(sample, type=TEST)
        predict = mymodel(pre)
        predicts = []
        for pre in predict:
            pre = torch.nn.Sigmoid(pre)
            predicts.append(pre)
        predicts = np.array(predicts)
        print("this test prediction: "+ str(predict))

        # print("刚出炉的predict到底是什么shape：" + str(predict.shape))


        # 预测值转换为文字标签
        preLabelLabel = datapy.sigmoid2strings(predict)
        print("this test prediction after to string: "+ str(preLabelLabel))

        predicts.extend(preLabelLabel)
        testPicNameList.extend(name)


    # print("some preLabels: "+str(predicts[:10]))
    mkfileForSubmit()

def mkfileForSubmit():
    global testPicNameList
    testPicNameList = np.array(testPicNameList)
    df = pd.DataFrame({"img_path":testPicNameList, "tags":predicts})

    # 列表变为字符串
    for i in range(len(testPicNameList)):
        df["tags"].loc[i] = ",".join(str(label.strip("\n")) for label in df["tags"].loc[i])

    if not os.path.exists(rootdict + resultdict):
        os.mkdir(rootdict + resultdict)
    df.to_csv(rootdict + resultdict + resultCsv , index=False, encoding="utf-8" )

if __name__ == "__main__":
    # global testList
    testData = datapy.XzyTestData()
    testLoader = data.DataLoader(testData, batch_size=batch_size, shuffle=False)

    if os.path.exists(rootdict + testDatasaveddict + testnpy)\
            and not NEED_READ_ORIGION_PICTURE:  #数据文件npy已经处理保存与本地
        testList = np.load(rootdict + testDatasaveddict + testnpy)
        load_model()
    else:
        load_data()
