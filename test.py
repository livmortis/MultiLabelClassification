
import torch
import data as datapy
import os
import numpy as np
import pandas as pd

rootdict = "../MutitagData"
resultdict = "/submit"
resultCsv = "/result.csv"
modelsaveddict = "/modelSaved"
modelpkl = "/mySavedModel.pkl"
testDatasaveddict = "/testdataSaved"
testnpy = "/testDataSaved.npy"
TEST = "test"

NEED_READ_ORIGION_PICTURE = True    #是否需要从头读取图片文件
testList = []
testPicNameList = []
predicts = []

def load_data():
    global testList
    global testPicNameList
    testList, testPicNameList = datapy.loadTestPic()

    load_model()

def load_model():
    if os.path.exists(rootdict + modelsaveddict):
        mymodel = torch.load(rootdict + modelsaveddict + modelpkl)
        predict_with_model(mymodel)



def predict_with_model(mymodel):
    global testList
    global predicts


    mymodel.eval()
    testList = torch.from_numpy(testList)
    testList = testList.view(-1, 3, 224, 224)
    testList = testList.type(torch.FloatTensor)
    for sample in testList:
        sample = sample.unsqueeze(dim=0)
        if torch.cuda.is_available():
            mymodel.cuda()
            sample = sample.cuda()
        # torch.unsqueeze(sample,dim=1)
        # print("the one sample shape before train is: " + str(sample.shape))
        predict = mymodel(sample, type=TEST)
        # print("刚出炉的predict到底是什么shape：" + str(predict.shape))


        # 预测值转换为文字标签
        # predict = predict.data.numpy()
        preLabelLabel = datapy.sigmoid2strings(predict)

        predicts.append(preLabelLabel)
        # print("test data prediction is: "+str(predict))


    # print("some preLabels: "+str(predicts[:10]))
    mkfileForSubmit()

def mkfileForSubmit():
    global testPicNameList
    testPicNameList = np.array(testPicNameList)
    # result = []
    # for i in range(len(testPicNameList)):
        # oneItem = str(testPicNameList[i]) + ',' + str(predicts[i])
        # result.append(oneItem)
    df = pd.DataFrame({"img_path":testPicNameList, "tags":predicts})

    # 列表变为字符串
    for i in range(len(testPicNameList)):
        df["tags"].loc[i] = ",".join(str(label.strip("\n")) for label in df["tags"].loc[i])

    if not os.path.exists(rootdict + resultdict):
        os.mkdir(rootdict + resultdict)
    df.to_csv(rootdict + resultdict + resultCsv , index=False, encoding="utf-8" )

if __name__ == "__main__":
    # global testList
    if os.path.exists(rootdict + testDatasaveddict + testnpy)\
            and not NEED_READ_ORIGION_PICTURE:  #数据文件npy已经处理保存与本地
        testList = np.load(rootdict + testDatasaveddict + testnpy)
        load_model()
    else:
        load_data()
