
import keras.backend as K
import data as datapy
import model
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import torch
import torch.utils.data as data
import os
import numpy as np

# epoch_size = 50
epoch_size = 50
batch_size = 128  #需要8的倍数
NEED_RESTART_TRAIN = True  #是否不读现有模型，重头开始训练
rootdict = "../MutitagData"
modelsaveddict = "/modelSaved"
modelpkl = "/mySavedModel.pkl"
TRAIN = "train"
EVAL = "eval"
learning_rate = 0.001
TEST = "test"

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)


def fmeasureByTorch(y_true, y_pred):
    if torch.cuda.is_available():
        y_true = y_true.cpu.data.numpy()
        y_pred = y_pred.cpu.data.numpy()
    else:
        y_true = y_true.data.numpy()
        y_pred = y_pred.data.numpy()

    numerator = y_true * y_pred
    numerator = 2 * np.sum(numerator, axis=1)
    denominator = np.sum(y_true, axis=1) + np.sum(y_pred, axis=1)
    fraction = numerator / denominator
    fraction_sum = np.sum(fraction) / len(fraction)

    return fraction_sum




def begin_train_new():

    # myModel = model.mtModel()   #移到外边，可在训练前选择是否读取已有模型。
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    myOptim = torch.optim.Adam(myModel.parameters(), lr=learning_rate)

    myModel.train()     #模型的训练模式
    print("begin train ... ")

    for index, (x, y) in enumerate(loaderTrain, 0):
        print("batch num: "+str(index))
        x = x.view(-1, 3, 224, 224)    #即使使用Dataset，也需要调整NHWC为NCHW。
        x = x.type(torch.FloatTensor)   #即使使用Dataset，也需要调整类型为float。
        y = y.type(torch.FloatTensor)   #即使使用Dataset，也需要调整类型为float。

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            myModel.cuda()

        myOptim.zero_grad()
        predict = myModel(x, type=None)
        # myLoss = criterion(y, predict)
        myLoss = criterion(predict, y)
        myLoss.backward()
        myOptim.step()


        print("loss is : "+str(myLoss.data))
        # print("train fmeasure is: " + str(fmeasureByTorch(y, predict)))
            #train的过程就不fmeasure评价了，因为fmeasure要求预测值是sigmoid后的，train过程没有。验证时再评价。

    torch.save(myModel, rootdict + modelsaveddict + modelpkl)  # 保存模型(每一个epoch保存一次)

def begin_eval():


    # myModel = model.mtModel()   #移到外边，可在训练前选择是否读取已有模型。
    # criterion = torch.nn.MultiLabelSoftMarginLoss()
    # myOptim = torch.optim.Adam(myModel.parameters(), lr=0.001)

    myModel.eval()     #模型的验证模式
    print("begin eval ... ")

    for index, (x, y) in enumerate(loaderEval, 0):
        print("batch num: "+str(index))
        x = x.view(-1, 3, 224, 224)    #即使使用Dataset，也需要调整NHWC为NCHW。
        x = x.type(torch.FloatTensor)   #即使使用Dataset，也需要调整类型为float。
        y = y.type(torch.FloatTensor)   #即使使用Dataset，也需要调整类型为float。

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            myModel.cuda()

        # myOptim.zero_grad()
        predict = myModel(x, type=TEST)
        # myLoss = criterion(y, predict)
        # myLoss = criterion(predict, y)
        # myLoss.backward()
        # myOptim.step()

        #TODO 引入验证集评判标准
        # print("loss is : "+str(myLoss.data))
        print("evla fmeasure is : "+str(fmeasureByTorch(y, predict)))

    # torch.save(myModel, rootdict + modelsaveddict + modelpkl)  # 保存模型


if __name__ == '__main__':
    # y_true = np.array([[1,0,1,1,1,0,1,0,1,0,1,0,1,1,0,0,0,0,1,0,1,0,0,1],
    #                    [1,1,0,1,1,0,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,1,0,1]])
    # y_pred = np.random.rand(2,24)
    # fmeasureByTorch(y_true, y_pred)
    isExist = os.path.exists(rootdict+modelsaveddict)
    if not isExist:
        os.mkdir(rootdict+modelsaveddict)
        print("create a 'modelSaved' file ")

    modelExist = os.path.exists(rootdict+modelsaveddict+modelpkl)

    if modelExist and not NEED_RESTART_TRAIN:
        myModel = torch.load(rootdict+modelsaveddict+modelpkl)   #读取模型
    else:
        myModel = model.mtModel()



    xzyDataTrain = datapy.XzyData(TRAIN)
    loaderTrain = torch.utils.data.DataLoader(xzyDataTrain, batch_size=batch_size, shuffle=True)
    xzyDataEval = datapy.XzyData(EVAL)
    loaderEval = torch.utils.data.DataLoader(xzyDataEval, batch_size=batch_size, shuffle=True)


    # begin_train_old()
    for i in range(epoch_size):
        print("epoch num: " + str(i))

        begin_train_new()
        begin_eval()

















def begin_train_old():
    print("begin train ... ")


    # forward
    x_train, x_val, y_train, y_val = data.loadPic()
    mymodel = model.mtModel()
    # mymodel.train()
    # print("8888",type(x_train))
    # print("8888training",x_train.shape)
    # print("8888type", type(x_train[0]))


    # y_train = torch.max(y_train, 1)[1] #由于多标签分类，这里不能进行反one-hot


    myloss = nn.CrossEntropyLoss()
    # myloss = nn.MultiLabelSoftMarginLoss()  #由于多标签分类


    optm = optim.SGD(mymodel.parameters(), lr=0.01)

    for i in range(epoch_size):

        outputs = mymodel(x_train)
        # print(mymodel)
        # print(outputs)


        # # loss
        # myloss = nn.CrossEntropyLoss()
        # print("before loss, outputs : ", outputs.shape)
        # print("before loss, label : ", y_train.shape)
                #输出和标签都是二维，[bachsize, 类别数]，交叉熵必须要求一维数据，因此用torch.max()将one-hot返回（由于是多分类单标签）
        # # outputs = torch.max(outputs,1)[1]
        # y_train = torch.max(y_train,1)[1]
        # print("before loss, outputs,after max() : ", outputs.shape)
        # print("before loss, label,after max()  : ", y_train.shape)
        # print("before loss, outputs,after max() : ", type(outputs))
        # print("before loss, label,after max()  : ", type(y_train))
        # # outputs = outputs.type(torch.FloatTensor)
        # # y_train = y_train.type(torch.FloatTensor)
        # print("before loss, outputs,after type() : ", type(outputs))
        # print("before loss, label,after type()  : ", type(y_train))

        # print("标签是： ",y_train)
        loss = myloss(outputs, y_train)


        # backward
        loss.backward()

        # optimizer
        optm.step()

        # prediction_labels = data.onehot2strings(outputs)
        # print("output before backonehot: " + str(outputs))
        # print("output: " + str(prediction_labels))


        # #暂时以单标签分类来对待预测值，查看结果
        # batchLabel = torch.max(outputs, 1)[1]
        # print(str(batchLabel))
        # batchLabelNumpy = batchLabel.numpy()
        # for i in range(len(batchLabelNumpy)):
        #     prediction = data.label2dic()[batchLabelNumpy[i]]
        #     print(str(prediction))

        # 借用代码 - 查全率
        fm = fmeasure(tf.convert_to_tensor(y_train.numpy()), tf.convert_to_tensor(outputs.data.numpy()))
        with tf.Session() as sess:
            print(sess.run(fm))

        # 展示结果
        outputsNp = outputs.data.numpy()
        # results = []
        # for i in range(len(outputsNp)):
        #     result = outputsNp[i][outputsNp[i]>0.8]
        #     results.append(result)
        # print(results)

        # 展示汉字结果(亲测可用)
        # outputsNp[outputsNp>0.7] = 1
        # outputsNp[outputsNp<=0.7] = 0
        # # print(outputsNp)
        #
        # for i in range(len(outputsNp)):
        #     print(data.onehot2strings(outputsNp[i]))