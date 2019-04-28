
# coding: utf-8

# # 数据预处理

# In[1]:


#载入必要库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
from glob import glob
# from tqdm import tqdm
import cv2
from PIL import Image


# In[2]:


#读取图片路径+标签文件
train_df = pd.read_csv('visual_china_train.csv')
for i in range(35000):
    train_df['img_path'].iloc[i] = 'train/' + train_df['img_path'].iloc[i].split('/')[-1]
img_paths = list(train_df['img_path'])

#制作对于标签对应的哈希表
def hash_tag(filepath):
    fo = open(filepath, "r",encoding='utf-8')
    hash_tag = {}
    i = 0
    for line in fo.readlines():                         
        line = line.strip()                               
        hash_tag[i] = line
        i += 1
    return hash_tag

def load_ytrain(filepath):  
    y_train = np.load(filepath)
    y_train = y_train['tag_train']
    
    return y_train

def arr2tag(arr):
    tags = []
    for i in range(arr.shape[0]):
        tag = []
        index = np.where(arr[i] > 0.5)  
        index = index[0].tolist()
        tag =  [hash_tag[j] for j in index]

        tags.append(tag)
    return tags

filepath = "valid_tags.txt"
hash_tag = hash_tag(filepath)

y_train = load_ytrain('tag_train.npz')


# In[3]:


img_paths[:10]


# In[4]:


hash_tag[0]


# In[5]:


y_train.shape


# In[6]:


#打乱并分割训练集和验证集
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

np.random.seed(2018)  
X_train_path,y_train = shuffle(np.array(img_paths),y_train)
X_train_path,X_val_path,y_train,y_val = train_test_split(X_train_path,y_train,test_size=0.2,random_state=0)
X_train_path,X_val_path = X_train_path.tolist(),X_val_path.tolist()


# In[7]:


X_train_path[:10],X_val_path[:10]


# 定义分批读取图片的生成器函数，不用将图片全部读入内存

# In[8]:


#读取图片函数
def get_image(img_paths, img_size):
    X = np.zeros((len(img_paths),img_size,img_size,3),dtype=np.uint8)
    i = 0
    blackIm = Image.new('RGB',(800, 800), 'Black')
    for img_path in img_paths:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        #平铺图片，不改变图片比例
        width, height = img.size
        copyIm = blackIm.copy()
        for left in range(0, 800, width):
            for top in range(0, 800, height):
                copyIm.paste(img, (left, top))
        img = copyIm
        img = img.resize((img_size,img_size),Image.LANCZOS) #用LANCZOS插值算法，resize质量高
        arr = np.asarray(img)
        X[i,:,:,:] = arr
        i += 1
    return X

def get_data_batch(X_path, Y, batch_size, img_size):
    while 1:
        for i in range(0, len(X_path), batch_size):
            x = get_image(X_path[i:i+batch_size], img_size)
            y = Y[i:i+batch_size]
            yield x, y  #返回生成器


# # 自定义metrics

# In[9]:


#建立keras后端计算fmeasure函数
import keras.backend as K

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
    return fbeta_score*100

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)


# In[10]:


#得到生成器
batch_size = 8
img_size = 500
train_generator = get_data_batch(X_train_path,y_train,batch_size=batch_size,img_size=img_size) 
val_generator = get_data_batch(X_val_path,y_val,batch_size=batch_size,img_size=img_size)


# # 搭建预训练fine-tune模型

# 1、预训练模型——InceptionResNetV2进行fine-tune训练

# In[11]:


from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *

def MODEL(MODEL,img_size,out_dims,func=None,weights=None,include_top=False):
    inputs = Input((img_size,img_size,3)) #实例化一个tensor
    x = inputs
    x = Lambda(func)(x)
    
    base_model = MODEL(weights=weights, include_top=include_top)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
#     x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(3072,activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dims, activation='sigmoid')(x)
    model = Model(inputs, x)
    return model


# In[12]:


from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input
model = MODEL(InceptionResNetV2,500,out_dims=6941,func=preprocess_input,weights='imagenet')
model.summary()


# In[13]:


checkpointer = ModelCheckpoint(filepath='Inresv2_weights_best.hdf5', 
                            monitor='val_fmeasure',verbose=1, save_best_only=True, mode='max')
reduce = ReduceLROnPlateau(monitor='val_fmeasure',factor=0.5,patience=1,verbose=1,min_delta=1e-4, mode='max')

adam = Adam(0.0001)
model.compile(optimizer = adam,
           loss='binary_crossentropy',
           metrics=['accuracy',fmeasure,recall,precision])
epochs = 20
history = model.fit_generator(train_generator,
       validation_data = val_generator,
       epochs=epochs,
       callbacks=[checkpointer,reduce],
       verbose=1,steps_per_epoch=np.ceil(28000/batch_size),validation_steps=np.ceil(7000/batch_size))


# 2、预训练模型——Xception进行fine-tune训练

# In[11]:


from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *

def MODEL(MODEL,img_size,out_dims,func=None,weights=None,include_top=False):
    inputs = Input((img_size,img_size,3)) #实例化一个tensor
    x = inputs
    x = Lambda(func)(x)
    
    base_model = MODEL(weights=weights, include_top=include_top)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
#     x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(2048,activation='relu')(x) #此处全连接与InceptionResNetV2不同
    x = Dropout(0.3)(x)
    x = Dense(out_dims, activation='sigmoid')(x)
    model = Model(inputs, x)
    return model


# In[12]:


from tensorflow.python.keras.applications.xception import preprocess_input
model = MODEL(Xception,500,out_dims=6941,func=preprocess_input,weights='imagenet')
model.summary()


# In[13]:


checkpointer = ModelCheckpoint(filepath='xception_weights_best.hdf5', 
                            monitor='val_fmeasure',verbose=1, save_best_only=True, mode='max')
reduce = ReduceLROnPlateau(monitor='val_fmeasure',factor=0.5,patience=1,verbose=1,min_delta=1e-4, mode='max')

adam = Adam(0.0001)
model.compile(optimizer = adam,
           loss='binary_crossentropy',
           metrics=['accuracy',fmeasure,recall,precision])
epochs = 20
history = model.fit_generator(train_generator,
       validation_data = val_generator,
       epochs=epochs,
       callbacks=[checkpointer,reduce],
       verbose=1,steps_per_epoch=np.ceil(28000/batch_size),validation_steps=np.ceil(7000/batch_size))


# 3、预训练模型——InceptionV3进行fine-tune训练

# In[12]:


from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *

def MODEL(MODEL,img_size,out_dims,func=None,weights=None,include_top=False):
    inputs = Input((img_size,img_size,3)) #实例化一个tensor
    x = inputs
    x = Lambda(func)(x)
    
    base_model = MODEL(weights=weights, include_top=include_top)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
#     x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(2048,activation='relu')(x) #此处全连接与InceptionResNetV2不同
    x = Dropout(0.3)(x)
    x = Dense(out_dims, activation='sigmoid')(x)
    model = Model(inputs, x)
    return model


# In[13]:


from tensorflow.python.keras.applications.inception_v3 import preprocess_input
model = MODEL(InceptionV3,500,out_dims=6941,func=preprocess_input,weights='imagenet')
model.summary()


# In[14]:


checkpointer = ModelCheckpoint(filepath='inceptionv3_weights_best.hdf5', 
                            monitor='val_fmeasure',verbose=1, save_best_only=True, mode='max')
reduce = ReduceLROnPlateau(monitor='val_fmeasure',factor=0.5,patience=1,verbose=1,min_delta=1e-4, mode='max')

adam = Adam(0.0001)
model.compile(optimizer = adam,
           loss='binary_crossentropy',
           metrics=['accuracy',fmeasure,recall,precision])
epochs = 20
history = model.fit_generator(train_generator,
       validation_data = val_generator,
       epochs=epochs,
       callbacks=[checkpointer,reduce],
       verbose=1,steps_per_epoch=np.ceil(28000/batch_size),validation_steps=np.ceil(7000/batch_size))


# # 模型融合

# In[15]:


#得到所有训练集和测试集
X_train_path = img_paths
# X_test_path = glob('test/*.jpg') #决赛测试集
X_test_path = glob('valid/*.jpg') #自由赛测试集
y_train2 = load_ytrain('tag_train.npz')

#test的生成器中没有y
def get_X_batch(X_path,batch_size,img_size):
    while 1:
        for i in range(0, len(X_path), batch_size):
            x = get_image(X_path[i:i+batch_size], img_size)

            yield x


# In[16]:


def build_MODEL(MODEL,img_size,out_dims,func=None,weights=None,include_top=False):
    inputs = Input((img_size,img_size,3)) 
    x = inputs
    x = Lambda(func)(x)
    
    base_model = MODEL(weights=weights, include_top=include_top) 
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(2048,activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dims, activation='sigmoid')(x)
    model = Model(inputs, x)
    return model

def FeatureExtract(MODEL,img_size,func=None,weight_path=None):
    base_model = build_MODEL(MODEL,img_size,out_dims=6941,func=func,weights=None)
    base_model.load_weights(weight_path)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
    
    batch_size = 8
    X_train_generator = get_X_batch(X_train_path, batch_size = batch_size, img_size = img_size)
    X_test_generator = get_X_batch(X_test_path, batch_size = batch_size, img_size = img_size)
    
    train_features = model.predict_generator(X_train_generator, steps = np.ceil(len(X_train_path) / batch_size), verbose=1) 
    test_features = model.predict_generator(X_test_generator,steps = np.ceil(len(X_test_path) / batch_size), verbose=1)
    
    # 保存bottleneck特征
    with h5py.File('%s_data.h5'%MODEL.__name__) as h:
        h.create_dataset("train",data = train_features)
        # h.create_dataset("test",data = test_features)
        h.create_dataset("valid",data = test_features)
        h.create_dataset('label',data = y_train2)


# 分别提取特征向量，便于后面进行融合

# In[18]:


from tensorflow.python.keras.applications.inception_v3 import preprocess_input
FeatureExtract(InceptionV3,500,func=preprocess_input,weight_path='inception_v3_weights_best_9_15_sigmoid_44.34666.hdf5')


# In[19]:


from tensorflow.python.keras.applications.xception import preprocess_input
FeatureExtract(Xception,500,func=preprocess_input,weight_path='xception_weights_best_9_15_sigmoid.hdf5')


# In[20]:


def build_MODEL2(MODEL,img_size,out_dims,func=None,weights=None,include_top=False):
    inputs = Input((img_size,img_size,3))
    x = inputs
    x = Lambda(func)(x)
    
    base_model = MODEL(weights=weights, include_top=include_top) 
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(3072,activation='relu')(x) #此处全连接与上面不同
    x = Dropout(0.3)(x)
    x = Dense(out_dims, activation='sigmoid')(x)
    model = Model(inputs, x)
    return model

def FeatureExtract(MODEL,img_size,func=None,weight_path=None):
    base_model = build_MODEL2(MODEL,img_size,out_dims=6941,func=func,weights=None)
    base_model.load_weights(weight_path)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
    
    batch_size = 8
    X_train_generator = get_X_batch(X_train_path, batch_size = batch_size, img_size = img_size)
    X_test_generator = get_X_batch(X_test_path, batch_size = batch_size, img_size = img_size)
    
    train_features = model.predict_generator(X_train_generator, steps = np.ceil(len(X_train_path) / batch_size), verbose=1) 
    test_features = model.predict_generator(X_test_generator,steps = np.ceil(len(X_test_path) / batch_size), verbose=1)
    
    # 保存bottleneck特征
    with h5py.File('%s_data.h5'%MODEL.__name__) as h:
        h.create_dataset("train",data = train_features)
        # h.create_dataset("test",data = test_features)
        h.create_dataset("valid",data = test_features)
        h.create_dataset('label',data = y_train2)


# In[21]:


from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input
FeatureExtract(InceptionResNetV2,500,func=preprocess_input,weight_path='Inresv2_weights_best_9_13_46.49088.hdf5')


# In[22]:


import h5py
X_train = []
X_test = []

#将保存好的特征向量提取出来并进行串接融合
for filename in ['Xception_data.h5','InceptionV3_data.h5','InceptionResNetV2_data.h5']:
    with h5py.File(filename,'r') as h:
        X_train.append(np.array(h['train']))
        # X_test.append(np.array(h['test']))
        X_test.append(np.array(h['valid']))
        y_train = np.array(h['label'])
X_train = np.concatenate(X_train,axis=1)
X_test = np.concatenate(X_test,axis=1)

from sklearn.utils import shuffle
np.random.seed(2018)
X_train,y_train = shuffle(X_train,y_train)

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=2018)


# In[23]:


#搭建融合后的模型
inputs = Input((X_train.shape[1:]))
x = Dropout(0.7)(inputs)
x = Dense(6941, activation='sigmoid')(x)
model = Model(inputs, x)

checkpointer = ModelCheckpoint(filepath='embedding.best_dropout0.7_9_20.hdf5',monitor='val_fmeasure',mode='max',
                               verbose=1, save_best_only=True) #保存最好模型权重
reduce = ReduceLROnPlateau(monitor='val_fmeasure',factor=0.5,patience=5,verbose=1,mode='max')
adam = Adam(0.0001)
model.compile(optimizer = adam,
           loss='binary_crossentropy',
           metrics=['accuracy',fmeasure,recall,precision])
epochs = 200
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs, batch_size=128,callbacks=[checkpointer,reduce],verbose=1)


# # 模型预测，得到结果

# In[24]:


model.load_weights('embedding.best_dropout0.7_9_20_59.93025.hdf5')
y_pred = model.predict(X_test)


# In[26]:


# Python
threshold = 0.5
def arr2tag(arr):
    tags = []
    for i in range(arr.shape[0]):
        tag = []
        index = np.where(arr[i] > threshold)  
        index = index[0].tolist()
        tag =  [hash_tag[j] for j in index]
        tags.append(tag)
    return tags
y_tags = arr2tag(y_pred)

import os
# img_name = os.listdir('test/')
img_name = os.listdir('valid/')#自由赛测试集

df = pd.DataFrame({'img_path':img_name, 'tags':y_tags})
for i in range(df['tags'].shape[0]):
    df['tags'].iloc[i] = ','.join(str(e) for e in  df['tags'].iloc[i])
df.to_csv('merged_moudle_best9_27_3_%s.csv'%(threshold),index=None)

