import pandas as pd
import torch
import numpy as np
import json
import zipfile
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def preProc(dataTrain, toTensor=False, toNumpy=False):
    '''
    :param dataTrain: 训练集原始数据，DataFrame格式
    :param toTensor: 是否输出为Tensor
    :param toNumpy: 是否输出为Numpy
    :return: 处理后数据xTrain,yTrain,标准化器standscaler
    '''
    # 去空值
    dataTrain.dropna(axis=0, how='any', inplace=True)
    # 去无用的列
    dataTrain.drop(dataTrain.columns[0:2], axis=1, inplace=True)
    # 100%采样，随机排列顺序
    # dataTrain = dataTrain.sample(frac=1)
    # 划分特征与标签
    yTrain = dataTrain.iloc[:, -1]
    xTrain = dataTrain.iloc[:, :-1]
    # 非数值型特征转成one-hot
    xTrain = pd.get_dummies(xTrain, columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage'])
    # 调整特征顺序 强迫症行为
    temp = xTrain.pop('Driving_License')
    xTrain.insert(13, 'Driving_License', temp)
    temp = xTrain.pop('Previously_Insured')
    xTrain.insert(13, 'Previously_Insured', temp)
    # 数值型特征标准化，然后与非数值型特征连接
    standscaler=StandardScaler()
    # columns=xTrain.iloc[:, :5].columns
    # xTrain = pd.concat([xTrain.iloc[:, :5].apply(lambda x: (x - np.mean(x)) / np.std(x)), xTrain.iloc[:, 5:]], axis=1)
    # xTrain = pd.concat([pd.DataFrame(standscaler.fit_transform(xTrain.iloc[:, :5]),columns=columns), xTrain.iloc[:, 5:]], axis=1)
    xTrain = pd.DataFrame(standscaler.fit_transform(xTrain),columns=xTrain.columns)
    # 下采样解决类别不平衡问题，但是效果不好
    # rus = RandomUnderSampler(random_state=0)
    # xTrain, yTrain = rus.fit_sample(xTrain, yTrain)
    # print(xTrain.head(10))
    # xTrain.boxplot(fontsize=6)
    # plt.show()

    # 特征比较少，无需降维
    # pca = PCA(n_components=10)
    # pca.fit(xTrain)
    # print(pca.explained_variance_ratio_)
    # xTrain = pca.transform(xTrain)

    # 输出格式转换
    if toNumpy:
        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)
    if toTensor:
        xTrain = torch.from_numpy(np.array(xTrain)).type(torch.FloatTensor)
        # xTrain =torch.unsqueeze(torch.from_numpy(np.array(xTrain)).type(torch.FloatTensor),dim=1)
        yTrain = torch.from_numpy(np.array(yTrain)).type(torch.LongTensor)

    return xTrain, yTrain,standscaler


def preProcTest(dataTest, scaler,toTensor=False, toNumpy=False):
    '''
    :param dataTest: 测试集原始数据,DataFrame格式
    :param scaler: 训练集上拟合的标准化器
    :param toTensor: 是否输出为Tensor
    :param toNumpy: 是否输出为Numpy
    :return: 处理后数据xTest
    '''
    dataTest.dropna(axis=0, how='any', inplace=True)
    xTest = dataTest.drop(dataTest.columns[0], axis=1)
    xTest = pd.get_dummies(xTest, columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage'])
    temp = xTest.pop('Driving_License')
    xTest.insert(13, 'Driving_License', temp)
    temp = xTest.pop('Previously_Insured')
    xTest.insert(13, 'Previously_Insured', temp)
    # columns=xTest.iloc[:,:5].columns
    # xTest = pd.concat([xTest.iloc[:, :5].apply(lambda x: (x - np.mean(x)) / np.std(x)), xTest.iloc[:, 5:]], axis=1)
    # xTest = pd.concat([pd.DataFrame(scaler.transform(xTest.iloc[:, :5]),columns=columns), xTest.iloc[:, 5:]], axis=1)
    xTest = pd.DataFrame(scaler.transform(xTest),columns=xTest.columns)
    # 输出格式转换
    if toNumpy:
        xTest = np.array(xTest)
    if toTensor:
        xTest = torch.from_numpy(np.array(xTest)).type(torch.FloatTensor)
    return xTest


def toJson(pred, toZip=True):
    '''
    :param pred: 可迭代的预测结果,numpy
    :param toZip: 是否生成压缩文件.zip
    :return:
    '''
    # 有序字典
    dic = OrderedDict()
    # 预测结果有序存入字典
    for i, res in enumerate(pred):
        dic[str(i)] = res.tolist()
    # 字典转为jsonString
    jsonStr = json.dumps(dic, indent=4)
    # 写入json文件
    with open('submission.json', 'w') as f:
        f.write(jsonStr)
    # 压缩为zip
    if toZip:
        zip_file = zipfile.ZipFile('submission.zip', 'w')
        zip_file.write('submission.json')
        zip_file.close()

def sampleWeight(y,weight_negative):
    '''
    :param y: label
    :param weight_negative: 赋予反例的权重
    :return: 权重列表
    '''
    sample_weight=[]
    for i in y:
        if i==1:
            sample_weight.append(1-weight_negative)
        else:
            sample_weight.append(weight_negative)
    return sample_weight
# dataTrain=pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
#
# xTrain,yTrain,scaler=preProc(dataTrain)
# print(xTrain.columns)
# xTrain.boxplot()
# plt.show()
# print(xTrain.head(10))
# dataTest=pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
# xTest=preProcTest(dataTest,scaler)
# print(xTest.head(10))