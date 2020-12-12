import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from collections import OrderedDict
import json


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        # self.hidden3=torch.nn.Linear(n_hidden,n_hidden)
        # self.hidden4=torch.nn.Linear(n_hidden,n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # x=F.relu(self.hidden3(x))
        # x=F.relu(self.hidden4(x))
        x = self.predict(x)
        return x


def preProcTest(dataTest):
    dataTest.dropna(axis=0, how='any', inplace=True)
    xTest=dataTest.drop(dataTest.columns[0], axis=1)
    # print(yTest)
    # print(xTest)
    xTest = pd.get_dummies(xTest, columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage'])
    temp = xTest.pop('Driving_License')
    xTest.insert(13, 'Driving_License', temp)
    temp = xTest.pop('Previously_Insured')
    xTest.insert(13, 'Previously_Insured', temp)
    xTest = pd.concat([xTest.iloc[:, :5].apply(lambda x: (x - np.min(x)) / np.std(x)), xTest.iloc[:, 5:]], axis=1)
    print(xTest.columns)
    print(xTest.head(10))
    # xTest.boxplot(fontsize=6)
    # plt.show()
    xTest = torch.from_numpy(np.array(xTest)).type(torch.FloatTensor)
    return xTest

def preProc(dataTrain):
    dataTrain.dropna(axis=0, how='any', inplace=True)
    dataTrain.drop(dataTrain.columns[0:2], axis=1, inplace=True)
    yTrain = dataTrain.iloc[:, -1]
    xTrain = dataTrain.iloc[:, :-1]
    # print(yTrain)
    # print(xTrain)
    xTrain = pd.get_dummies(xTrain, columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage'])
    temp = xTrain.pop('Driving_License')
    xTrain.insert(13, 'Driving_License', temp)
    temp = xTrain.pop('Previously_Insured')
    xTrain.insert(13, 'Previously_Insured', temp)
    xTrain = pd.concat([xTrain.iloc[:, :5].apply(lambda x: (x - np.min(x)) / np.std(x)), xTrain.iloc[:, 5:]], axis=1)
    print(xTrain.columns)
    # rus = RandomUnderSampler(random_state=0)
    # xTrain, yTrain = rus.fit_sample(xTrain, yTrain)
    # print(xTrain.head(10))
    # xTrain.boxplot(fontsize=6)
    # plt.show()
    xTrain = torch.from_numpy(np.array(xTrain)).type(torch.FloatTensor)
    yTrain = torch.from_numpy(np.array(yTrain)).type(torch.LongTensor)
    return xTrain, yTrain

def accuracyCal(y1, y2):
    return sum(y1 == y2) / y1.shape[0]


def precisionCal(y1, y2):
    return sum(np.all([y1 == 1, y1 == y2], axis=0)) * 1.0 / sum(y1 == 1)


def recallCal(y1, y2):
    return sum(np.all([y1 == 1, y1 == y2], axis=0)) * 1.0 / sum(y2 == 1)


# def fscoreCal(y1, y2):
def fscoreCal(p, r, b):
    return ((1 + b ** 2) * p * r) / ((p * b ** 2) + r)
    # tp = sum(np.all([y1 == 1, y1 == y2], axis=0))
    # tn = sum(np.all([y1 == 0, y1 == y2], axis = 0))
    # return 2*tp*1.0/(y1.shape[0]+tp-tn)

# dataTest=pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
dataTrain=pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
# xTest=preProcTest(dataTest)
xTest,yTest=preProc(dataTrain)
net = Net(14, 10, 2)
# xTest = torch.load('xTest.pt')
# yTest = torch.load('yTest.pt')
# threshold = torch.load('threshold.pt')
# net = torch.load('carInsurancePred.pt')
net.load_state_dict(torch.load('carInsurancePredParams.pt'))


# dic = OrderedDict()
output = net(xTest)
prob = F.softmax(output, dim=1)
# probUpdate = torch.cat((prob[:, 0].unsqueeze(-1) * threshold, prob[:, 1].unsqueeze(-1)), -1)
# prediction = torch.max(probUpdate, 1)[1]
prediction = torch.max(prob, 1)[1]
pred_y = prediction.numpy()
target_y = yTest.numpy()
p = precisionCal(pred_y, target_y)
r = recallCal(pred_y, target_y)
# fscore = fscoreCal(pred_y, target_y)
fscore = fscoreCal(p, r, 1)
print('Fvalue:', fscore)
equal1count = sum(pred_y == 1)
print('预测结果为1的数量:', equal1count)

print(pred_y)
# for i, res in enumerate(pred_y):
#     dic[str(i)] = res.tolist()
# # 一维张量dim只有-1和0，-1就是最高维1维
# jsonStr = json.dumps(dic, indent=4)
# with open('submission.json', 'w') as f:
#     f.write(jsonStr)
