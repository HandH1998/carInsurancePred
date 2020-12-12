import json
from collections import OrderedDict
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

def preProc(dataTain):
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
    # print(xTrain.columns)
    # print(xTrain.head(10))
    # xTrain.boxplot(fontsize=6)
    # plt.show()
    # xTrain = torch.from_numpy(np.array(xTrain)).type(torch.FloatTensor)
    # yTrain = torch.from_numpy(np.array(yTrain)).type(torch.LongTensor)
    return xTrain, yTrain

dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
# response = dataTrain['Response'].value_counts()
# threshold = response[1] * 1.0 / response[0]
xTrain, yTrain = preProc(dataTrain)
rus=RandomUnderSampler(random_state=0)
xTrain,yTrain=rus.fit_sample(xTrain,yTrain)
print(xTrain)
print(yTrain)
print(sum(yTrain==1))
