import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as tree
import sklearn.metrics as metrics
import joblib
from collections import OrderedDict

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
    # print(xTrain.columns)
    # print(xTrain.head(10))
    # xTrain.boxplot(fontsize=6)
    # plt.show()
    # xTrain = torch.from_numpy(np.array(xTrain)).type(torch.FloatTensor)
    # yTrain = torch.from_numpy(np.array(yTrain)).type(torch.LongTensor)
    return xTrain, yTrain

def preProcTest(dataTest):
    dataTest.dropna(axis=0, how='any', inplace=True)
    xTest = dataTest.drop(dataTest.columns[0], axis=1)
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
    # xTest = torch.from_numpy(np.array(xTest)).type(torch.FloatTensor)
    return xTest

dataTrain=pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
xTrain,yTrain=preProc(dataTrain)
model=tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced',random_state=0,max_depth=8,min_impurity_decrease=0.01,min_samples_leaf=4)
model.fit(xTrain,yTrain)
pred_y=model.predict(xTrain)
fscore=metrics.f1_score(yTrain,pred_y)
print('Fvalue:',fscore)
joblib.dump(model,'carInsurancePredDTree.model')

# tree.plot_tree(model)
# tree.export_graphviz(model)
model=joblib.load('carInsurancePredDTree.model')
dataTest=pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
xTest=preProcTest(dataTest)
pred_y=model.predict(xTest)

countEqual1=sum(pred_y==1)
print('预测结果为1的数量:',countEqual1)
dic=OrderedDict()
for i,res in enumerate(pred_y):
    dic[str(i)]=res.tolist()

jsonStr=json.dumps(dic,indent=4)
with open('submission.json','w') as f:
    f.write(jsonStr)




