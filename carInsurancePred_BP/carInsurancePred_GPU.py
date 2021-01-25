import random
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from utils.dataprocess import preProc,preProcTest,toJson
import sklearn.metrics as metrics

'''
    GPU版本主要是为了调整参数
'''


torch.manual_seed(1000)

BATCH_SIZE = 20000
EPOCH = 50
LR = 0.01
n_hidden=15
layer_num=3
weight_decay=0.001
momentum=0.4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, m=2):
        '''
        :param n_feature: 输入特征数
        :param n_hidden: 中间隐藏层单元个数
        :param n_output: 输出个数
        :param m: 隐藏层数量
        '''
        super().__init__()
        self.layer_num = m
        # 隐藏层列表
        self.hiddens = []
        # BN层列表
        self.bns = []
        # # 激活函数层
        self.act=[]
        for i in range(self.layer_num):
            if i == 0:
                hidden = torch.nn.Linear(n_feature, n_hidden)
            else:
                hidden = torch.nn.Linear(n_hidden, n_hidden)
            bn = torch.nn.BatchNorm1d(n_hidden, momentum=momentum)
            act= torch.nn.ReLU()
            setattr(self, 'hidden%i' % (i + 1), hidden)
            setattr(self, 'bn%i' % (i + 1), bn)
            setattr(self, 'act%i' % (i+1), act)
            self.hiddens.append(hidden)
            self.bns.append(bn)
            self.act.append(act)
        # 全连接层
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        '''
        :param x: batchsize*feature_nums
        :return: 正负类的概率
        '''
        for i in range(self.layer_num):
            x = self.act[i](self.bns[i](self.hiddens[i](x)))
        x = self.predict(x)
        return x
# best_fvalue=[]
# for weight_decay in [0.001,0.01,0.1,0.2,0.5,1,2]:
dataTrain = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_train.csv')
weight_negative = sum(dataTrain['Response']) * 1.0 / dataTrain.shape[0]
# 由于类别不平衡，采用权重解决，分别设置负类和正类的权重
weights = [weight_negative, 1 - weight_negative]  # [0.12293666666666667, 0.8770633333333333]
# pytorch要求权重输入为tensor
weights = torch.from_numpy(np.array(weights)).type(torch.FloatTensor)
xTrain,yTrain,scaler=preProc(dataTrain,toTensor=True)
torch.save(xTrain, 'xTrain.pt')
torch.save(yTrain, 'yTrain.pt')
xTrain,xValidation,yTrain,yValidation=train_test_split(xTrain,yTrain,test_size=0.2,random_state=1)

torch_dataset = Data.TensorDataset(xTrain, yTrain)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# here to(device)
net = Net(14, n_hidden, 2,layer_num).to(device)
print(net)
# weight_decay为L2正则项系数
optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99),weight_decay=weight_decay)
# here to(device) weights本身是CPU的tensor
loss_func = torch.nn.CrossEntropyLoss(weight=weights.to(device))
plt.ion()
fscoreList = []
epochList = []
lossList = []
xValid=xValidation.to(device)
yValid=yValidation.to(device)
for epoch in range(EPOCH):
    print('epoch:', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        # here to(device)
        x=batch_x.to(device)
        y=batch_y.to(device)
        output = net(x)
        # here to(device)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    output=net(xValid)
    prob = F.softmax(output, dim=1)
    pred_y = torch.max(prob, 1)[1]
    # change here
    fscore = metrics.f1_score(yValid.cpu(),pred_y.cpu())
    # if epoch==EPOCH-1:
    #     best_fvalue.append(fscore)
    # change here
    print('预测结果为1的数量：', sum(pred_y == 1).cpu().numpy())
    print('Fvalue:', fscore)
    # change here
    print('loss:', loss.cpu().detach().numpy())
    epochList.append(epoch)
    fscoreList.append(fscore)
    lossList.append(loss)

    plt.plot(epochList, fscoreList)
    plt.xlabel(u'迭代次数')
    plt.ylabel('Fvalue')
    plt.title(u'训练过程')
    #
    # # plt.plot(epochList, lossList)
    # # plt.xlabel(u'迭代次数')
    # # plt.ylabel(u'误差')
    # # plt.title(u'误差训练过程')
    plt.pause(0.2)
    net.train()

# print(best_fvalue)
# print('最优L2lambda ',best_fvalue.index(max(best_fvalue)))
# print('最优F1 ',max(best_fvalue))

plt.ioff()
plt.show()



torch.save(net, 'carInsurancePred.pt')
torch.save(net.state_dict(), 'carInsurancePredParams.pt')

dataTest = pd.read_csv(r'C:\Users\ZY\Desktop\ML\VI_test.csv')
xTest = preProcTest(dataTest,scaler,toTensor=True)
# net.load_state_dict(torch.load('carInsurancePredParams.pt'))
net.eval()

dic = OrderedDict()
# change here
output = net(xTest.to(device))
prob = F.softmax(output, dim=1)
pred_y = torch.max(prob, 1)[1].cpu().numpy()
equal1count = sum(pred_y == 1)
print('预测结果为1的数量:', equal1count)
toJson(pred_y)