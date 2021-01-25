import numpy as np

def accuracyCal(y1, y2):
    '''
    :param y1: 预测结果，numpy
    :param y2: 正确结果，numpy
    :return: accuracy
    '''
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