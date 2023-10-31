import torch as tf
from torch import nn
import numpy as np


def pool_2d(X,pool_size,mode='max'):
    """池化层"""
    p_h,p_w=pool_size
    Y=tf.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode =='max':#最大池化
                Y[i,j]=X[i:i+p_h,j:j+p_w].max()
            elif mode == 'avg':#平均池化
                Y[i,j]=X[i:i+p_h,j:j+p_w].mean()
    return Y

if __name__=='__main__':
    X=tf.tensor(np.array(range(9)).reshape((3,3)),dtype=float)
    print(X)
    print('------------------------')
    print(pool_2d(X,(2,2)))
    print('------------------------')

    """多通道"""
    X=tf.tensor(np.array(range(16)).reshape((1,1,4,4)),dtype=float)
    pool=nn.MaxPool2d(3,padding=1,stride=2)
    print(pool(X))#设置填充步幅