import os
import numpy as np
import pandas as pd
import torch as tf
from torch.utils import data
from torchvision import transforms,models
import matplotlib.pyplot as plt
from tqdm import tqdm

device='cuda' if tf.cuda.is_available() else 'cpu'

class Accumulator:
    """累加器"""
    def __init__(self,n):
        self.data=[0.0 for i in range(n)]

    def add(self,*args):#累加
        self.data=[a+float(b) for a,b in zip(self.data,args)]

    def reset(self):#归零
        self.data=[0.0 for i in range(len(self.data))]

    def __getitem__(self,idx):
        return self.data[idx]

def accuracy(y_hat,y):
    """计算一个batch的正确样本数"""
    y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,dataloader):
    """评估模型在指定数据集上的精度"""
    m=Accumulator(2)
    net.eval()
    with tf.no_grad():
        for X,y in dataloader:
            X=X.to(device); y=y.to(device)
            m.add(accuracy(net(X),y),y.numel())
    return m[0]/m[1]

def train_val_epoch(net,dataloader,loss_f,updater,mode):
    """
    单个epoch训练或预测
    返回平均预测损失函数值、平均预测精度
    """
    net.train()
    m=Accumulator(3)
    if mode=='train':
        print("training...")
        for X,y in tqdm(dataloader):
            X=X.to(device); y=y.to(device)
            y_hat=net(X)
            l=loss_f(y_hat,y)

            updater.zero_grad()
            l.sum().backward()
            updater.step()

            with tf.no_grad():
                m.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    elif mode=='val':
        print("valuating...")
        with tf.no_grad():
            for X,y in tqdm(dataloader):
                X=X.to(device); y=y.to(device)
                y_hat=net(X)
                l=loss_f(y_hat,y)
                m.add(float(l.sum()),accuracy(y_hat,y),y.numel())

    return m[0]/m[2],m[1]/m[2]

def train(net,train_dataloader,val_dataloader,loss_f,epoch_cnt,updater):

    train_loss_lst,val_loss_lst,train_accuracy_lst,val_accracy_lst=[],[],[],[]
    print("training device: ",device)
    
    for epoch in range(1,epoch_cnt+1):
        print("epoch %d\n---------------------------------:"%(epoch))
        train_loss, train_accuracy=train_val_epoch(net,train_dataloader,loss_f,updater,mode='train')
        val_loss, val_accuracy=train_val_epoch(net,val_dataloader,loss_f,updater,mode='val')
        print("train_loss: %f, val_loss: %f, train_accuracy %f, val_accuracy %f"
        %(train_loss,val_loss,train_accuracy,val_accuracy))

        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)
        train_accuracy_lst.append(train_accuracy)
        val_accracy_lst.append(val_accuracy)

    plt.figure()
    plt.plot(list(range(1,epoch_cnt+1)),train_accuracy_lst,label='train accuracy')
    plt.plot(list(range(1,epoch_cnt+1)),val_accracy_lst,label='test accuracy')
    plt.title('accuracy')
    plt.legend()
    
    plt.figure()
    plt.plot(list(range(1,epoch_cnt+1)),train_loss_lst,label='train loss')
    plt.plot(list(range(1,epoch_cnt+1)),val_loss_lst,label='val loss')
    plt.title('loss')
    plt.legend()