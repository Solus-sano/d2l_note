# from this import d
import numpy as np
import torch as tf
from torch.utils import data
from torch import nn

def synthetic_data(w,b,data_cnt):#(权重向量，常数项，样本数量)
    """
    生成 y = Xw + b + 噪声 的数据
    """
    x=tf.normal(0,1,(data_cnt,len(w)))
    y=tf.matmul(x,w.reshape((-1,1)))+b
    y+=tf.normal(0,0.01,y.shape)
    return x,y

def load_array(data_arrays,batch_size,is_train=True):
    """
    生成pytorch数据迭代器
    """
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

if __name__=='__main__':
    w0=tf.tensor([2,-3.4])
    b0=4.2

    features,labels = synthetic_data(w0,b0,1000)

    batch_size=10

    data_iter=load_array((features,labels),batch_size)
    # for i in data_iter:
    #     print(i)
    #     break

    """
    定义网络模型: 
    单层网络，全连接层
    并初始化
    """
    net=nn.Sequential(nn.Linear(2,1))
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = tf.optim.SGD(net.parameters(),lr=0.03)

    """
    训练:
    """
    epoch_cnt=3
    for epoch in range(epoch_cnt):
        for X,y in data_iter:
            l=loss(net(X),y)
            trainer.zero_grad()
            l.mean().backward()#反向传播计算梯度
            trainer.step()#根据梯度更新参数
        l = loss(net(features),labels)
        print("epoch %d: loss:%f"%(epoch+1,l))
            
    """
    检验
    """
    w=net[0].weight.data
    b=net[0].bias.data
    print("w估计误差，", w0-w.reshape(w0.shape))
    print("b估计误差：",b0-b)