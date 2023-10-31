import torch as tf
import random
import matplotlib.pyplot as plt


def synthetic_data(w,b,data_cnt):#(权重向量，常数项，样本数量)
    """
    生成 y = Xw + b + 噪声 的数据
    """
    x=tf.normal(0,1,(data_cnt,len(w)))
    y=tf.matmul(x,w.reshape((-1,1)))+b
    y+=tf.normal(0,0.01,y.shape)
    return x,y

def data_iter(batch_size,features,labels):
    """
    将样本随机排序后，分成多份数据依次返回，
    每份数据有batch_size个样本
    """
    fea_cnt=len(features)
    idx=list(range(fea_cnt))
    random.shuffle(idx)

    for i in range(0,fea_cnt,batch_size):
        batch_idx=tf.tensor(idx[i:min(i+batch_size,fea_cnt)])
        yield features[batch_idx],labels[batch_idx]

def linreg(x,w,b):
    """
    模型
    """
    return tf.matmul(x,w.reshape((-1,1)))+b

def squared_loss(y_hat,y):#(预测值，真实值)
    """
    损失函数
    """
    return (y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params,lr,batch_size):#(参数(包含w、b)，学习率，批量大小(用于梯度下降步长规范化))
    """
    小批量随机梯度下降
    """
    with tf.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()#梯度归零

if __name__=='__main__':
    """
    生成真实数据：
    """
    w0=tf.tensor([2,-3.4])
    b0=4.2
    features,labels=synthetic_data(w0,b0,1000)
    # print(labels)
    # plt.figure()
    # plt.scatter(features[:,1],labels,1)
    # plt.show()


    """
    超参数：
    """
    batch_size=10
    lr=0.01
    epoch_cnt=8
    net=linreg
    loss=squared_loss

    # for x,y in data_iter(batch_size,features,labels):
    #     print(x)
    #     print(y)
    #     break;

    """
    参数初始化：
    """
    w = tf.normal(0,0.01,size=(2,),requires_grad=True)
    b=tf.zeros(1,requires_grad=True)
    # print(w,'\n',b)

    for epoch in range(epoch_cnt):
        for X,y in data_iter(batch_size,features,labels):
            l = loss(net(X,w,b),y)# X、y的小批量损失
            l.sum().backward()
            sgd([w,b],lr,batch_size)
        
        with tf.no_grad():
            train_l=loss(net(features,w,b),labels).mean()
            print("epoch %d, loss %f"%(epoch+1,train_l))


        plt.figure()
        plt.scatter(features[:,1],labels,1)
        plt.scatter(features[:,1],net(features,w,b).detach(),1)
        plt.show()


    print(f'w的估计误差: {w0-w.reshape(w0.shape)}')
    print(f'b的估计误差: {b0-b}')