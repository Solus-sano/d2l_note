import torch as tf
from torch.utils import data 
from torch import nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt

class Accumulator:
    """
    累加器
    """
    def __init__(self,n):
        self.data=[0.0 for i in range(n)]

    def add(self,*args):#累加
        self.data=[a+float(b) for a,b in zip(self.data,args)]

    def reset(self):#归零
        self.data=[0.0 for i in range(len(self.data))]

    def __getitem__(self,idx):
        return self.data[idx]

def accuracy(y_hat,y):
    """
    计算一个batch的预测精度
    """
    if y_hat.shape[0]>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)#取概率最大的下标作为预测类别
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())#预测正确的样本数

def evaluate_accuracy(net, data_iter):
    """
    评估模型在指定数据集上的精度
    """
    if isinstance(net,tf.nn.Module):
        net.eval()#将模型设置为评估模式(即不计算梯度)
    m = Accumulator(2)
    with tf.no_grad():
        for X,y in data_iter:
            m.add(accuracy(net(X),y),y.numel())
    return m[0]/m[1]

def load_data(batch_size,resize=None,worker_cnt=4):
    """
    下载数据集，并将其加入到内存中，生成迭代器并返回
    """
    trans=[transforms.ToTensor()]
    if resize:
        trans.append(transforms.Resize(resize))
    trans=transforms.Compose(trans)

    #下载训练数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data",
        train=True,
        transform=trans,
        download=True
    )

    #下载验证数据集
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data",
        train=False,
        transform=trans,
        download=True
    )
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,
                            num_workers=worker_cnt),
            data.DataLoader(mnist_test,batch_size,shuffle=True,
                            num_workers=worker_cnt))

def init_weight(m):
    """初始化网络权重"""
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

def train_epoch(net,train_iter,loss,updater):
    """
    训练模型时的单个迭代周期
    """
    if isinstance(net,tf.nn.Module):
        net.train()#将模型设置为训练模式（可计算梯度）
    m = Accumulator(3)#累加器，（损失总和，预测正确样本数，样本总数）

    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)

        #使用pytorch内置优化器：
        if isinstance(updater,tf.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()

        #自定义优化器：
        else:
            l.sum().backward()
            updater(X.shape[0])#即batch_size

        m.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return m[0]/m[2] , m[1]/m[2]

def train(net, train_iter, test_iter, loss, epoch_cnt, updater):
    """总训练函数"""
    # anim = Animator(xlabel='epoch', xlim=[1, epoch_cnt], ylim=[0.3, 0.9],
    #                 legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(epoch_cnt):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        # anim.add(epoch+1, train_metrics + (test_acc,))
        print("epoch %d, loss %f, train_accuracy %f, test_accuracy %f"
        %(epoch+1,train_metrics[0], train_metrics[1], test_acc))
    train_loss, train_acc = train_metrics

def labels_to_txt(labels):
    """
    返回数据集对应的文本标签
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_image(imgs,row_cnt,col_cnt,titles=None,scale=1.5):
    """
    绘制图像
    """
    fig_size=(col_cnt*scale,row_cnt*scale)
    _,axis = plt.subplots(row_cnt,col_cnt,figsize=fig_size)
    axis = axis.flatten()
    for i,(ax,img) in enumerate(zip(axis,imgs)):
        img=img[0]
        if tf.is_tensor(img):#图片张量
            ax.imshow(img.numpy())
        else:#PIL图片
            ax.imshow(img)
        # ax.axis.get_xaxis().set_visible(False)
        # ax.axis.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axis

def predict(net, test_iter, cnt=6):
    """测试"""
    X,y=next(iter(test_iter))
    trues = labels_to_txt(y)
    preds = labels_to_txt(net(X).argmax(axis=1))
    titles=[t + '\n' + p for t,p in zip(trues,preds)]
    show_image(X[0:cnt], 1, cnt, titles[0:cnt])


if __name__=='__main__':
    batch_size=256
    epoch_cnt=10
    lr=0.1
    train_iter, test_iter=load_data(batch_size)

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784,256),
                        nn.ReLU(),
                        nn.Linear(256,10))
    net.apply(init_weight)

    loss = nn.CrossEntropyLoss()
    updater = tf.optim.SGD(net.parameters(), lr=lr)

    train(net, train_iter, test_iter, loss, epoch_cnt, updater)

    predict(net, test_iter)
    plt.show()


