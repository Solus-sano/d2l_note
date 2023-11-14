import torch as tf
import torchvision
from torch.utils import data
from torchvision import transforms
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

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None,
                xlim=None, ylim=None, xscale='linear', yscale='linear', 
                fmts=('-', 'm--', 'g-.', 'r:'),
                nrows=1, ncols=1,
                figsize=(3.5,2.5)):
                
        #增量得绘制多条线
        if legend is None:
            legend = []
        
        self.fig, self.axes = plt.subplots(nrows,ncols,figsize=figsize)
        if nrows*ncols==1:
            self.axes=[self.axes]

        # 捕获参数
        # def self.config_axes()
        self.axes[0].set_xlabel(xlabel)
        self.axes[0].set_ylabel(ylabel)
        self.axes[0].set_xlim(xlim)
        self.axes[0].set_ylim(ylim)
        self.axes[0].set_xscale(xscale)
        self.axes[0].set_yscale(yscale)
        if legend:self.axes[0].legend()
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self,x,y):
        """向图表中加入多个数据点"""
        if not hasattr(y,'__len__'):
            y=[y]
        n=len(y)
        if not hasattr(x, "__len__"):
                x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        # self.config_axes()

        plt.show()
        


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

def softmax(X):
    """
    对矩阵X的每一行进行softmax操作
    """
    X_exp=tf.exp(X)
    partition=X_exp.sum(axis=1,keepdim=True)
    return X_exp/partition

def net(X):
    """
    实现softmax回归模型
    """
    return softmax(tf.matmul(X.reshape((-1,w.shape[0])),w)+b)
    #X直接reshape是否正确？

def cross_entropy(y_hat,y):
    """
    交叉熵损失函数
    """
    return -tf.log(y_hat[range(len(y_hat)),y]).sum()

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
        net.eval()#将模型设置为评估模式
    m = Accumulator(2)
    with tf.no_grad():
        for X,y in data_iter:
            m.add(accuracy(net(X),y),y.numel())
    return m[0]/m[1]

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
            l.mean().backward()
            updater.step()

        #自定义优化器：
        else:
            l.sum().backward()
            updater(X.shape[0])#即batch_size

        m.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return m[0]/m[2] , m[1]/m[2]

def train(net, train_iter, test_iter, loss, epoch_cnt, updater):
    # anim = Animator(xlabel='epoch', xlim=[1, epoch_cnt], ylim=[0.3, 0.9],
    #                 legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(epoch_cnt):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        # anim.add(epoch+1, train_metrics + (test_acc,))
        print("epoch %d, loss %f, train_accuracy %f, test_accuracy %f"
        %(epoch+1,train_metrics[0], train_metrics[1], test_acc))
    train_loss, train_acc = train_metrics
    
def updater(batch_size):
    """
    小批量随机梯度下降
    """
    with tf.no_grad():
        for param in [w,b]:
            param-=lr*param.grad/batch_size
            param.grad.zero_()#梯度归零

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

if __name__=="__main__":
    batch_size=256
    lr=0.1
    epoch_cnt=10
    train_iter,test_iter=load_data(batch_size)

    ip_cnt=28*28
    op_cnt=10

    w=tf.normal(0,0.01,size=(ip_cnt,op_cnt),requires_grad=True)
    b=tf.zeros(op_cnt,requires_grad=True)

    print("模型初始准确率：%f"%(evaluate_accuracy(net,test_iter)))

    """训练"""

    train(net, train_iter, test_iter, cross_entropy,epoch_cnt,updater)

    """测试"""

    predict(net, test_iter)
    plt.show()
