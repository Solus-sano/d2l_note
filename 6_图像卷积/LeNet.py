import torch as tf
from torch import nn
from torch.utils import data
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt

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

"""LeNet"""
LeNet = nn.Sequential(         #init_shape: (1,1,28,28)
    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),#shape-> (1,6,28,28)
    nn.AvgPool2d(kernel_size=2,stride=2),#shape-> (1,6,14,14)
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),#shape-> (1,16,10,10)
    nn.AvgPool2d(kernel_size=2,stride=2),#shape-> (1,16,5,5)

    nn.Flatten(),#shape-> (1,16*5*5)
    nn.Linear(16*5*5,120),nn.ReLU(),#shape-> (1,120)
    nn.Linear(120,84),nn.ReLU(),#shape-> (1,84)
    nn.Linear(84,10)#shape-> (1,10)
)

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

def accuracy(y_hat,y):
    """计算一个batch的预测正确的样本数"""
    if y_hat.shape[0]>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)#取概率最大的下标作为预测类别
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())#预测正确的样本数

def init_weight(m):
    """初始化网络权重"""
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def evaluate_accuracy_gpu(net, data_iter,device=None):
    """评估模型在指定数据集上的精度"""
    if isinstance(net,tf.nn.Module):
        net.eval()#将模型设置为评估模式(即不计算梯度)
        if not device:
            device = next(iter(net.parameters())).device
    m = Accumulator(2)
    with tf.no_grad():
        for X,y in data_iter:
            X=X.to(device)
            y=y.to(device)
            m.add(accuracy(net(X),y),y.numel())
    return m[0]/m[1]
def train_epoch(net,train_iter,loss,updater,device):
    """训练模型时的单个迭代周期"""
    if isinstance(net,tf.nn.Module):
        net.train()#将模型设置为训练模式（可计算梯度）
    m = Accumulator(3)#累加器，（损失总和，预测正确样本数，样本总数）

    for X,y in train_iter:
        updater.zero_grad()
        X,y=X.to(device),y.to(device)
        y_hat = net(X)
        l = loss(y_hat,y)

        
        l.backward()
        updater.step()
        with tf.no_grad():
            m.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return m[0]/m[2] , m[1]/m[2]#loss,accuracy

def train_gpu(net, train_iter, test_iter, loss, epoch_cnt, updater,device=None):
    """总训练函数"""
    if not device:
        device = next(iter(net.parameters())).device
    net.apply(init_weight)
    print('train on ',device)
    net.to(device)
    for epoch in range(epoch_cnt):
        train_metrics = train_epoch(net, train_iter, loss, updater,device)
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # anim.add(epoch+1, train_metrics + (test_acc,))
        print("epoch %d, loss %f, train_accuracy %f, test_accuracy %f"
        %(epoch+1,train_metrics[0], train_metrics[1], test_acc))
    # train_loss, train_acc = train_metrics

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
        if titles:
            ax.set_title(titles[i])
    return axis

def predict(net, test_iter, cnt=6):
    """测试"""
    X,y=next(iter(test_iter))
    X, y = X.to(device), y.to(device)
    trues = labels_to_txt(y)
    preds = labels_to_txt(net(X).argmax(axis=1))
    titles=[t + '\n' + p for t,p in zip(trues,preds)]
    show_image(X[0:cnt].cpu(), 1, cnt, titles[0:cnt])



def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = tf.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with tf.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print("epoch %d, loss %f, train_accuracy %f, test_accuracy %f"
        %(epoch+1,train_l, train_acc, test_acc))
        # animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')

if __name__=='__main__':
    device = "cuda" if tf.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    batch_size=256
    epoch_cnt=10
    lr=0.9
    train_iter, test_iter=load_data(batch_size)

    net=LeNet
    net.apply(init_weight)
    loss=nn.CrossEntropyLoss()
    updater=tf.optim.SGD(net.parameters(),lr=lr)

    train_gpu(net,train_iter,test_iter,loss,epoch_cnt,updater,device)
    # train_ch6(net, train_iter, test_iter, epoch_cnt, lr, device)
    predict(net, test_iter)
    plt.show()
