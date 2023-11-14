from cProfile import label
from cmath import sin
import torch as tf
import matplotlib.pyplot as plt

def back_ward_opt():
    x=tf.arange(4.0)
    print(x)
    x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
    x.grad  # 默认值是None
    y=2*tf.dot(x,x)
    print(x)
    print(y)
    print("---------------")
    print("反向传播求梯度:")
    y.backward()
    print(x.grad)

    x.grad.zero_()#算下一个前要归零
    y=x.sum()
    y.backward()
    print(x.grad)

    x.grad.zero_()#算下一个前要归零
    y=x*x
    y.sum().backward()
    print(x.grad)

def pro_5():
    x=tf.linspace(0,20.0,500)
    x.requires_grad_(True)
    y=tf.sin(x)
    # print(x)
    y.sum().backward()
    plt.plot(x.detach(),y.detach(),label='y')
    plt.plot(x.detach(),x.grad,label='dy/dx')
    plt.legend()
    plt.show()
if __name__=="__main__":
    # back_ward_opt()
    pro_5()