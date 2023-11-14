import torch as tf
import numpy as np
import os

def creat_data():
    x=tf.arange(12)
    print(x)
    print(x.shape)
    print("---------------")
    print("reshape:")
    x1=x.reshape(3,4)
    print(x1)

def data_opt():
    x=tf.zeros(3,4)
    y=tf.ones(3,4)
    print(tf.cat((x,y),dim=0))
    print(tf.cat((x,y),dim=1))
    print(y.sum())
    print("---------------")
    print("广播机制:")
    a=tf.arange(3).reshape((3,1))
    b=tf.arange(2).reshape((1,2))
    print(a)
    print(b)
    print(a+b)
    print("---------------")
    print("转为numpy:")
    print((a+b).numpy())

def linar_opt():
    A=tf.arange(40,dtype=tf.float32).reshape((2,5,4))
    B=A.clone()
    print(B)
    print(B.sum(axis=0))
    print(B.mean(axis=0))
    print("---------------")
    print("范数:")
    a=tf.tensor([3.0,4.0])
    print(a.norm(2))
            


if __name__=='__main__':
    # creat_data()
    # data_opt()
    linar_opt()