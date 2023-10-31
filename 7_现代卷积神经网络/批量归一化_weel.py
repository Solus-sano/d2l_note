import torch as tf
from torch import nn

def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    """批量归一化"""
    if not tf.is_grad_enabled():
        #非训练(不算梯度)情况
        X_hat=(X-moving_mean)/tf.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4)#X是全连接层或卷积层
        #求均值方差:
        if len(X.shape)==2:
            mean=X.mean(dim=0)
            var=((X-mean)**2).mean(dim=0)
        else:
            mean=X.mean(dim=(0,2,3),keepdim=True)
            var=((X-mean)**2).mean(dim=(0,2,3),keepdim=True)
        
        X_hat=(X-mean)/tf.sqrt(var+eps)

        moving_mean=momentum*moving_mean+(1.0-momentum)*mean
        moving_var=momentum*moving_var+(1.0-momentum)*var
        #迭代很多次后可收敛于全局均值和方差

    Y=gamma*X_hat+beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    def __init__(self,features_cnt,dims_cnt):
        super().__init__()
        if dims_cnt==2: shape=(1,features_cnt)
        else: shape=(1,features_cnt,1,1)
        #整个层需要维护四个参数，前两个需要被学习
        self.gamma=nn.Parameter(tf.ones(shape))
        self.beta=nn.Parameter(tf.zeros(shape))
        self.moving_mean=tf.zeros(shape)
        self.moving_var=tf.ones(shape)


    def forward(self,X):
        if self.moving_mean.device!=X.device:
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var=self.moving_var.to(X.device)

        Y,self.moving_mean,self.moving_var=batch_norm(
            X,self.gamma,self.beta,self.moving_mean,self.moving_var,
            eps=1e-5,momentum=0.9
        )
        return Y


if __name__=='__main__':
    A=tf.Tensor(range(12)).reshape((4,3))
    print(A.mean(dim=0))