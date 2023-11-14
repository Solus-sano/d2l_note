import torch as tf

def trans_conv(X,K):
    """转置卷积"""
    h,w=K.shape
    Y=tf.zeros((X.shape[0]+h-1,X.shape[1]+w-1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i+h,j:j+w]+=X[i,j]*K
    return Y


if __name__=='__main__':
    X=tf.tensor([
        [0.0,1.0],
        [2.0,3.0]
    ])
    K=tf.tensor([
        [0.0,1.0],
        [2.0,3.0]
    ])
    print(trans_conv(X,K))
    """用torch库实现"""
    X,K=X.reshape((1,1,2,2)),K.reshape((1,1,2,2))
    tconv=tf.nn.ConvTranspose2d(1,1,kernel_size=2,bias=False)
    tconv.weight.data=K
    
    print(tconv(X))