from nlp_dataset import load_data
import torch as tf
from torch import nn
from RNN_model import RNN_model
from utils import Accumulator,grad_clipping
from time import time
import math
import matplotlib.pyplot as plt
device="cuda" if tf.cuda.is_available() else "cpu"

batch_size, num_steps = 32, 30
train_iter, vocab = load_data(batch_size, num_steps)

num_hiddens=512
rnn_layer=nn.RNN(len(vocab),num_hiddens)
state = tf.zeros((1, batch_size, num_hiddens))

net=RNN_model(rnn_layer,vocab_size=len(vocab))
net=net.to(device)

num_epochs, lr = 200, 1

def predict(prefix,num_preds,net,vocab):
    net.load_state_dict(tf.load(r"latest.pt"))
    state=net.begin_state(batch_size=1,device=device)
    op=[vocab[prefix[0]]]
    get_input = lambda: tf.tensor([op[-1]], device=device).reshape((1, 1))

    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        op.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        op.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in op])

def train(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    loss=nn.CrossEntropyLoss()
    updater=tf.optim.SGD(net.parameters(),lr=lr)
    ppl_lst=[]
    for epoch in range(1,num_epochs+1):
        state, t_begin, m = None, time(),Accumulator(2)

        for X,Y in train_iter:
            if state is None or use_random_iter:
                state=net.begin_state(batch_size=X.shape[0],device=device)
            else:
                state.detach_()
            y=Y.T.reshape(-1)
            X,y=X.to(device),y.to(device)
            y_hat,state=net(X,state)
            l=loss(y_hat, y.long()).mean()

            updater.zero_grad()
            l.backward()
            grad_clipping(net,1)
            updater.step()

            m.add(l*y.numel(),y.numel())

        ppl,speed=math.exp(m[0]/m[1]), m[1]/(time()-t_begin)
        print("epoch: %d 困惑度: %.2lf speed: %.2lf"%(epoch,ppl,speed))
        ppl_lst.append(ppl)
        tf.save(net.state_dict(),"latest.pt")

    plt.plot(list(range(1,num_epochs+1)),ppl_lst)



if __name__ == '__main__':
    train(net,train_iter,vocab,lr,num_epochs,device)
    # plt.show()
    op=predict("time",100,net,vocab)
    print(op)


    