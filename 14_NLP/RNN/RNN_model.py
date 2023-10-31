import torch as tf
from torch import nn
from torch.nn import functional as F
device="cuda" if tf.cuda.is_available() else "cpu"

class RNN_model(nn.Module):
    def __init__(self,rnn_layer,vocab_size):
        super().__init__()
        self.rnn=rnn_layer
        self.vocab_size=vocab_size
        self.num_hiddens=self.rnn.hidden_size
        self.num_directions=2 if self.rnn.bidirectional else 1
        self.linear=nn.Linear(self.num_hiddens*self.num_directions,self.vocab_size)

    def forward(self,ip,state):
        X=F.one_hot(ip.T.long(),self.vocab_size)
        X=X.to(tf.float32)

        Y,state=self.rnn(X,state)
        op=self.linear(Y.reshape((-1,Y.shape[-1])))
        return op,state

    def begin_state(self,device,batch_size=1):
        return tf.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)


    