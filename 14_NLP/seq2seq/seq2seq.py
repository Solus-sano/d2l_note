import collections
import math
import torch
from torch import nn

class Seq2SeqEncoder(nn.Module):
    def __init__(
            self, vocab_size, embed_size,
            num_hiddens, num_layer,
            dropout=0, **kwargs
    ):
        super().__init__(**kwargs)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layer, dropout=dropout)
    
    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        op, state = self.rnn(X)
        return op, state
    

class Seq2SeqDecoder(nn.Module):
    def __init__(
            self, vocab_size, embed_size,
            num_hiddens, num_layer,
            dropout=0, **kwargs
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layer, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def forward(self, X, state):
        X = self.embedding(X).permute(1,0,2)
        context = state[-1].repeat(X.shape[0],1,1)
        X_and_cont = torch.cat((X, context), 2)
        op, state = self.rnn(X_and_cont, state) 
        op = self.dense(op).permute(1,0,2)
        return op, state

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange(
        (maxlen), dtype=torch.float32,
        device=X.device                    
    )[None, :] >= valid_len[:, None] 
    X[mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带mask的softmax交叉熵损失函数"""

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len=valid_len)
        self.reduction = 'none' #保留shape不取mean
        unweighted_loss = super().forward(pred.permute(0,2,1),label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss 






def test_Enc():
    encoder = Seq2SeqEncoder(
        vocab_size = 10,
        embed_size = 8,
        num_hiddens = 16,
        num_layer = 2
    )

    encoder.eval()
    X = torch.zeros((4,7), dtype=torch.long)
    op, state = encoder(X)
    print(op.shape,'\n',state.shape)

def test_Dec():
    encoder = Seq2SeqEncoder(
        vocab_size = 10,
        embed_size = 8,
        num_hiddens = 16,
        num_layer = 2
    )
    decoder = Seq2SeqDecoder(
        vocab_size = 10,
        embed_size = 8,
        num_hiddens = 16,
        num_layer = 2
    )

    encoder.eval()
    decoder.eval()
    X = torch.zeros((4,7), dtype=torch.long)
    state = decoder.init_state(encoder(X))
    op, state = decoder(X, state)
    print(op.shape,'\n',state.shape)

if __name__ == '__main__':
    # test_Dec()
    X = torch.arange(1,7).reshape((2,3))
    print(sequence_mask(X,torch.tensor([1,2]))) 