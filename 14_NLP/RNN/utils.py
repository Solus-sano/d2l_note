import os
import collections
import torch as tf
from torch import nn

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

class Vocab:
    def __init__(self,tokens=[],min_freq=0,reserved_tokens=[]):
        counter=count_corpus(tokens)
        self.token_freqs=sorted(counter.items(),key=lambda x:x[1],reverse=True)

        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1      

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self,tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def unk(self): return 0
    def get_token_freqs(self): return self.token_freqs

def count_corpus(tokens):
    """统计词频"""
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for token in tokens]
    return collections.Counter(tokens)

def load_corpus(max_tokens=-1):
    with open(r"source/poetry.txt",'r',encoding='UTF-8') as f:
        data=f.readlines()
    tokens=[i for item in data for i in item[:-1]]

    vocab=Vocab(tokens)

    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        #取出模型中参与训练的参数
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = tf.sqrt(sum(tf.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

if __name__=='__main__':
    corpus, vocab = load_corpus()
    print(len(corpus))
    print(len(vocab))
    print(vocab.token_freqs[:100])
