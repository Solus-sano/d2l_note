import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch import nn
from attention_score import AdditiveAttention
from seq2seq.seq2seq import Seq2SeqEncoder

class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self,
            vocab_size,
            embed_size,
            num_hiddens,
            num_layers,
            dropout=0,
            **kwargs         
        ):
        super().__init__(**kwargs)

        self.attention = AdditiveAttention(
            num_hiddens,
            num_hiddens,
            num_hiddens,
            dropout
        )

        self.embedding = nn.Embedding(vocab_size,embed_size)
        
        self.rnn = nn.GRU(
            embed_size + num_hiddens,
            num_hiddens,
            num_layers,
            dropout = dropout
        )

        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1,0,2), hidden_state, enc_valid_lens)
    
    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1,0,2)

        outputs, self._attention_weights = [], []
        for x in X:
            Q = torch.unsqueeze(hidden_state[-1], dim=1)
            # query的形状为(batch_size,1,num_hiddens)

            context = self.attention(
                Q, enc_outputs, enc_outputs, enc_valid_lens
            )# context的形状为(batch_size,1,num_hiddens)
            x = torch.cat(
                (context, torch.unsqueeze(x,dim=1)),
                dim=-1
            )# 在特征维度上连结
            print(x.shape)

            out, hidden_state = self.rnn(x.permute(1,0,2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weight)
        
        # 全连接层变换后，outputs的形状为(num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs,dim=0))
        return outputs.permute(1,0,2), [enc_outputs, 
                                        hidden_state, enc_valid_lens]
    
    def attention_weights(self):
        return self._attention_weights
    

def test():
    encoder = Seq2SeqEncoder(
        vocab_size=10,
        embed_size=8,
        num_hiddens=16,
        num_layer=3
    ); encoder.eval()

    decoder = Seq2SeqAttentionDecoder(
        vocab_size=10,
        embed_size=8,
        num_hiddens=16,
        num_layers=3
    ); decoder.eval()

    X = torch.zeros((4,7), dtype=torch.long) # (batch_size,num_steps)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    # print(output.shape)
    # print(len(state))
    # print(state[0].shape)
    # print(len(state[1]))
    # print(state[1][0].shape)

if __name__ == '__main__':
    test()