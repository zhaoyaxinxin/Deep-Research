import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .modules import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,head,model,dk,dv,dropout):
        super().__init__()
        self.head,self.model,self.dk,self.dv,self.dropout=head,model,dk,dv,nn.Dropout(dropout)
        self.qs=nn.Linear(model,head*dk,bias=False)
        self.ks=nn.Linear(model,head*dk,bias=False)
        self.vs=nn.Linear(model,head*dv,bias=False)
        self.fc=nn.Linear(head*dv,model,bias=False)
        self.norm=nn.LayerNorm(model,eps=1e-6)
        self.attention=ScaledDotProductAttention(dk**0.5,dropout)

    def forward(self,q,k,v,mask):
        batchsize=q.size(0)
        qlength,klength,vlength=q.size(1),k.size(1),v.size(1)
        residual=q
        q=self.qs(q).view(batchsize,qlength,self.head,self.dk)
        k=self.ks(k).view(batchsize,klength,self.head,self.dk)
        v=self.vs(v).view(batchsize,vlength,self.head,self.dv)
        q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q,attn=self.attention(q,k,v,mask)
        q=q.transpose(1,2).contiguous().view(batchsize,qlength,-1)
        q=self.dropout(self.fc(q))
        q+=residual
        q=self.norm(q)
        return q,attn

class PositionalFeedforward(nn.Module):
    def __init__(self,din,dhid,dropout):
        super().__init__()
        self.linear1=nn.Linear(din,dhid)
        self.linear2=nn.Linear(dhid,din)
        self.dropout=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(din,eps=1e-6)

    def forward(self,x):
        residual=x
        x=self.linear2(F.relu(self.linear1(x)))
        x=self.dropout(x)
        x+=residual
        x=self.norm(x)
        return x
