import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self,temperature,dropout):
        super().__init__()
        self.tempe=temperature
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,q,k,v,mask=None):
        attention=(q/self.tempe)@k.transpose(2,3)
        if mask is not None:
            attention=attention.masked_fill(mask==0,-1e9)
        attention=self.dropout(F.softmax(attention,dim=-1))
        output=attention@v
        return output,attention