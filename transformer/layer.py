import torch.nn as nn
import torch
from .sublayers import MultiHeadAttention, PositionalFeedforward


class EncoderBlock(nn.Module):
    def __init__(self,encoder_layer):
        super().__init__()
        self.layer=encoder_layer

    def forward(self,x_mask):
        x,mask=x_mask
        x,_=self.layer(x,mask)
        return x,mask


class DecoderBlock(nn.Module):
    def __init__(self,decoder_layer):
        super().__init__()
        self.layer=decoder_layer

    def forward(self,x_params):
        x,enc_output,src_mask,trg_mask=x_params
        x,_,_=self.layer(x,enc_output,trg_mask,src_mask)
        return x,enc_output,src_mask,trg_mask

class EncoderLayer(nn.Module):
    def __init__(self,dmodel,dinner,headnum,dk,dv,dropout=0.1):
        super().__init__()
        self.selfattn=MultiHeadAttention(headnum,dmodel,dk,dv,dropout)
        self.posfeedforward=PositionalFeedforward(dmodel,dinner,dropout)

    def forward(self,input,selfattn_mask=None):
        output,selfattn=self.selfattn(input,input,input,selfattn_mask)
        output=self.posfeedforward(output)
        return output,selfattn
    

class DecoderLayer(nn.Module):
    def __init__(self,dmodel,dinner,headnum,dk,dv,dropout=0.1):
        super().__init__()
        self.selfattn=MultiHeadAttention(headnum,dmodel,dk,dv,dropout)
        self.enc_eattn=MultiHeadAttention(headnum,dmodel,dk,dv,dropout)
        self.posfeedforward=PositionalFeedforward(dmodel,dinner,dropout)

    def forward(self,input,enc_output,selfattn_mask=None,decattn_mask=None):
        output,selfattn=self.selfattn(input,input,input,selfattn_mask)
        output,decattn=self.enc_eattn(output,enc_output,enc_output,decattn_mask)
        output=self.posfeedforward(output)
        return output,selfattn,decattn
