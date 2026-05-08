import torch
import torch.nn as nn
import numpy as np
from transformer.layer import EncoderLayer, DecoderLayer, EncoderBlock, DecoderBlock


def padding_mask(seq,pad_idx):
    return (seq!=pad_idx).unsqueeze(-2)

def subsequent_mask(seq):
    batchsize,length=seq.size()
    return (1-torch.triu(torch.ones((1,length,length),device=seq.device),diagonal=1)).bool()

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):
    def __init__(self,inputnum,dwordvec,layersnum,head,dk,dv,dmodel,dinner,pad_idx,dropout=0.1,n_position=200,scale_emb=False):
        super().__init__()
        self.wordembedding=nn.Embedding(inputnum,dwordvec,pad_idx)
        self.positionencoding=PositionalEncoding(dwordvec,n_position)
        self.dropout=nn.Dropout(dropout)
        self.layer_stack=nn.Sequential(*[EncoderBlock(EncoderLayer(dmodel,dinner,head,dk,dv,dropout)) for _ in range(layersnum)])
        self.norm=nn.LayerNorm(dmodel,eps=1e-6)
        self.scale_emb=scale_emb
        self.dmodel=dmodel

    def forward(self,src_seq,src_mask):
        enc_output=self.wordembedding(src_seq)
        if self.scale_emb:
            enc_output*=self.dmodel**0.5
        enc_output=self.dropout(self.positionencoding(enc_output))

        enc_output,_=self.layer_stack((enc_output,src_mask))

        enc_output=self.norm(enc_output)
        return enc_output

class Decoder(nn.Module):
    def __init__(self,inputnum,dwordvec,layersnum,head,dk,dv,dmodel,dinner,pad_idx,dropout=0.1,n_position=200,scale_emb=False):
        super().__init__()
        self.wordembedding=nn.Embedding(inputnum,dwordvec,pad_idx)
        self.positionencoding=PositionalEncoding(dwordvec,n_position)
        self.dropout=nn.Dropout(dropout)
        self.layer_stack=nn.Sequential(*[DecoderBlock(DecoderLayer(dmodel,dinner,head,dk,dv,dropout)) for _ in range(layersnum)])
        self.norm=nn.LayerNorm(dmodel,eps=1e-6)
        self.scale_emb=scale_emb
        self.dmodel=dmodel

    def forward(self,trg_seq,enc_output,src_mask,trg_mask):
        dec_output=self.wordembedding(trg_seq)
        if self.scale_emb:
            dec_output*=self.dmodel**0.5
        dec_output=self.dropout(self.positionencoding(dec_output))

        dec_output,_,_,_=self.layer_stack((dec_output,enc_output,src_mask,trg_mask))

        dec_output=self.norm(dec_output)
        return dec_output

class Transformer(nn.Module):
    def __init__(self,inputnum,dwordvec,enc_layers,dec_layers,head,dk,dv,dmodel,dinner,pad_idx,dropout=0.1,n_position=200,scale_emb=False):
        super().__init__()
        self.encoder=Encoder(inputnum,dwordvec,enc_layers,head,dk,dv,dmodel,dinner,pad_idx,dropout,n_position,scale_emb)
        self.decoder=Decoder(inputnum,dwordvec,dec_layers,head,dk,dv,dmodel,dinner,pad_idx,dropout,n_position,scale_emb)
        self.projection=nn.Linear(dmodel,inputnum,bias=False)

    def forward(self,src_seq,trg_seq,src_mask,trg_mask):
        enc_output=self.encoder(src_seq,src_mask)
        dec_output=self.decoder(trg_seq,enc_output,src_mask,trg_mask)
        output=self.projection(dec_output)
        return output
