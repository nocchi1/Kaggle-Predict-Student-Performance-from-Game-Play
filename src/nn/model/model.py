from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(config, cat_inp_dim: List[int], table_dim: int = None, with_table: bool = False):
    if with_table:
        model = PSPTransformerModelWithTable(
            cat_inp_dim=cat_inp_dim,
            cat_emb_dim=config.cat_emb_dim,
            q_dim=18, # config.q_out_dict[level_group]
            dense_dim=config.dense_dim,
            table_dim=table_dim,
            nhead=config.nhead,
            dim_ff=config.dim_ff,
            dropout=config.tfm_drop_out,
            num_layers=config.num_layers
        )
        model = model.to(config.device)
    else:
        model = PSPTransformerModel(
                cat_inp_dim=cat_inp_dim,
                cat_emb_dim=config.cat_emb_dim,
                q_dim=18, # config.q_out_dict[level_group]
                dense_dim=config.dense_dim,
                nhead=config.nhead,
                dim_ff=config.dim_ff,
                dropout=config.tfm_drop_out,
                num_layers=config.num_layers
            )
        model = model.to(config.device)
    return model


class SequenceDropout(nn.Module):
    def __init__(self, p: float = 0.50):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        
    def forward(self, x):
        if self.training:
            binary_mask = (torch.rand(x.shape[1], device=x.device) > self.p).to(torch.float) / (1 - self.p) # inversed dropout
            binary_mask = binary_mask[None, ..., None]
            x = x * binary_mask
        return x


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x, mask):
        mask_ = mask.unsqueeze(dim=-1)
        mean_out = (x * mask_).sum(dim=1) / mask_.sum(dim=1) # (batch, dim)
        return mean_out


class AttentionPooling(nn.Module):
    def __init__(self, q_dim, dense_dim):
        super(AttentionPooling, self).__init__()
        self.q_emb = nn.Embedding(q_dim, dense_dim)
        self.dense_dim = dense_dim

    def forward(self, x, questions, mask):
        q = self.q_emb(questions) # (batch, q_num, dim)
        k = x.transpose(1, 2) # (batch, dim, seq)
        qk = torch.matmul(q, k) / torch.sqrt(torch.tensor(self.dense_dim, device=x.device))
        qk = F.softmax(qk + torch.where(mask == 1, 0., float('-inf')).unsqueeze(dim=1), dim=-1).unsqueeze(dim=2) # (batch, q_num, 1, seq)
        v = k.unsqueeze(dim=1) # (batch, 1, dim, seq)
        attention_out = (qk * v).sum(dim=-1) # (batch, q_num, dim)
        return attention_out


class PSPTransformerModel(nn.Module):
    def __init__(
        self,
        cat_inp_dim: int,
        cat_emb_dim: int,
        q_dim: int,
        dense_dim: int = 128,
        nhead: int = 8,
        dim_ff: int = 512,
        dropout: float = 0.1,
        num_layers: int = 1
    ):
        super().__init__()
        self.dense_dim = dense_dim
        self.q_dim = q_dim
        cat_dim1, cat_dim2, cat_dim3 = cat_inp_dim
        cat_emb1, cat_emb2, cat_emb3 = cat_emb_dim
        self.emb1 = nn.Embedding(cat_dim1, cat_emb1, padding_idx=0)
        self.emb2 = nn.Embedding(cat_dim2, cat_emb2, padding_idx=0)
        self.emb3 = nn.Embedding(cat_dim3, cat_emb3, padding_idx=0)
        
        inp_dim = cat_emb1 + cat_emb2 + cat_emb3
        self.linear = nn.Linear(inp_dim, dense_dim//2)
        self.linear2 = nn.Linear(1, dense_dim//2)
        self.linear3 = nn.Linear(dense_dim, dense_dim)
        
        self.lstm1 = nn.LSTM(input_size=dense_dim, hidden_size=dense_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=dense_dim, hidden_size=dense_dim, num_layers=1, batch_first=True, bidirectional=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dense_dim, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(dense_dim * 2, 1)
        self.mean_prehead = nn.Sequential(
            nn.Linear(dense_dim, dense_dim),
            nn.CELU()
        )
        self.attention_prehead = nn.Sequential(
            nn.Linear(dense_dim * q_dim, dense_dim),
            nn.CELU()
        )
        self.seq_dropout = SequenceDropout(p=0.10)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.10)
        self.relu = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(dense_dim)
        self.layernorm2 = nn.LayerNorm(dense_dim)
        self.layernorm3 = nn.LayerNorm(dense_dim * 2)
        self.mean_pooling = MeanPooling()
        self.attention_pooling = AttentionPooling(q_dim, dense_dim)

    def forward(self, cat_feat, num_feat, mask, questions, length):
        ### Input ###
        # cat input
        cat_x1 = self.emb1(cat_feat[:, :, 0])
        cat_x2 = self.emb2(cat_feat[:, :, 1])
        cat_x3 = self.emb3(cat_feat[:, :, 2])
        cat_x = torch.cat([cat_x1, cat_x2, cat_x3], dim=-1)
        cat_x = self.linear(cat_x)
        # num input
        num_x = num_feat.unsqueeze(dim=-1)
        num_x = self.linear2(num_x)
        # input
        base_x = torch.cat([cat_x, num_x], dim=-1)
        x = self.seq_dropout(base_x)

        ### LSTM ###
        x, _ = self.lstm1(x)
        # skip connection
        x = x + base_x
        x = self.layernorm1(x)
        
        ### Transformer ###
        x = self.encoder(x)

        ### Pooling ###
        # mean pooling
        mean_out = self.mean_pooling(x, mask)
        mean_out = mean_out.unsqueeze(dim=1).repeat(1, self.q_dim, 1)
        # attention pooling
        attention_out = self.attention_pooling(x, questions, mask)

        # all output concat
        concat_out = torch.cat([mean_out, attention_out], dim=-1)
        concat_out = self.layernorm3(concat_out)
        out = self.head(concat_out)
        return out.squeeze(dim=-1)
    

# まだ完全ではない
class PSPTransformerModelWithTable(nn.Module):
    def __init__(
        self,
        cat_inp_dim: int,
        cat_emb_dim: int,
        q_dim: int,
        table_dim: int,
        dense_dim: int = 128,
        nhead: int = 8,
        dim_ff: int = 512,
        dropout: float = 0.1,
        num_layers: int = 1
    ):
        super().__init__()
        
        self.dense_dim = dense_dim
        cat_dim1, cat_dim2, cat_dim3 = cat_inp_dim
        cat_emb1, cat_emb2, cat_emb3 = cat_emb_dim
        self.emb1 = nn.Embedding(cat_dim1, cat_emb1, padding_idx=0)
        self.emb2 = nn.Embedding(cat_dim2, cat_emb2, padding_idx=0)
        self.emb3 = nn.Embedding(cat_dim3, cat_emb3, padding_idx=0)
        self.q_emb = nn.Embedding(q_dim, dense_dim)
        
        inp_dim = cat_emb1 + cat_emb2 + cat_emb3
        self.linear = nn.Linear(inp_dim, dense_dim//2)
        self.linear2 = nn.Linear(1, dense_dim//2)
        self.linear3 = nn.Linear(dense_dim, dense_dim)
        
        self.lstm1 = nn.LSTM(input_size=dense_dim, hidden_size=dense_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=dense_dim, hidden_size=dense_dim, num_layers=1, batch_first=True, bidirectional=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dense_dim, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(table_dim, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(512, dense_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(dense_dim * 3, q_dim)

        self.seq_dropout = SequenceDropout(p=0.10)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.10)
        self.relu = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(dense_dim)
        self.layernorm2 = nn.LayerNorm(dense_dim)
        self.layernorm3 = nn.LayerNorm(dense_dim * 3)
        self.mean_pooling = MeanPooling()
        self.attention_pooling = AttentionPooling(q_dim, dense_dim)
        self.attention_prehead = nn.Sequential(
            nn.Linear(q_dim * dense_dim, dense_dim),
            nn.CELU()
        )

    def forward(self, cat_feat, num_feat, table_feat, mask, questions, length):
        ### Table MLP ###
        tab_out = self.mlp(table_feat)
        
        ### Sequence Input ###
        # cat input
        cat_x1 = self.emb1(cat_feat[:, :, 0])
        cat_x2 = self.emb2(cat_feat[:, :, 1])
        cat_x3 = self.emb3(cat_feat[:, :, 2])
        cat_x = torch.cat([cat_x1, cat_x2, cat_x3], dim=-1)
        cat_x = self.linear(cat_x)
        # num input
        num_x = num_feat.unsqueeze(dim=-1)
        num_x = self.linear2(num_x)
        # input
        base_x = torch.cat([cat_x, num_x], dim=-1)
        x = self.seq_dropout(base_x)

        ### LSTM ###
        x, _ = self.lstm1(x)
        # skip connection
        x = x + base_x
        x = self.layernorm1(x)
        
        ### Transformer ###
        x = self.encoder(x)

        ### Pooling ###
        # mean pooling
        mean_out = self.mean_pooling(x, mask)
        # attention pooling
        attention_out = self.attention_pooling(x, questions, mask)
        q_num, dim = attention_out.shape[1], attention_out.shape[2]
        attention_out = attention_out.view(-1, q_num * dim)
        attention_out = self.attention_prehead(attention_out)

        # all output concat
        concat_out = torch.cat([mean_out, attention_out, tab_out], dim=-1)
        concat_out = self.layernorm3(concat_out)
        out = self.head(concat_out)
        return out


