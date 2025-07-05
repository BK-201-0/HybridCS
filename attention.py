import torch
import torch.nn as nn
import math, copy
import torch.nn.functional as F

INF = 1e20

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1, device=None):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        # self.linears = self.clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.device = device
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # mask must be four dimension
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        residual = key

        # print(f"Shape of query: {query.shape}")
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k)
             for l, x in zip(self.linears, (query, key, value))]

        # print(f"Shape of query: {query.shape}")

        # ---------------------
        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = self.attention(query, key, value, mask=mask,
        #                               dropout=self.dropout)
        #
        # # 3) "Concat" using a view and apply a final linear.
        # x = x.contiguous() \
        #     .view(nbatches, -1, self.h * self.d_k)
        # return self.linears[-1](x)

        # U = torch.randn(self.d_k * self.d_k, nbatches).view(nbatches, 1, self.d_k, self.d_k)
        # U = U.to(self.device)
        # attn_weight = torch.matmul(query, U)

        attn_weight = query
        attn_weight = torch.matmul(attn_weight, key.transpose(-1, -2))
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask == 0, -INF)
        attn_weight = self.dropout(F.softmax(attn_weight, dim=-1))
        context = torch.matmul(attn_weight.transpose(-1,-2), value)
        context = context.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        context = self.dropout(context)
        output = context + residual
        return output


    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -INF)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class gMultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1, device=None):
        super(gMultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # Ensure that d_model can be divided by h
        self.d_k = d_model // h  # Each head's dimension
        self.h = h
        self.d_model = d_model
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)  # batch_size



        residual = query  # For residual connection

        # Apply the linear projections to the input vectors for query, key, and value
        # Here, we reshape input of shape [batch_size, d_model] to [batch_size, 1, h, d_k]
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # Attention mechanism: computing attention weights
        # Compute attention scores between query and key
        attn_weight = torch.matmul(query, key.transpose(-1, -2))  # [batch_size, h, seq_len, seq_len]

        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask == 0, -float('inf'))


        # attn_weight = attn_weight / math.sqrt(self.d_k)


        attn_weight = self.dropout(F.softmax(attn_weight, dim=-1))

        # Compute the context vector by applying attention weights on the value vectors
        context = torch.matmul(attn_weight, value)  # [batch_size, h, seq_len, d_k]

        context = context.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # Add residual connection
        context = self.dropout(context)
        output = context + residual

        return output

    def clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.d = embed_dim

    def forward(self, x, y, z):

        Q = self.query(x)
        K = self.key(y)
        V = self.value(z)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

