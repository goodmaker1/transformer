import math
import torch
import numpy as np
import torch.nn as nn
import copy
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(feature))
        self.b = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layer_norm(x + sublayer(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float, max_len=5000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"不能使用sin/cos位置编码，得到奇数维度 {dim}")
        pe = torch.zeros(max_len, dim)                # (max_len, D)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)                          # → (1, max_len, D) 方便与 (B,S,D) 广播
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):               # emb: (B,S,D)
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:, :emb.size(1)]     # 按 S 维对齐
        else:
            emb = emb + self.pe[:, step:step+1]
        return self.drop_out(emb)


def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn_softmax = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)
    return torch.matmul(self_attn_softmax, value), self_attn_softmax

class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn_softmax = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        x, self.attn_softmax = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linear_out(x)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def subsequent_mask(size, device=None):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    t = torch.from_numpy(mask) == 0
    return t.to(device) if device is not None else t

def pad_mask(src, trg, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(1)
    trg_pad = (trg != pad_idx).unsqueeze(1)
    seq_mask = subsequent_mask(trg.size(1), device=trg.device)
    trg_mask = trg_pad & seq_mask
    return src_mask, trg_mask

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)

def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)

    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, trg_mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, trg_mask))
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory, src_mask))
        return self.sublayer_connection[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super(Decoder, self).__init__()
        self.layers = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, trg_mask)
        return x

class FeatEmbedding(nn.Module):

    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model))

    def forward(self, x):
        return self.video_embeddings(x)


class TextEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class BaseTransformer(nn.Module):
    def __init__(self, vocab, d_model, d_ff, n_heads, n_layers, dropout, pad_idx, d_feat=None):
        super(BaseTransformer, self).__init__()
        self.vocab = vocab
        self.pad_idx = pad_idx

        attn = MultiHeadAttention(n_heads, d_model, dropout)
        feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.src_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.trg_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.pos_embed = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(n_layers, EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(feed_forward), dropout))
        self.decoder = Decoder(n_layers, DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(feed_forward), dropout))

        self.generator = Generator(d_model, vocab.n_vocabs)

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.pos_embed(x)
        return self.encoder(x, src_mask)

    def decode(self, trg, memory, src_mask, trg_mask):
        y = self.trg_embed(trg)
        y = self.pos_embed(y)
        return self.decoder(y, memory, src_mask, trg_mask)

    def forward(self, src, trg):
        # 只用最简两种 mask
        src_mask, trg_mask = pad_mask(src, trg, self.pad_idx)
        memory = self.encode(src, src_mask)
        out = self.decode(trg, memory, src_mask, trg_mask)
        logits = self.generator(out)
        return logits