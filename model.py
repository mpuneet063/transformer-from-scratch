from json import encoder

import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)    # in the embedding layer, we scale the embeddings by the square root of the model dimension to maintain the variance of the embeddings.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)  # to prevent overfitting, we apply dropout to the positional encodings during training.

        # create a matrix of shape (seq_len, d_model) to hold the positional encodings
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len,) to hold the position indices
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # shape: (seq_len, 1)
        # create a vector of shape (d_model,) to hold the dimension indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # shape: (d_model/2,)
        # apply the sine function to the even indices of the positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)  # shape: (seq_len, d_model/2)
        # apply the cosine function to the odd indices of the positional encodings
        pe[:, 1::2] = torch.cos(position * div_term)  # shape: (seq_len, d_model/2)

        # add a batch dimension to the positional encodings
        pe = pe.unsqueeze(0)  # shape: (1, seq_len, d_model)
        # register the positional encodings as a buffer so that they are not updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add the positional encodings to the input embeddings and apply dropout
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # shape: (batch_size, seq_len, d_model)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))  # learnable scale parameter (multiplicative, hence ones)
        self.beta = nn.Parameter(torch.zeros(1))  # learnable shift parameter (additive, hence zeros are possible)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # compute the mean along the last dimension
        std = x.std(-1, keepdim=True)  # compute the standard deviation along the last dimension
        return self.gamma * (x - mean) / (std + self.eps) + self.beta  # normalize and scale/shift

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # first linear layer to expand the dimension
        self.dropout = nn.Dropout(dropout)  # dropout layer to prevent overfitting
        self.linear2 = nn.Linear(d_ff, d_model)  # second linear layer to project back to the original dimension

    def forward(self, x):
        x = self.linear1(x)  # apply the first linear transformation
        x = torch.relu(x)  # apply ReLU activation function
        x = self.dropout(x)  # apply dropout
        x = self.linear2(x)  # apply the second linear transformation
        return x  # return the output

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads  # dimension of each head

        # define the linear layers for query, key, and value
        self.w_q = nn.Linear(d_model, d_model)  # W_q
        self.w_k = nn.Linear(d_model, d_model)  # W_k
        self.w_v = nn.Linear(d_model, d_model)  # W_v
        # output layer
        self.w_o = nn.Linear(d_model, d_model)  # W_o

        self.dropout = nn.Dropout(dropout)  # dropout layer to prevent overfitting

    @staticmethod
    # define the scaled dot-product attention function
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]  # dimension of each head

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # shape: (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = torch.softmax(attention_scores, dim=-1)  # shape: (batch_size, num_heads, seq_len, seq_len)
        if dropout is not None:
            attention_probs = dropout(attention_probs)  # apply dropout to the attention probabilities

        out = torch.matmul(attention_probs, value)  # shape: (batch_size, num_heads, seq_len, d_k)
        return out, attention_probs  # return the output and the attention probabilities (useful for visualization and analysis)
    
    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)  # shape: (batch_size, seq_len, d_model)
        key = self.w_k(k)  # shape: (batch_size, seq_len, d_model)
        value = self.w_v(v)  # shape: (batch_size, seq_len, d_model)

        # split the query, key, and value into multiple heads
        # (Batch size, Sequence Length, d_model) -> (Batch size, Sequence Length, num_heads, d_k) -> (Batch size, num_heads, Sequence Length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, d_k)

        x, self.attn = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], x.shape[2], self.num_heads * self.d_k)  # shape: (batch_size, seq_len, d_model) # contiguous() is used to ensure that the tensor is stored in a contiguous block of memory, which is necessary for the view operation to work correctly.
        x = self.w_o(x)  # shape: (batch_size, seq_len, d_model)
        return x  # return the output of the multi-head attention layer

class AddAndNorm(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # dropout layer to prevent overfitting
        self.norm = LayerNormalization()  # layer normalization to stabilize the training

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # apply layer normalization, then the sublayer, then dropout, and finally add the original input (residual connection)

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.attention_block = self_attention_block  # multi-head attention 
        self.feed_forward_block = feed_forward_block  # feed-forward network
        self.add_and_norms = nn.ModuleList([AddAndNorm(dropout) for _ in range(2)])  # two Add & Norm layers, one for attention and one for feed-forward

    def forward(self, x, src_mask):
        x = self.add_and_norms[0](x, lambda x: self.attention_block(x, x, x, src_mask))  # its called self-attention because the query, key, and value are all x itSELF. The src_mask is used to prevent attending to padding tokens or future tokens in the sequence.
        x = self.add_and_norms[1](x, self.feed_forward_block)  # apply the feed-forward network
        return x  # return the output of the encoder block

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # list of encoder blocks
        self.norm = LayerNormalization()  # final layer normalization after all encoder blocks

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  # pass the input through each encoder block
        return self.norm(x)  # apply final layer normalization and return the output

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # multi-head self-attention
        self.cross_attention_block = cross_attention_block  # multi-head cross-attention
        self.feed_forward_block = feed_forward_block  # feed-forward network
        self.add_and_norms = nn.ModuleList([AddAndNorm(dropout) for _ in range(3)])  # three Add & Norm layers, one for self-attention, one for cross-attention, and one for feed-forward

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.add_and_norms[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.add_and_norms[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))  
        x = self.add_and_norms[2](x, self.feed_forward_block)  # apply the feed-forward network
        return x  # return the output of the decoder block

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # list of decoder blocks
        self.norm = LayerNormalization()  # final layer normalization after all decoder blocks

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)  # pass the input through each decoder block
        return self.norm(x)  # apply final layer normalization and return the output

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # linear layer to project the decoder output to the vocabulary size

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)  # apply log softmax to get the log probabilities of the next token in the sequence; log softmax is used instead of softmax for numerical stability and to work better with the negative log-likelihood loss function during training.



class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder  # encoder module
        self.decoder = decoder  # decoder module
        self.src_embedding = src_embedding  # source embedding layer
        self.tgt_embedding = tgt_embedding  # target embedding layer
        self.tgt_pos = tgt_pos  # target positional encoding layer
        self.src_pos = src_pos  # source positional encoding layer
        self.projection_layer = projection_layer  # projection layer to map decoder output to vocabulary size

    def encode(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_embedding(src)  # embed the source sequence
        src = self.src_pos(src)  # add positional encoding to the source sequence
        return self.encoder(src, src_mask)  # pass the source sequence through the encoder

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        tgt = self.tgt_embedding(tgt)  # embed the target sequence
        tgt = self.tgt_pos(tgt)  # add positional encoding to the target sequence
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # pass the target sequence and encoder output through the decoder

    def project(self, x):
        return self.projection_layer(x)  # project the decoder output to the vocabulary size

def build_transformer(src_size:int, tgt_size:int, src_seq_len:int, tgt_seq_len:int, d_model:int = 512, N: int = 6, num_heads: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # create the embeddings
    src_embedding = InputEmbedding(d_model, src_size)
    tgt_embedding = InputEmbedding(d_model, tgt_size)

    # create the positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder and decoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_size)

    # create the complete transformer model
    transformer =  Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)  

    # initialize the parameters using Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # Xavier initialization for weights with more than one dimension

    return transformer  # return the complete transformer model