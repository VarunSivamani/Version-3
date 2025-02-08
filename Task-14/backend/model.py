import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

class LlamaRMSNorm(nn.Module):
    """
        LlamaRMSNorm is equivalent to T5LayerNorm.
        x / sqrt(mean(x^2, axis=-1) + eps) * weight
    """
    
    def __init__(self, d_model, eps=1e-5):
        """
        Args:
            d_model (int)  : Dimensionality of the input tensor.
            eps     (float): A small constant added to avoid division by zero.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x): # return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / norm
    
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        """
        Initializes the LlamaRotaryEmbedding module.

        Args:
            dim  (int)  : The dimensionality of the input embeddings.
            base (float): The base for calculating inverse frequency.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        # Create inv_freq for half the dimension
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        """
        Generates sinusoidal embeddings for a given sequence length.

        Args:
            seq_len (int): The length of the input sequence.
            device (torch.device): The device on which the tensors will be created.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sine and cosine embeddings of shape [seq_len, dim//2].
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        sinusoid_inp = torch.einsum('i,j->ij', t, self.inv_freq)  # Outer product
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()
        return sin, cos

    def apply_rotary_emb(self, x, emb):
        """
        Applies rotary positional embeddings to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_heads, dim].
            emb (Tuple[torch.Tensor, torch.Tensor]): Tuple of sine and cosine embeddings.

        Returns:
            torch.Tensor: Tensor with rotary embeddings applied, shape [batch_size, seq_len, num_heads, dim].
        """
        sin, cos = emb
        
        # Reshape sin and cos to match input dimensions
        # Add batch and head dimensions, and ensure the last dimension matches
        sin = sin.view(1, sin.shape[0], 1, sin.shape[1])  # [1, seq_len, 1, dim//2]
        cos = cos.view(1, cos.shape[0], 1, cos.shape[1])  # [1, seq_len, 1, dim//2]
        
        # Split input into half along last dimension
        x_half1, x_half2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # Apply rotary embeddings
        x_out1 = x_half1 * cos - x_half2 * sin
        x_out2 = x_half2 * cos + x_half1 * sin
        
        # Concatenate back along last dimension
        return torch.cat([x_out1, x_out2], dim=-1)
        
class LlamaSdpaAttention(nn.Module):
    """
    Implements multi-head self-attention with rotary embeddings and scaled dot-product attention.
    """
    
    def __init__(self, hidden_size, num_heads):
        """
        Args:
            hidden_size (int): Total dimensionality of the input embeddings.
            num_heads   (int): Number of attention heads.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask (torch.Tensor, optional): Mask to control which positions are attended to.
                                                     Shape [batch_size, seq_len].

        Returns:
            torch.Tensor: The output tensor after applying multi-head attention, of shape [batch_size, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = x.shape

        # Project queries, keys and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply rotary enbeddings
        rotary_emb = self.rotary_emb(seq_len, x.device)
        q = self.rotary_emb.apply_rotary_emb(q, rotary_emb)
        k = self.rotary_emb.apply_rotary_emb(k, rotary_emb)

        # Reshape for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot product attention
        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

class LlamaMLP(nn.Module):
    """
    Implements the MLP (Multilayer Perceptron) block used in LLaMA transformer layers.
    The MLP block applies a feed-forward network with non-linearity and projections.
    """
    
    def __init__(self, hidden_size, intermediate_size):
        """
        Initializes the MLP block with three linear layers and an activation function.

        Args:
            hidden_size (int): Dimensionality of the input and output embeddings.
            intermediate_size (int): Dimensionality of the intermediate feed-forward layer.
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        return self.down_proj(self.act_fn(self.gate_proj(x) + self.up_proj(x)))
    
class LlamaDecoderLayer(nn.Module):
    """
    Implements a single transformer decoder layer used in the LLaMA model. Each layer
    consists of a self-attention mechanism, followed by a feed-forward network (MLP),
    with residual connections and normalization applied at steps.
    """
    
    def __init__(self, hidden_size, num_heads, intermediate_size):
        """
        Args:
            hidden_size (int): Dimensionality of input and output embeddings.
            num_heads (int): Number of attention heads for multi-head self-attention.
            intermediate_size (int): Dimensionality of the feed-forward network.
        """
        super().__init__()
        self.attn = LlamaSdpaAttention(hidden_size, num_heads)
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass for a single decoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask (torch.Tensor, optional): Mask for controlling which positions to attend to.
                                                     Shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        # Self-attention
        residual = x
        x = self.attn(self.input_layernorm(x), attention_mask)
        x = x + residual

        # Feedforward
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = x + residual

        return x
    
class LlamaModel(nn.Module):
    """
    Implements the LLaMA transformer model, consisting of an embedding layer,
    a stack of decoder layers, and an output normalization layer.
    """
    
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_heads, intermediate_size):
        """
        Args:
            vocab_size (int): Size of the input vocabulary.
            hidden_size (int): Dimensionality of the input embeddings and hidden states.
            num_hidden_layers (int): Number of stacked decoder layers in the model.
            num_heads (int): Number of attention heads in each decoder layer.
            intermediate_size (int): Dimensionality of the feedforward layers in the MLPs.
        """
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(hidden_size)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Input tensor of token IDs of shape [batch_size, seq_len].
            attention_mask (torch.Tensor, optional): Attention mask to control which tokens are attended to.
                                                     Shape [batch_size, seq_len].

        Returns:
            torch.Tensor: The final hidden states of shape [batch_size, seq_len, hidden_size].
        """
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.norm(x)
    
class LlamaForCausalLM(nn.Module):
    """
    LLaMA model for causal language modeling. It integrates the
    transformer model (`LlamaModel`) with an additional output head for predicting 
    the next token in a sequence (causal language modeling task). The model shares weights 
    between the token embeddings and the output logits layer (lm_head).
    """
    
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_heads, intermediate_size, max_seq_len):
        """
        Args:
            vocab_size (int): The size of the input vocabulary.
            hidden_size (int): Dimensionality of the input embeddings and hidden states.
            num_hidden_layers (int): Number of stacked decoder layers.
            num_heads (int): Number of attention heads for each decoder layer.
            intermediate_size (int): Size of the intermediate layer in the MLPs.
            max_seq_len (int): Maximum sequence length.
        """
        super().__init__()
        self.model = LlamaModel(vocab_size, hidden_size, num_hidden_layers, num_heads, intermediate_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight # Weight Sharing

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Input tensor of token IDs with shape [batch_size, seq_len].
            attention_mask (torch.Tensor, optional): Attention mask to control which tokens are attended to.
                                                     Shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Logits for each token in the vocabulary with shape [batch_size, seq_len, vocab_size].
        """
        x = self.model(input_ids, attention_mask)
        logits = self.lm_head(x)
        return logits
    
    def resize_token_embeddings(self, new_num_tokens):
        """
        Args:
            new_num_tokens (int): The new number of tokens in the vocabulary.

        Returns:
            torch.nn.Embedding: The new token embedding layer with the updated vocabulary size.
        """
        # Resize the token embeddings
        old_embeddings = self.model.embed_tokens
        new_emmbeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)

        # Copy the exisiting embeddings
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_emmbeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        # Replace the old embeddings
        self.model.embed_tokens = new_emmbeddings

        # Update the lm_head to match the new embedding size
        old_lm_head = self.lm_head
        self.lm_head = nn.Linear(old_lm_head.in_features, new_num_tokens, bias=False)

        # Copy the exisiting weights
        num_tokens_to_copy = min(old_lm_head.out_features, new_num_tokens)
        self.lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]

        # Maintain weight sharing
        self.lm_head.weight = self.model.embed_tokens.weight

        return self.model.embed_tokens