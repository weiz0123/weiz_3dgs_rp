import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Minimal cross-attention block for later token fusion between
    DINO and VGGT token sequences.

    query_tokens:   [B, Nq, Cq]
    context_tokens: [B, Nk, Ck]
    output:         [B, Nq, Cq]
    """

    def __init__(self, query_dim, context_dim, num_heads=8, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()

        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim={query_dim} must be divisible by num_heads={num_heads}")

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.out_dropout = nn.Dropout(proj_dropout)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_ctx = nn.LayerNorm(context_dim)

    def forward(self, query_tokens, context_tokens, attention_mask=None, return_attention=False):
        if query_tokens.ndim != 3:
            raise ValueError(
                f"Expected query_tokens [B, Nq, Cq], got {tuple(query_tokens.shape)}"
            )
        if context_tokens.ndim != 3:
            raise ValueError(
                f"Expected context_tokens [B, Nk, Ck], got {tuple(context_tokens.shape)}"
            )

        batch_size, num_query, _ = query_tokens.shape
        _, num_context, _ = context_tokens.shape

        query_tokens = self.norm_q(query_tokens)
        context_tokens = self.norm_ctx(context_tokens)

        q = self.q_proj(query_tokens)
        k = self.k_proj(context_tokens)
        v = self.v_proj(context_tokens)

        q = q.view(batch_size, num_query, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_query, self.query_dim)
        output = self.out_proj(attended)
        output = self.out_dropout(output)

        if return_attention:
            return output, attn_weights
        return output


class SelfAttention(nn.Module):
    """
    Minimal self-attention block for token refinement within a single
    token sequence.

    tokens: [B, N, C]
    output: [B, N, C]
    """

    def __init__(self, embed_dim, num_heads=8, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(proj_dropout)

    def forward(self, tokens, attention_mask=None, return_attention=False):
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, N, C], got {tuple(tokens.shape)}")

        batch_size, num_tokens, _ = tokens.shape
        tokens = self.norm(tokens)

        qkv = self.qkv_proj(tokens)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.embed_dim)
        output = self.out_proj(attended)
        output = self.out_dropout(output)

        if return_attention:
            return output, attn_weights
        return output



import torch
import torch.nn as nn

class DenseFusionBlock(nn.Module):
    """
    A single layer that fuses VGGT (Geometry) with DINO (Semantics).
    """
    def __init__(self, vggt_dim=2048, dino_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 1. Cross-Attention: VGGT (Q) queries DINO (K, V)
        self.cross_attn = CrossAttention(
            query_dim=vggt_dim, 
            context_dim=dino_dim, 
            num_heads=num_heads,
            proj_dropout=dropout
        )
        
        # 2. Self-Attention: Refine fused VGGT tokens
        self.self_attn = SelfAttention(
            embed_dim=vggt_dim, 
            num_heads=num_heads,
            proj_dropout=dropout
        )
        
        # 3. MLP / Feed-Forward Network
        self.mlp = nn.Sequential(
            nn.Linear(vggt_dim, vggt_dim * 2),
            nn.GELU(),
            nn.Linear(vggt_dim * 2, vggt_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(vggt_dim)
        self.norm2 = nn.LayerNorm(vggt_dim)

    def forward(self, vggt_tokens, dino_tokens):
        # vggt_tokens: [B, N, 2048]
        # dino_tokens: [B, N, 768]
        
        # Cross-Attention (Residual Connection)
        x = vggt_tokens + self.cross_attn(vggt_tokens, dino_tokens)
        
        # Self-Attention (Residual Connection)
        x = x + self.self_attn(self.norm1(x))
        
        # FFN (Residual Connection)
        x = x + self.mlp(self.norm2(x))
        
        return x

class DenseFusionTransformer(nn.Module):
    def __init__(self, vggt_dim=2048, dino_dim=768, depth=2, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseFusionBlock(vggt_dim, dino_dim, num_heads) 
            for _ in range(depth)
        ])

    def forward(self, vggt_tokens, dino_tokens):
        # Reshape: VGGT and DINO often come in as [B, V, N, C]
        # We need to flatten the View dimension into the Batch dimension 
        # or sequence dimension for the transformer.
        
        b, v, n, c_v = vggt_tokens.shape
        _, _, _, c_d = dino_tokens.shape
        
        # Flatten B and V to process all views in parallel
        vggt_flat = vggt_tokens.view(b * v, n, c_v)
        dino_flat = dino_tokens.view(b * v, n, c_d)
        
        x = vggt_flat
        for layer in self.layers:
            x = layer(x, dino_flat)
            
        # Reshape back to [B, V, N, C]
        return x.view(b, v, n, c_v)