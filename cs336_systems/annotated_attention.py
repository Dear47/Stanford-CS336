#%%
from einops import einsum, rearrange
import einx
import math
from jaxtyping import Int
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.nn_utils import softmax
from cs336_basics.model import CausalMultiHeadSelfAttention
#%%
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    with nvtx.range("final matmul"):
        o = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return o

# %%
class NVTXWrappedMHA(torch.nn.Module):
    def __init__(self, origin: CausalMultiHeadSelfAttention):
        super().__init__()
        self.origin = origin
    
    def forward(self, x:torch.Tensor, token_positions:Int[torch.Tensor, " ... seq"] | None = None)->torch.Tensor:
        *b, sequence_length, d_model = x.size()
        assert d_model == self.origin.d_model

        with nvtx.range("MHA"):
            with nvtx.range("QKV projections"):
                Q = self.origin.q_proj(x)
                K = self.origin.k_proj(x)
                V = self.origin.v_proj(x)
                Q, K, V = (
                    rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.origin.num_heads)
                    for X in (Q, K, V)
                )

            if token_positions is None:
                token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))
            
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            
            with nvtx.range("RoPE(Q)"):
                Q = self.origin.positional_encoder(Q, token_positions)
            with nvtx.range("RoPE(K)"):
                K = self.origin.positional_encoder(K, token_positions)

            with nvtx.range("Causal Mask"):
                seq = torch.arange(sequence_length, device=x.device)
                qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
                kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
                causal_mask = qi >= kj  # (query, key)  

            attn_output = annotated_scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)
            
            # with nvtx.range("Rearrange attention"):
            attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

            with nvtx.range("Output projection"):
                output = self.origin.output_proj(attn_output)
        return output            


# %%
