#%%
import os
import torch
import json
# from cs336_basics.utils.log import get_logger
from cs336_basics.utils.logger import get_logger
logger = get_logger(__name__,"module.txt")
#%%
class Linear(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        """
        Parameters:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output
            device (torch.device | None): Device to stores the parameters on
            dtype (torch.dtype | None): Data type of the parameters
            
            self.weight (Float[Tensor, "d_out, d_in"]): The linear weights to use
        """
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
        )
        sigma = (2/(in_features+out_features))**(0.5)
        torch.nn.init.trunc_normal_(self.weight,mean=0,std=sigma,a=-3*sigma,b=3*sigma)
        self.logger = logger

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters:
            x (Float[Tensor,"... d_in"]):
        Returns:
            out (Float[Tensor,"... d_out"]):
        """
        return x @ self.weight.transpose(-1,-2)
    
#%%
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        """
        map integer token IDs into a vector space of dimention d_model

        Parameters:
            num_embeddings (int): Size of the vocab
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters

            self.weight (Float[Tensor, "num_embeddings embedding_dim"]): The embedding vectors to fetch from
        """
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight,mean=0,std=1,a=-3,b=3)
        self.logger = logger

    def forward(self, token_ids:torch.Tensor)->torch.Tensor:
        """
        select the embedding vector for each token IDs by indexing into an embedding matrix using a token IDs
        Parameters:
            token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
        Returns:
            out (Float[Tensor, "... d_model"]):
        """
        if token_ids.device != self.weight.device:
            token_ids = token_ids.to(self.weight.device)
        return self.weight.index_select(0, token_ids.reshape(-1)).reshape(*token_ids.shape, -1)

#%%
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        """
        Parameters:
            d_model (int): Hidden dimension of the model
            eps (float): Epsilon value for numerical stability
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters

            self.weight (Float[Tensor, "d_model"]): RMSNorm weights
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model,device=device,dtype=dtype)
        )
        self.logger = logger

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters:
            x (Float[Tensor,"... d_model"]):
        Returns:
            out (Float[Tensor,"... d_model"]):
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        den =torch.sqrt(self.eps + (x * x).sum(-1) / self.d_model)
        den = den.unsqueeze(-1)
        o = x * self.weight / den
        return o.to(in_dtype)

#%%
class SWiGLUFFN(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        """
        Parameters:
            d_model (int): Dimensionality of the feedforward input and output
            d_ff (int): Dimensionality of the up-project happening internally to the swiglu
            self.w1 (Float[Tensor, "d_ff d_model"]): w1_proj tensor
            self.w2 (Float[Tensor, "d_model d_ff"]): w2_proj tensor
            self.w3 (Float[Tensor, "d_ff d_model"]): w3_proj tensor
        """
        super().__init__()
        self.w1 = Linear(d_model,d_ff,device,dtype)
        self.w2 = Linear(d_ff,d_model,device,dtype)
        self.w3 = Linear(d_model,d_ff,device,dtype)
        self.logger = logger

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters:
            x (Float[Tensor, "... d_model"]):
        Returns:
            out (Float[Tensor, "... d_model"]):
        """
        w1 = self.w1(x)  # [...,d_model] -> [...,d_ff]
        silu = w1 * torch.sigmoid(w1)
        w3 = self.w3(x)  # [...,d_model] -> [...,d_ff]
        w2 = self.w2(silu * w3)  # [...,d_ff] -> [...,d_model]
        return w2

#%%
class RoPE(torch.nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        """
        Parameters:
            theta (float): theta value for the RoPE
            d_k (int): Dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None): Device to store the buffer on
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for standard RoPE"

        position = torch.arange(0,max_seq_len,device=device).unsqueeze(-1)  # [max_seq_len,1]
        ids = torch.arange(0,d_k//2,device=device)  # [d_k//2],
        theta_map = position * (theta**(-2 * ids / d_k)) # [max_seq_len, d_k//2]
        self.register_buffer('cos',torch.cos(theta_map),persistent=False)
        self.register_buffer('sin',torch.sin(theta_map),persistent=False)
        self.logger = logger

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        """
        Parameters:
            x (Float[Tensor, "... seq_len d_k"]):
            token_position (Int[Tensor, "... seq_len"]): Used to slice cos and sin tensors along the seq dimension

        Returns:
            out (Float[Tensor, "... seq_len d_k"]):
        """
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        even = x[...,0::2]  # [..., seq_len, d_k//2]
        odd = x[...,1::2]  # [..., seq_len, d_k//2]
        o = torch.empty_like(x)
        o[...,0::2] = even * cos - odd * sin
        o[...,1::2] = odd * cos + even * sin
        return o

#%%
def silu(x:torch.Tensor):
    return x / (1 + torch.exp(-x))

#%%
def softmax(x:torch.Tensor, dim:int=-1)->torch.Tensor:
    """
    apply softmax to the i-th dimension of the input tensor x
    the output tensor have the same shape as the input tensor
    Parameters:
        x (Float[Tensor, "..."]):
        i (int): The i-th dimension you want to apply softmax to.
    """
    x = x - x.max(dim=dim)[0].unsqueeze(dim=dim).expand(*x.size())
    exp_x = torch.exp(x)
    sfm = exp_x / exp_x.sum(dim=dim).unsqueeze(dim) # [*x.size()]
    return sfm

#%%
def ScaledDotProductAttention(query:torch.Tensor,
                              key:torch.Tensor,
                              value:torch.Tensor,
                              mask:torch.Tensor|None=None)->torch.Tensor:
    """
    Parameters:
        query (Float[Tensor,"batchsize ... queries d_k"]): query tensor
        key (Float[Tensor,"batchsize ... keys d_k"]): key tensor
        value (Float[Tensor,"batchsize ... values d_v"]): value tensor
        mask (Bool[Tensor,"... queries keys"]): mask tensor
    Returns:
        out (Float[Tensor,"batchsize ... d_v"]): output of SDPA
    """
    qk = (query @ key.transpose(-1,-2)) # "batchsize ... queries keys"
    if mask is not None:
        qk = qk.masked_fill(mask==False,float('-inf'))
    o = softmax( qk * (key.size()[-1])**(-0.5),-1) @ value
    return o

#%%
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int, theta:float|None, device=None,dtype=None):
        """
        Parameters:
            d_model (int): Dimensionality of the feedforward input and output
            num_heads (int): Number of heads to use in multi-headed attention
            max_seq_len (int): Maximum sequence length to pre-cache
            theta (float | None): RoPE parameter
            device (torch.device | None): Device to store the buffer on
            dtype (torch.dtype | None): Data type of the parameters

            self.QKV_proj (Float[Tensor, "3 * h * d_k d_model"]): QKV_proj tensor
            self.O_proj (Float[Tensor, "d_model h * d_v"]): O_proj tensor
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        if theta is not None:
            self.rope = RoPE(theta,self.head_dim,max_seq_len,device)

        # seperated projection matrix
        # self.Q_proj = Linear(d_model,d_model,device,dtype)
        # self.K_proj = Linear(d_model,d_model,device,dtype)
        # self.V_proj = Linear(d_model,d_model,device,dtype)

        # a single projection matrix
        self.QKV_proj = Linear(d_model,d_model*3,device,dtype)
        self.O_proj = Linear(d_model,d_model,device,dtype)
        # pre-cache causal_mask
        self.register_buffer('causal_mask',None,persistent=False)
        self.device = device
        self.logger = logger

    def forward(self, x:torch.Tensor,mask:bool,token_positions:torch.Tensor)->torch.Tensor:
        """
        Parameters:
            x (Float[Tensor, "... seq_len d_model"]):
            mask (bool):
            token_position (Int[Tensor, "... seq_len"] | None):
        Returns:
            Float[Tensor, "... seq_len d_model"]
        """
        seq_len = x.size()[-2]
        if mask:
            if (self.causal_mask is None or self.causal_mask.size(0)<seq_len):
                self.causal_mask = torch.triu(torch.ones(seq_len,seq_len,dtype=torch.bool)).T.to(self.device)
        causal_mask = self.causal_mask[:seq_len,:seq_len]
        
        QKV = self.QKV_proj(x)  # "... seq_len 3*h_d_k"
        
        # seperated projection matrix
        # Q = self.Q_proj(x)  # "... seq_len h*d_k"
        # K = self.K_proj(x)  # "... seq_len h*d_k"
        # V = self.V_proj(x)  # "... seq_len h*d_v"

        # a single projection matrix
        Q,K,V = QKV.chunk(3,dim=-1)  # "... seq_len h*d_k"

        Q = Q.view(*x.size()[:-1],self.num_heads,self.head_dim).transpose(-2,-3)  # "... h seq_len d_k"
        K = K.view(*x.size()[:-1],self.num_heads,self.head_dim).transpose(-2,-3)  # "... h seq_len d_k"
        V = V.view(*x.size()[:-1],self.num_heads,self.head_dim).transpose(-2,-3)  # "... h seq_len d_v"

        if token_positions is not None:
            Q = self.rope(Q,token_positions)  # "...h seq_len d_k"
            K = self.rope(K,token_positions)  # "...h seq_len d_k"

        attention = ScaledDotProductAttention(Q,K,V,causal_mask)  # "... h seq_len d_v"
        attention = attention.transpose(-2,-3).contiguous().view(*x.size()[:-1],-1)  # "... seq_len h*d_v"
        O = self.O_proj(attention)
        return O
    
#%%
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len:int, theta:float, eps:float=1e-5, device=None, dtype=None):
        """
        Parameters:
            d_model (int): Dimensionality of the Transformer block input
            num_heads (int): Number of heads to use in multi-head self-attention
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer
            max_seq_len (int): Maximum sequence length to pre-cache
            theta (float | None): RoPE parameter
            eps (float): RMSNorm parameter
            device (torch.device | None): Device to store the buffer on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model,num_heads,max_seq_len,theta,device,dtype)
        self.swiglu = SWiGLUFFN(d_model,d_ff,device,dtype)
        self.rmsnorm1 = RMSNorm(d_model,eps,device,dtype)
        self.rmsnorm2 = RMSNorm(d_model,eps,device,dtype)
        self.logger = logger

    def forward(self, x:torch.Tensor, mask:bool=True)->torch.Tensor:
        """
        Parameters:
            x (Float[Tensor, "batch ... seq_len d_model"]):
            mask (Float[bool, "batch... seq_len seq_len"]):
            token_position (Int[Tensor, "batch... seq_len"] | None):

        Returns:
            out (Float[Tensor, "batch ... seq_len d_model"])
        """
        token_positions = torch.arange(x.size()[-2],dtype=torch.int)
        attn = self.mha(self.rmsnorm1(x),mask,token_positions)
        x = x + attn
        return x + self.swiglu(self.rmsnorm2(x))

#%%
class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size:int, context_length:int, num_layers:int,
                 d_model:int, num_heads:int, d_ff:int, theta:float, eps:float=1e-5,
                 device=None, dtype=None):
        """"
        Parameters:
            vocab_size (int): The size of the vocab, necessary for determining the dimensionality of the token embedding matrix
            context_length (int): The max context length, necessary for determining the dimensionality of the position embedding matrix
            num_layers (int): The number of Transformer blocks to use            
            d_model (int): Dimensionality of the Transformer block input
            num_heads (int): Number of heads to use in multi-head self-attention
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer
            theta (float | None): RoPE parameter
            eps (float): RMSNorm parameter
            device (torch.device | None): Device to store the buffer on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.context_length = context_length
        self.transblock = torch.nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,theta,eps,device,dtype) for _ in range(num_layers)])
        self.embedding = Embedding(vocab_size,d_model,device,dtype)
        self.rmsnorm = RMSNorm(d_model,eps,device,dtype)
        self.OLinear = Linear(d_model,vocab_size,device,dtype)
        self.logger = logger

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters:
            x (Float[Tensor, "batchsize ... seq_len"]):
        Returns:
            out (Float[Tesnor, "batchsize ... seq_len vocab_size"])
        """
        x = self.embedding(x)  # "batchsize ... seq_len d_model"
        for block in self.transblock:
            x = block(x)  # "batchsize ... seq_len d_model"
        x = self.rmsnorm(x)  # "batchsize ... seq_len d_model"
        x = self.OLinear(x)  # "batchsize ... seq_len vocab_size"
        return x

    @torch.no_grad()
    def generate(self,
                x: torch.Tensor,
                max_new_token: int,
                temperature: float = 1.0,
                top_k: int | None = None,
                top_p: float | None = None,
                eos_token_id: int | None = None) -> torch.Tensor:
        """
        Parameters:
            x (Float[Tensor, "1 seq_len"|"seq_len"]): Input IDs to condition when generating.
            max_new_token (int): Max number of tokens to generate.
            temperature (float): Temperature to use during generation.
            top_k (int | None): Sample from the top_k vocab items by probability.
            top_p (float | None): Sample from the smallest set of words with a cumulative probability not exceeding p. 
            eos_token_id (int | None): stop generation when generate this ID.
        Returns:
            A LongTensor of shape (max_new_token,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        original_seq_len = x.size(-1)
        
        for _ in range(max_new_token):
            if x.size(1) > self.context_length:
                x = x[:, -self.context_length:]
            
            logits = self.forward(x)  # (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_logits.size(-1))
                top_k_vals, _ = torch.topk(next_token_logits, k, dim=-1)
                threshold = top_k_vals[:, -1:]  # (1, 1)
                next_token_logits[next_token_logits < threshold] = float('-inf')

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                sorted_probs = softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cum_probs > top_p

                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                next_token_logits.masked_fill_(indices_to_remove, float('-inf'))

            probs = softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            x = torch.cat([x, next_token_id], dim=-1)

        return x[:, original_seq_len:]