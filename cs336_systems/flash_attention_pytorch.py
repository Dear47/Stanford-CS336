import torch
from einops import einsum

def flash_attention_bwd(Q, K, V, O, L, dO, is_causal):
    d = Q.shape[2]
    S = einsum(Q, K, "b q d, b k d -> b q k") * d**(-0.5)   # (batchsize, N_q, N_k)
    if is_causal:
        mask = torch.tril(torch.ones(S.shape[1],S.shape[2],device=S.device))
        S = S.masked_fill(mask==0,float("-inf"))
    P = torch.exp(S - L.unsqueeze(-1).to(torch.float32))  # (batchsize, N_q, N_k)
    dV = einsum(P, dO, "b q k, b q d -> b k d")  # (batchsize, N_k, d)
    dP = einsum(dO, V, "b q d, b k d -> b q k")  # (batchsize, N_q, N_k)
    D = torch.sum(O * dO, dim=-1)  # (batchsize, N_q,)
    dS = P * (dP - D.unsqueeze(-1))  # (batchsize, N_q, N_k)
    dQ = einsum(dS, K, "b q k, b k d -> b q d") * d**(-0.5)  # (batchsize, N_q, d)
    dK = einsum(dS, Q, "b q k, b q d -> b k d") * d**(-0.5)  # (batchsize, N_k, d)
    return dQ, dK, dV

# flash_attention_bwd_compiled = torch.compile(flash_attention_bwd,mode="max-autotune",fullgraph=False)

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Parameters:
            Q (Float[Tensor,"batchsize N_q d"]):Q Tensor
            K (Float[Tensor,"batchsize N_k d"]):K Tensor
            V (Float[Tensor,"batchsize N_k d"]):V Tensor
            is_causal (bool): causal mask
        Returns:
            Tuple[Float[Tensor, "batchsize N_q d"], Float[Tensor, "batchsize N_q"]]
            - O :O Tensor
            - L :logsumexp Tensor
        """
        batch_size = Q.shape[0]
        device = Q.device
        dtype = Q.dtype
        N_q = Q.shape[1]
        N_k = K.shape[1]
        d = Q.shape[2]
        scale = d ** (-0.5)
        B_q = 16
        B_k = 16
        T_q = (N_q + B_q - 1) // B_q
        T_k = (N_k + B_k - 1) // B_k

        O = torch.zeros_like(Q, dtype=dtype)  # (batchsize, N_q, d)
        L = torch.zeros(batch_size,N_q,device=device, dtype=dtype)  # (batchsize, N_q)
        
        for b in range(batch_size):
            Q_b = Q[b]  # (N_q,d)
            K_b = K[b]  # (N_k,d)
            V_b = V[b]  # (N_k,d)
            for i in range(T_q):
                row_start = i * B_q
                row_end = min(row_start + B_q , N_q)
                Q_i = Q_b[row_start:row_end, :]  # (B_q,d)
                # 累加器应该为float32以保证数值稳定性
                O_i = torch.zeros(row_end-row_start, d, device=device,dtype=torch.float32)  # (B_q,d)
                l_i = torch.zeros(row_end-row_start, device=device, dtype=torch.float32)  # (B_q,)
                m_i_old = torch.full((row_end-row_start,), float("-inf"), device=device, dtype=torch.float32)  # (B_q,)
                
                for j in range(T_k):
                    col_start = j * B_k
                    col_end = min(col_start + B_k, N_k)
                    K_j = K_b[col_start:col_end,:]  # (B_k,d)
                    V_j = V_b[col_start:col_end,:]  # (B_k,d)
                    S_ij = (Q_i @ K_j.T * scale).to(torch.float32)  # (B_q,B_k)
                    if is_causal:
                        offset = j * B_k - i * B_q
                        # 可以这么理解, 先计算全局行/列位置：
                        # global_row = i * B_q + r_indices
                        # global_col = j * B_k + c_indices
                        r_indices = torch.arange(S_ij.shape[0],device=device).unsqueeze(1)  # (B_q,1)
                        c_indices = torch.arange(S_ij.shape[1],device=device).unsqueeze(0)  # (1,B_k)
                        # mask所有global_row > global_col的位置, 即global_row > global_col的位置为False, 其余为True
                        mask = r_indices < (c_indices + offset)  # (B_q,B_k),bool
                        S_ij = torch.masked_fill(S_ij, mask, float("-inf"))  # (B_q,B_k)
                    m_i_new = torch.max(m_i_old, S_ij.max(dim=-1)[0])  # (B_q,)
                    P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))  # (B_q,B_k)
                    l_i = torch.exp(m_i_old - m_i_new) * l_i + torch.sum(P_ij,dim=-1)  # (B_q,)
                    O_i = torch.exp(m_i_old - m_i_new).unsqueeze(-1) * O_i + (P_ij @ V_j.to(torch.float32))  # (B_q,d)
                    m_i_old = m_i_new  # (B_q,)

                # 在存储回 O 和 L 之前，将 float32 累加器转换回 Q.dtype
                O[b,row_start:row_end,:] = (O_i / l_i.unsqueeze(1)).to(dtype)  # (B_q,d)
                L[b,row_start:row_end] = (m_i_new + torch.log(l_i)).to(dtype)  # (B_q,)
        
        ctx.save_for_backward(Q,K,V,O,L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        """
        Parameters:
            dO (Float[Tensor, "batchsize, N_q, d"]): the gradient of O with respect to loss function
            
        Returns:
            dQ (Float[Tensor, "batchsize, N_q, d"]): the gradient of Q with respect to loss function
            dK (Float[Tensor, "batchsize, N_k, d"]): the gradient of K with respect to loss function
            dV (Float[Tensor, "batchsize, N_k, d"]): the gradient of V with respect to loss function
            None (bool)
        """
        Q, K, V, O, L = ctx.saved_tensors
        # 将输入转换为 float32 以进行稳定的后向计算
        Q, K, V, O, L, dO = [t.to(torch.float32) for t in (Q, K, V, O, L, dO)]
        # dQ, dK, dV = flash_attention_bwd_compiled(Q, K, V, O, L, dO, ctx.is_causal)
        dQ, dK, dV = flash_attention_bwd(Q, K, V, O, L, dO, ctx.is_causal)
        return dQ.to(ctx.saved_tensors[0].dtype), dK.to(ctx.saved_tensors[1].dtype), dV.to(ctx.saved_tensors[2].dtype), None  # 返回值数量必须严格匹配 forward 的输入参数数量(包括非 Tensor 参数)