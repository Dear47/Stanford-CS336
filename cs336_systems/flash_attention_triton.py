import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}, num_warps=8, num_stages=5),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_warps=8, num_stages=5),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=8, num_stages=5),
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=8, num_stages=5)
    ],
    key=['N_QUERIES', 'N_KEYS', 'IS_CAUSAL'],
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL:tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    """
    K_block和V_block的offset不是(query_tile_index * K_TILE_SIZE, 0)!
    因为对于每一个Q_block, 需要迭代所有K_block和V_block, 如果K_block和V_block的初始偏移量设为(query_tile_index * K_TILE_SIZE, 0),
    这意味着Q的第query_tile_index块, 会与K和V的第query_tile_index块之后的所有块迭代, 丢失了query_tile_index之前的所有块
    因此K_block和V_block的初始偏移量设为(0, 0), 其在kernel的for循环内部通过.advance()来迭代
    """
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")  # (Q_TILE_SIZE,D)
    dtype = Q_i.dtype
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) # (Q_TILE_SIZE,D)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (Q_TILE_SIZE,)
    m_i_old = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  # (Q_TILE_SIZE,)
    m_i_new = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  # (Q_TILE_SIZE,)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")  # (K_TILE_SIZE,D)
        V_j = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")  # (K_TILE_SIZE,D)

        S_ij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)  # (Q_TILE_SIZE,K_TILE_SIZE)s
        S_ij = tl.dot(Q_i, tl.trans(K_j), acc=S_ij) * scale  # (Q_TILE_SIZE,K_TILE_SIZE)

        if IS_CAUSAL:
            offset = key_tile_index * K_TILE_SIZE - query_tile_index * Q_TILE_SIZE
            r_indices = tl.arange(0, Q_TILE_SIZE)[:,None]  # (Q_TILE_SIZE,1)
            c_indices = tl.arange(0, K_TILE_SIZE)[None,:]  # (1,K_TILE_SIZE)
            mask = r_indices < (c_indices + offset)  # (Q_TILE_SIZE,K_TILE_SIZE)
            S_ij = tl.where(mask, -1e6, S_ij)  # (Q_TILE_SIZE,K_TILE_SIZE)  tl中没有直接等价于torch.masked_fill的函数, 但可以用tl.where实现
        
        rowmax = tl.max(S_ij, axis=1)  # (Q_TILE_SIZE,)
        m_i_new = tl.maximum(m_i_old, rowmax)  # (Q_TILE_SIZE,)
        P_ij = tl.exp(S_ij - m_i_new[:, None])  # (Q_TILE_SIZE,K_TILE_SIZE)
        l_i = tl.exp(m_i_old - m_i_new) * l_i + tl.sum(P_ij, axis=1)  # (Q_TILE_SIZE,)
        O_i = tl.exp(m_i_old - m_i_new)[:, None] * O_i + tl.dot(P_ij.to(V_j.dtype), V_j).to(tl.float32)  # (Q_TILE_SIZE,D)
        m_i_old = m_i_new  # (Q_TILE_SIZE,)

        K_block_ptr = tl.advance(K_block_ptr,(K_TILE_SIZE,0))
        V_block_ptr = tl.advance(V_block_ptr,(K_TILE_SIZE,0))

    O_i = (O_i / l_i[:, None]).to(dtype)  # (Q_TILE_SIZE,D)
    l_i = (m_i_new + tl.log(l_i)).to(dtype)  # (Q_TILE_SIZE,)
    tl.store(O_block_ptr, O_i, boundary_check=(0,1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))

@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    dO_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,    
    stride_dob, stride_doq, stride_dod,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL:tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(0, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (K_TILE_SIZE, D)
    V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32) # (K_TILE_SIZE, D)
    # 累加器用float32保证数值稳定性
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)  # (K_TILE_SIZE, D)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)  # (K_TILE_SIZE, D)

    for query_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        # 转换为 float32 以进行稳定的后向计算
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)  # (Q_TILE_SIZE,)
        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (Q_TILE_SIZE, D)
        O_i = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (Q_TILE_SIZE, D)
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (Q_TILE_SIZE, D)
        
        D_i = tl.sum(dO_i * O_i, axis=1)  # (Q_TILE_SIZE,)
        S_ij = (tl.dot(Q_i, tl.trans(K_j)) * scale)  # (Q_TILE_SIZE, K_TILE_SIZE)

        if IS_CAUSAL:
            offset = key_tile_index * K_TILE_SIZE - query_tile_index * Q_TILE_SIZE
            r_indices = tl.arange(0, Q_TILE_SIZE)[:,None]  # (Q_TILE_SIZE,1)
            c_indices = tl.arange(0, K_TILE_SIZE)[None,:]  # (1,K_TILE_SIZE)
            mask = r_indices < (c_indices + offset)  # (Q_TILE_SIZE,K_TILE_SIZE)
            S_ij = tl.where(mask, -1e6, S_ij)

        P_ij = tl.exp(S_ij - L_i[:,None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        dV_j += tl.dot(tl.trans(P_ij), dO_i)  # (K_TILE_SIZE, D)
        dP_ij = tl.dot(dO_i, tl.trans(V_j))  # (Q_TILE_SIZE, K_TILE_SIZE)
        dS_ij = P_ij * (dP_ij - D_i[:,None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        dK_j += tl.dot(tl.trans(dS_ij), Q_i) * scale  # # (K_TILE_SIZE, D)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dK_j.to(K_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_j.to(V_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    dO_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,    
    stride_dob, stride_doq, stride_dod,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL:tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)  # (Q_TILE_SIZE,)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (Q_TILE_SIZE, D)
    O_i = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (Q_TILE_SIZE, D)
    dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (Q_TILE_SIZE, D)
    # dQ_i_current = tl.load(dQ_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (Q_TILE_SIZE, D)
    dQ_i_current = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (Q_TILE_SIZE, D)
    D_i = tl.sum(dO_i * O_i, axis=1)  # (Q_TILE_SIZE,)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (K_TILE_SIZE, D)
        
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        if IS_CAUSAL:
            offset = key_tile_index * K_TILE_SIZE - query_tile_index * Q_TILE_SIZE
            r_indices = tl.arange(0, Q_TILE_SIZE)[:,None]  # (Q_TILE_SIZE,1)
            c_indices = tl.arange(0, K_TILE_SIZE)[None,:]  # (1,K_TILE_SIZE)
            mask = r_indices < (c_indices + offset)  # (Q_TILE_SIZE,K_TILE_SIZE)
            S_ij = tl.where(mask, -1e6, S_ij)

        P_ij = tl.exp(S_ij - L_i[:,None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        dP_ij = tl.dot(dO_i, tl.trans(V_j))  # (Q_TILE_SIZE, K_TILE_SIZE)
        dS_ij = P_ij * (dP_ij - D_i[:,None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        dQ_i_current += tl.dot(dS_ij, K_j) * scale  # (Q_TILE_SIZE, D)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_i_current.to(Q_block_ptr.type.element_ty), boundary_check=(0, 1))

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal:bool=False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Our pointer arithmetic will assume contiguous"
        batchsize, N_QUERIES, D = Q.shape
        dtype = Q.dtype
        device = Q.device
        N_KEYS = K.shape[-2]
        scale = D**(-0.5)

        # META['Q_TILE_SIZE'] 将由 Autotuner 自动填充
        grid = lambda META:(triton.cdiv(N_QUERIES, META['Q_TILE_SIZE']), batchsize)

        O = torch.empty_like(Q, device=device, dtype=dtype)
        L = torch.empty(batchsize, N_QUERIES, device=device, dtype=dtype)

        # 在内核调用中不显式注入 Q_TILE_SIZE 和 K_TILE_SIZE, 它们将由 @triton.autotune 自动注入
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS, scale,
            D, 
            IS_CAUSAL=is_causal
        )

        best_config = flash_fwd_kernel.best_config
        ctx.Q_TILE_SIZE = best_config.kwargs['Q_TILE_SIZE']
        ctx.K_TILE_SIZE = best_config.kwargs['K_TILE_SIZE']

        ctx.IS_CAUSAL = is_causal
        ctx.save_for_backward(Q,K,V,O,L)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        batchsize, N_QUERIES, D = Q.shape
        device = Q.device
        N_KEYS = K.shape[-2]
        scale = D**(-0.5)

        dkv_grid = lambda META: (triton.cdiv(N_KEYS, META['K_TILE_SIZE']), batchsize)
        dq_grid = lambda META: (triton.cdiv(N_QUERIES, META['Q_TILE_SIZE']), batchsize)

        dQ = torch.zeros(batchsize, N_QUERIES, D, device=device, dtype=Q.dtype)
        dK = torch.zeros(batchsize, N_KEYS, D, device=device, dtype=K.dtype)
        dV = torch.zeros(batchsize, N_KEYS, D, device=device, dtype=V.dtype)

        flash_bwd_dkv_kernel[dkv_grid](
            Q, K, V, O, L,
            dO, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            N_QUERIES, N_KEYS, scale,
            D, 
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            IS_CAUSAL=ctx.IS_CAUSAL
        )
        flash_bwd_dq_kernel[dq_grid](
            Q, K, V, O, L,
            dO, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            N_QUERIES, N_KEYS, scale,
            D, 
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            IS_CAUSAL=ctx.IS_CAUSAL
        )
        return dQ, dK, dV, None  # 返回值数量必须严格匹配 forward 的输入参数数量(包括非 Tensor 参数)