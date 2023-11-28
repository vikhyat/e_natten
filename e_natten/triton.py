import torch
import triton
import triton.language as tl

TILE_SIZE = 3

@triton.jit
def triton_window_start(i, seq_len, kernel_size):
    start = tl.maximum(i - kernel_size // 2, 0)
    if i + kernel_size // 2 >= seq_len:
        start += seq_len - i - kernel_size // 2 - 1
    return start

def get_window_start(i, seq_len, kernel_size):
    start = max(i - kernel_size // 2, 0)
    if i + kernel_size // 2 >= seq_len:
        start += seq_len - i - kernel_size // 2 - 1
    return start

def get_backward_window_start(i, kernel_size):
    if i < kernel_size:
        return 0
    else:
        return i - kernel_size // 2

def get_backward_window_end(i, seq_len, kernel_size):
    if i >= seq_len - kernel_size:
        return seq_len
    else:
        return i + kernel_size // 2 + 1

def tile_start(i, j, seq_len, tile_start, kernel_size):
    if i + j <= kernel_size // 2:
        return 0
    elif i + j >= seq_len - kernel_size // 2 - 1:
        return seq_len - tile_start - kernel_size
    else:
        return j

@triton.jit
def triton_softmax_jacobian(x, C: tl.constexpr):
    # First construct diagonal.
    rows = tl.arange(0, C)[:, None]
    cols = tl.arange(0, C)[None, :]
    diag = rows == cols
    diag = diag * x
    # Now outer product.
    y = tl.reshape(x, (C,))
    out_p = y[:, None] * y[None, :]
    return diag - out_p


def softmax_jacobian(x):
    B, H, T, C = x.shape
    jacobian = torch.zeros(B, H, C, C, dtype=x.dtype, device=x.device)
    for b in range(B):
        for h in range(H):  # Head dimension
            # Extract the softmax vector for this batch and head
            xi = x[b, h, 0, :]
            diag_p = torch.diag(xi)
            outer_p = xi.unsqueeze(-1) @ xi.unsqueeze(0)
            jacobian[b, h, :, :] = diag_p - outer_p
    return jacobian

@triton.jit
def _attn_fwd_1d(Q, K, V, kernel_size: tl.constexpr, Out,
                 stride_qb, stride_qn, stride_qt, stride_qc,
                 stride_kb, stride_kn, stride_kt, stride_kc,
                 stride_vb, stride_vn, stride_vt, stride_vc,
                 stride_ob, stride_on, stride_ot, stride_oc,
                 K_TILE_SIZE: tl.constexpr, B: tl.constexpr,
                 N: tl.constexpr, T: tl.constexpr, C: tl.constexpr):

    assert stride_kb == stride_qb and stride_vb == stride_qb and stride_ob == stride_qb
    assert stride_kn == stride_qn and stride_vn == stride_qn and stride_on == stride_qn

    bn_offset = tl.program_id(0)
    t_offset = tl.program_id(1)
    b_offset = bn_offset // N
    n_offset = bn_offset % N
    qkv_offset = b_offset.to(tl.int32) * stride_qb + n_offset.to(tl.int32) * stride_qn
    kv_start = triton_window_start(t_offset, T, kernel_size)
    
    # Load Q
    q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(T, C),
        strides=(stride_qt, stride_qc),
        offsets=(t_offset, 0),
        block_shape=(1, C),
        order=(1, 0)
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1))
    
    # Load K
    k_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(T, C),
        strides=(stride_kt, stride_kc),
        offsets=(kv_start, 0),
        block_shape=(K_TILE_SIZE, C),
        order=(1, 0)
    )
    k = tl.load(k_block_ptr, boundary_check=(0, 1))

    # Load V
    v_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(T, C),
        strides=(stride_vt, stride_vc),
        offsets=(kv_start, 0),
        block_shape=(K_TILE_SIZE, C),
        order=(1, 0)
    )
    v = tl.load(v_block_ptr, boundary_check=(0, 1))

    k_idx_range = tl.arange(0, K_TILE_SIZE)
    mask = k_idx_range < kernel_size

    # Compute QK^T.
    S_j = tl.sum(q * k, 1)

    # Compute softmax.
    S_j_minus_max = S_j - tl.max(S_j, axis=0) # Subtract max for numeric stability
    numerator = tl.where(mask, tl.math.exp2(S_j_minus_max), 0)
    denominator = tl.sum(numerator, axis=0)
    P_j = numerator / denominator

    # Compute output.
    o_j = tl.sum(P_j[:, None] * v, 0).to(Out.dtype.element_ty)

    # Save output.
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(T, C),
        strides=(stride_ot, stride_oc),
        offsets=(t_offset, 0),
        block_shape=(1, C),
        order=(1, 0)
    )
    tl.store(O_block_ptr, o_j[None, :], boundary_check=(0, 1))

@triton.jit
def _attn_bwd_1d(Q, K, V, kernel_size: tl.constexpr, dO, dQ, dK, dV,
                    stride_qb, stride_qn, stride_qt, stride_qc,
                    stride_kb, stride_kn, stride_kt, stride_kc,
                    stride_vb, stride_vn, stride_vt, stride_vc,
                    stride_dob, stride_don, stride_dot, stride_doc,
                    stride_dqb, stride_dqn, stride_dqt, stride_dqc,
                    stride_dkb, stride_dkn, stride_dkt, stride_dkc,
                    stride_dvb, stride_dvn, stride_dvt, stride_dvc,
                    K_TILE_SIZE: tl.constexpr, B: tl.constexpr,
                    N: tl.constexpr, T: tl.constexpr, C: tl.constexpr):

    assert (stride_kb == stride_qb and stride_vb == stride_qb and stride_dob == stride_qb
            and stride_dqb == stride_qb and stride_dkb == stride_qb and stride_dvb == stride_qb)
    assert (stride_kn == stride_qn and stride_vn == stride_qn and stride_don == stride_qn
            and stride_dqn == stride_qn and stride_dkn == stride_qn and stride_dvn == stride_qn)

    bn_offset = tl.program_id(0)
    t_offset = tl.program_id(1)
    b_offset = bn_offset // N
    n_offset = bn_offset % N
    qkv_offset = b_offset.to(tl.int32) * stride_qb + n_offset.to(tl.int32) * stride_qn
    kv_start = triton_window_start(t_offset, T, kernel_size)
    
    # Load Q
    q_ptrs = Q + qkv_offset + t_offset * stride_qt + tl.arange(0, C) * stride_qc
    q = tl.load(q_ptrs)

    # Load dO
    do_ptrs = dO + qkv_offset + t_offset * stride_dot + tl.arange(0, C) * stride_doc
    do = tl.load(do_ptrs)
    
    # Load K
    k_ptrs = K + qkv_offset + kv_start * stride_kt + tl.arange(0, K_TILE_SIZE)[:, None] * stride_kt + tl.arange(0, C) * stride_kc
    k = tl.load(k_ptrs)

    # Load V
    v_ptrs = V + qkv_offset + kv_start * stride_vt + tl.arange(0, K_TILE_SIZE)[:, None] * stride_vt + tl.arange(0, C) * stride_vc
    v = tl.load(v_ptrs)

    k_idx_range = tl.arange(0, K_TILE_SIZE)
    mask = k_idx_range < kernel_size

    # Compute QK^T and dP_j.
    S_j = tl.sum(q * k, 1)
    dP_j = tl.sum(do * v, 1)

    # Compute softmax.
    S_j_minus_max = S_j - tl.max(S_j, axis=0) # Subtract max for numeric stability
    numerator = tl.where(mask, tl.math.exp2(S_j_minus_max), 0)
    denominator = tl.sum(numerator, axis=0)
    P_j = numerator / denominator

    # Now we need to compute the Jacobian of P_j and use it to get dS_j.
    Jac_P_j = triton_softmax_jacobian(P_j, K_TILE_SIZE)
    dS_j = tl.sum(Jac_P_j * dP_j, 1)
    dQ_j = tl.sum(dS_j[:, None] * k, 0)

    # Save dQ.
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ + qkv_offset,
        shape=(T, C),
        strides=(stride_dqt, stride_dqc),
        offsets=(t_offset, 0),
        block_shape=(1, C),
        order=(1, 0)
    )
    tl.store(dQ_block_ptr, dQ_j[None, :], boundary_check=(0, 1))

    # Update dK.
    dK_update = q * dS_j[:, None]
    dK_update = dK_update.to(dK.dtype.element_ty)
    dK_update_ptr = dK + qkv_offset + kv_start * stride_dkt + tl.arange(0, K_TILE_SIZE)[:, None] * stride_dkt + tl.arange(0, C) * stride_dkc
    tl.atomic_add(dK_update_ptr, dK_update)

    # Update dV.
    dV_update = P_j[:, None] * do
    dV_update = dV_update.to(dV.dtype.element_ty)
    dV_update_ptr = dV + qkv_offset + kv_start * stride_dvt + tl.arange(0, K_TILE_SIZE)[:, None] * stride_dvt + tl.arange(0, C) * stride_dvc
    tl.atomic_add(dV_update_ptr, dV_update)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["B", "N", "H", "W", "C", "kernel_size"],
)
@triton.jit
def _attn_fwd_2d(Q, K, V, kernel_size: tl.constexpr, Out, 
                 stride_qb, stride_qn, stride_qh, stride_qw, stride_qc,
                 stride_kb, stride_kn, stride_kh, stride_kw, stride_kc,
                 stride_vb, stride_vn, stride_vh, stride_vw, stride_vc,
                 stride_ob, stride_on, stride_oh, stride_ow, stride_oc,
                 K_TILE_SIZE: tl.constexpr, B: tl.constexpr, N: tl.constexpr,
                 H: tl.constexpr, W: tl.constexpr, C: tl.constexpr):

    assert stride_kb == stride_qb and stride_vb == stride_qb and stride_ob == stride_qb
    assert stride_kn == stride_qn and stride_vn == stride_qn and stride_on == stride_qn

    bn_offset = tl.program_id(0)
    h_offset = tl.program_id(1)
    w_offset = tl.program_id(2)
    b_offset = bn_offset // N
    n_offset = bn_offset % N
    qkv_offset = b_offset * stride_qb + n_offset * stride_qn
    kv_start_x = triton_window_start(h_offset, H, kernel_size)
    kv_start_y = triton_window_start(w_offset, W, kernel_size)

    # Load Q, K and V.
    q = tl.load(tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(H, W, C),
        strides=(stride_qh, stride_qw, stride_qc),
        offsets=(h_offset, w_offset, 0),
        block_shape=(1, 1, C),
        order=(2, 1, 0)
    ), boundary_check=(0, 1, 2))
    k = tl.load(tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(H, W, C),
        strides=(stride_kh, stride_kw, stride_kc),
        offsets=(kv_start_x, kv_start_y, 0),
        block_shape=(K_TILE_SIZE, K_TILE_SIZE, C),
        order=(2, 1, 0)
    ), boundary_check=(0, 1, 2))
    v = tl.load(tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(H, W, C),
        strides=(stride_vh, stride_vw, stride_vc),
        offsets=(kv_start_x, kv_start_y, 0),
        block_shape=(K_TILE_SIZE, K_TILE_SIZE, C),
        order=(2, 1, 0)
    ), boundary_check=(0, 1, 2))

    mask = tl.arange(0, K_TILE_SIZE)
    mask = tl.where(mask < kernel_size, 1, 0)
    mask = mask[:, None] + mask[None, :]
    mask = tl.where(mask > 1, 1, 0)
    mask2 = tl.reshape(mask, (1, K_TILE_SIZE**2))

    # Compute QK^T.
    q = tl.reshape(q, (1, C))
    k = tl.reshape(k * mask[:, :, None], (K_TILE_SIZE**2, C))
    S_ij = tl.sum(q * k, 1)

    # Compute softmax.
    S_ij_minus_max = S_ij - tl.max(S_ij, axis=0) # Subtract max for numeric stability
    numerator = tl.math.exp2(S_ij_minus_max) * mask2.to(S_ij.dtype)
    denominator = tl.sum(numerator, axis=1)
    P_ij = numerator / denominator
    
    # Compute output.
    P_ij = tl.reshape(P_ij, (K_TILE_SIZE**2, 1))
    v = tl.reshape(v * mask[:, :, None], (K_TILE_SIZE**2, C))
    o_ij = tl.sum(P_ij * v, 0).to(Out.dtype.element_ty)

    # Save output.
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(H, W, C),
        strides=(stride_oh, stride_ow, stride_oc),
        offsets=(h_offset, w_offset, 0),
        block_shape=(1, 1, C),
        order=(2, 1, 0)
    )
    tl.store(O_block_ptr, o_ij[None, None, :], boundary_check=(0, 1, 2))

@triton.jit
def _attn_bwd_2d(Q, K, V, kernel_size: tl.constexpr, dO, dQ, dK, dV,
                    stride_qb, stride_qn, stride_qh, stride_qw, stride_qc,
                    stride_kb, stride_kn, stride_kh, stride_kw, stride_kc,
                    stride_vb, stride_vn, stride_vh, stride_vw, stride_vc,
                    stride_dob, stride_don, stride_doh, stride_dow, stride_doc,
                    stride_dqb, stride_dqn, stride_dqh, stride_dqw, stride_dqc,
                    stride_dkb, stride_dkn, stride_dkh, stride_dkw, stride_dkc,
                    stride_dvb, stride_dvn, stride_dvh, stride_dvw, stride_dvc,
                    K_TILE_SIZE: tl.constexpr, B: tl.constexpr, N: tl.constexpr,
                    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr):
    
    assert (stride_kb == stride_qb and stride_vb == stride_qb and stride_dob == stride_qb
            and stride_dqb == stride_qb and stride_dkb == stride_qb and stride_dvb == stride_qb)
    assert (stride_kn == stride_qn and stride_vn == stride_qn and stride_don == stride_qn
            and stride_dqn == stride_qn and stride_dkn == stride_qn and stride_dvn == stride_qn)
    
    bn_offset = tl.program_id(0)
    hw_offset = tl.program_id(1)
    b_offset = bn_offset // N
    n_offset = bn_offset % N
    h_offset = hw_offset // W
    w_offset = hw_offset % W
    qkv_offset = b_offset * stride_qb + n_offset * stride_qn
    kv_start_x = triton_window_start(h_offset, H, kernel_size)
    kv_start_y = triton_window_start(w_offset, W, kernel_size)

    # Load Q
    q = tl.load(tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(H, W, C),
        strides=(stride_qh, stride_qw, stride_qc),
        offsets=(h_offset, w_offset, 0),
        block_shape=(1, 1, C),
        order=(2, 1, 0)
    ), boundary_check=(0, 1, 2))
    q = tl.reshape(q, (1, C))

    # Load dO
    do = tl.load(tl.make_block_ptr(
        base=dO + qkv_offset,
        shape=(H, W, C),
        strides=(stride_doh, stride_dow, stride_doc),
        offsets=(h_offset, w_offset, 0),
        block_shape=(1, 1, C),
        order=(2, 1, 0)
    ), boundary_check=(0, 1, 2))
    do = tl.reshape(do, (1, C))
    
    k = tl.load(tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(H, W, C),
        strides=(stride_kh, stride_kw, stride_kc),
        offsets=(kv_start_x, kv_start_y, 0),
        block_shape=(K_TILE_SIZE, K_TILE_SIZE, C),
        order=(2, 1, 0)
    ))
    v = tl.load(tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(H, W, C),
        strides=(stride_vh, stride_vw, stride_vc),
        offsets=(kv_start_x, kv_start_y, 0),
        block_shape=(K_TILE_SIZE, K_TILE_SIZE, C),
        order=(2, 1, 0)
    ))

    mask = tl.arange(0, K_TILE_SIZE)
    mask = tl.where(mask < kernel_size, 1, 0)
    mask = mask[:, None] + mask[None, :]
    mask = tl.where(mask > 1, 1, 0)
    mask2 = tl.reshape(mask, (1, K_TILE_SIZE**2,))

    # Compute QK^T and dP_ij.
    k = tl.reshape(k * mask[:, :, None], (K_TILE_SIZE**2, C))
    S_ij = tl.sum(q * k, 1)
    v = tl.reshape(v * mask[:, :, None], (K_TILE_SIZE**2, C))
    dP_ij = tl.sum(do * v, 1)

    # Compute softmax.
    S_ij_minus_max = S_ij - tl.max(S_ij, axis=0) # Subtract max for numeric stability
    numerator = tl.math.exp2(S_ij_minus_max) * mask2.to(S_ij.dtype)
    denominator = tl.sum(numerator, axis=1)
    P_ij = numerator / denominator
    
    # Now we compute the Jacobian of P_j and use it to get dS_j.
    Jac_P_ij = triton_softmax_jacobian(P_ij, K_TILE_SIZE**2)
    dS_ij = tl.sum(Jac_P_ij * dP_ij, 1)
    dS_ij = tl.reshape(dS_ij, (K_TILE_SIZE**2, 1))
    dQ_ij = tl.sum(dS_ij * k, 0)
    
    # Save dQ.
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ + qkv_offset,
        shape=(H, W, C),
        strides=(stride_dqh, stride_dqw, stride_dqc),
        offsets=(h_offset, w_offset, 0),
        block_shape=(1, 1, C),
        order=(2, 1, 0)
    )
    tl.store(dQ_block_ptr, dQ_ij[None, None, :], boundary_check=(0, 1, 2))
    
    # Update dK.
    dK_update = q * dS_ij[:, None]
    dK_update = dK_update.to(dK.dtype.element_ty)
    dK_update = tl.reshape(dK_update, (K_TILE_SIZE, K_TILE_SIZE, C))
    dK_update_ptr = dK + qkv_offset + kv_start_x * stride_dkh + kv_start_y * stride_dkw + tl.arange(0, K_TILE_SIZE)[:, None, None] * stride_dkh + tl.arange(0, K_TILE_SIZE)[:, None] * stride_dkw + tl.arange(0, C) * stride_dkc
    tl.atomic_add(dK_update_ptr, dK_update)

    # Update dV.
    P_ij = tl.reshape(P_ij, (K_TILE_SIZE, K_TILE_SIZE, 1))
    dV_update = P_ij * do[None, :]
    dV_update = dV_update.to(dV.dtype.element_ty)
    dV_update_ptr = dV + qkv_offset + kv_start_x * stride_dvh + kv_start_y * stride_dvw + tl.arange(0, K_TILE_SIZE)[:, None, None] * stride_dvh + tl.arange(0, K_TILE_SIZE)[:, None] * stride_dvw + tl.arange(0, C) * stride_dvc
    tl.atomic_add(dV_update_ptr, dV_update)

class Natten1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kernel_size=7):
        """
        Compute 1D attention using Triton kernels.
        
        Args:
            q: Query tensor of shape (B, H, T, C).
            k: Key tensor of shape (B, H, T, C).
            v: Value tensor of shape (B, H, T, C).
            softmax_scale: Softmax scale.
            kernel_size: Kernel size.
        """
        B, N, T, C = q.shape
        o = torch.zeros_like(q)

        # Calculate next highest power of two greater than kernel_size
        k_tile_size = 2 ** (kernel_size - 1).bit_length()

        grid = (B*N, T)
        _attn_fwd_1d[grid](q, k, v, kernel_size, o,
                            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                            k_tile_size, B, N, T, C)
        
        ctx.save_for_backward(q, k, v, o)
        ctx.kernel_size = kernel_size

        return o

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, _ = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        B, N, T, C = Q.shape

        dO = grad_output

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # Calculate next highest power of two greater than kernel_size
        k_tile_size = 2 ** (kernel_size - 1).bit_length()

        grid = (B*N, T)
        _attn_bwd_1d[grid](Q, K, V, kernel_size, dO, dQ, dK, dV,
                            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
                            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
                            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
                            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
                            k_tile_size, B, N, T, C)
        
        return dQ, dK, dV, None

natten1d = Natten1d.apply

class Natten2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, kernel_size=7):
        """
        Compute 1D attention using Triton kernels.
        
        Args:
            q: Query tensor of shape (B, H, T, C).
            k: Key tensor of shape (B, H, T, C).
            v: Value tensor of shape (B, H, T, C).
            softmax_scale: Softmax scale.
            kernel_size: Kernel size.
        """
        B, N, H, W, C = Q.shape
        
        o = torch.zeros_like(Q)

        # Calculate next highest power of two greater than kernel_size
        k_tile_size = 2 ** (kernel_size - 1).bit_length()

        grid = (B*N, H, W)
        _attn_fwd_2d[grid](Q, K, V, kernel_size, o,
                            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), Q.stride(4),
                            K.stride(0), K.stride(1), K.stride(2), K.stride(3), K.stride(4),
                            V.stride(0), V.stride(1), V.stride(2), V.stride(3), V.stride(4),
                            o.stride(0), o.stride(1), o.stride(2), o.stride(3), o.stride(4),
                            k_tile_size, B, N, H, W, C)

        ctx.save_for_backward(Q, K, V, o)
        ctx.kernel_size = kernel_size

        return o

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, _ = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        B, N, H, W, C = Q.shape

        dO = grad_output

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # Calculate next highest power of two greater than kernel_size
        k_tile_size = 2 ** (kernel_size - 1).bit_length()

        grid = (B*N, H*W)
        _attn_bwd_2d[grid](Q, K, V, kernel_size, dO, dQ, dK, dV,
                            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), Q.stride(4),
                            K.stride(0), K.stride(1), K.stride(2), K.stride(3), K.stride(4),
                            V.stride(0), V.stride(1), V.stride(2), V.stride(3), V.stride(4),
                            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3), dO.stride(4),
                            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3), dQ.stride(4),
                            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3), dK.stride(4),
                            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3), dV.stride(4),
                            k_tile_size, B, N, H, W, C)
        
        return dQ, dK, dV, None

natten2d = Natten2d.apply