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
                 K_TILE_SIZE: tl.constexpr,
                 B: tl.constexpr, N: tl.constexpr, T: tl.constexpr, C: tl.constexpr):

    assert stride_kb == stride_qb and stride_vb == stride_qb and stride_ob == stride_qb
    assert stride_kn == stride_qn and stride_vn == stride_qn and stride_on == stride_qn

    bn_offset = tl.program_id(0)
    t_offset = tl.program_id(1)
    b_offset = bn_offset // N
    n_offset = bn_offset % N
    qkv_offset = b_offset.to(tl.int32) * stride_qb + n_offset.to(tl.int32) * stride_qn
    kv_start = triton_window_start(t_offset, T, kernel_size)
    
    # Load Q
    q_ptrs = Q + qkv_offset + t_offset * stride_qt + tl.arange(0, C) * stride_qc
    q = tl.load(q_ptrs)
    
    # Load K
    k_ptrs = K + qkv_offset + kv_start * stride_kt + tl.arange(0, K_TILE_SIZE)[:, None] * stride_kt + tl.arange(0, C) * stride_kc
    k = tl.load(k_ptrs)

    # Load V
    v_ptrs = V + qkv_offset + kv_start * stride_vt + tl.arange(0, K_TILE_SIZE)[:, None] * stride_vt + tl.arange(0, C) * stride_vc
    v = tl.load(v_ptrs)

    k_idx_range = tl.arange(0, K_TILE_SIZE)
    mask = k_idx_range < kernel_size

    # Compute QK^T.
    S_j = tl.sum(q * k, 1)

    # Compute softmax.
    S_j_minus_max = S_j - tl.max(S_j, axis=0) # Subtract max for numeric stability
    numerator = tl.where(mask, tl.exp(S_j_minus_max), 0)
    denominator = tl.sum(numerator, axis=0)
    P_j = numerator / denominator

    # Compute output.
    o_j = tl.sum(P_j[:, None] * v, 0)

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
    hw_offset = tl.program_id(1)
    b_offset = bn_offset // N
    n_offset = bn_offset % N
    h_offset = hw_offset // W
    w_offset = hw_offset % W
    qkv_offset = b_offset * stride_qb + n_offset * stride_qn
    kv_start_x = triton_window_start(h_offset, H, kernel_size)
    kv_start_y = triton_window_start(w_offset, W, kernel_size)

    # Load Q
    q_ptrs = Q + qkv_offset + h_offset * stride_qh + w_offset * stride_qw + tl.arange(0, C) * stride_qc
    q = tl.load(q_ptrs)
    
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

    # Compute QK^T.
    k = tl.reshape(k * mask[:, :, None], (K_TILE_SIZE**2, C))
    S_ij = tl.sum(q * k, 1)


    # Compute softmax.
    S_ij_minus_max = S_ij - tl.max(S_ij, axis=0) # Subtract max for numeric stability
    numerator = tl.exp(S_ij_minus_max) * mask2.to(S_ij.dtype)
    denominator = tl.sum(numerator, axis=1)
    P_ij = numerator / denominator
    
    # Compute output.
    P_ij = tl.reshape(P_ij, (K_TILE_SIZE**2, 1))
    v = tl.reshape(v * mask[:, :, None], (K_TILE_SIZE**2, C))
    o_ij = tl.sum(P_ij * v, 0)

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

        # dQ has the same neighborhood pattern, but dK and dV need us to invert the
        # neighborhood mapping, which is not symmetric. 

        dQ = torch.zeros(B, N, T, C, dtype=Q.dtype, device=Q.device)
        
        # Split dO into tiles of size Q_TILE_SIZE. For each q tile, we load corresponding k and v tiles
        # of size Q_TILE_SIZE + kernel_size - 1. 
        k_tile_size = TILE_SIZE + kernel_size - 1
        for i in range(0, T, TILE_SIZE):
            dO_tile = dO[:, :, i:i+TILE_SIZE, :]
            Q_tile = Q[:, :, i:i+TILE_SIZE, :]

            # Load k tile.
            kv_start = get_window_start(i, T, kernel_size)
            K_tile = K[:, :, kv_start:kv_start+k_tile_size, :]
            V_tile = V[:, :, kv_start:kv_start+k_tile_size, :]

            # Process each element in the query tile.
            iter_max = min(TILE_SIZE, Q_tile.shape[2])
            for j in range(0, iter_max):
                if i + j <= kernel_size // 2:
                    j2 = 0
                elif i + j >= T - kernel_size // 2 - 1:
                    j2 = T - kv_start - kernel_size
                else:
                    j2 = j

                S_j = Q_tile[:, :, j:j+1, :] @ K_tile[:, :, j2:j2+kernel_size, :].transpose(-1, -2)
                P_j = torch.softmax(S_j, dim=-1)
                P_j_jacobian = softmax_jacobian(P_j)
                dP_j = dO_tile[:, :, j:j+1, :] @ V_tile[:, :, j2:j2+kernel_size, :].transpose(-1, -2)
                dS_j = torch.matmul(P_j_jacobian, dP_j.transpose(-1, -2)).transpose(-1, -2)
                dQ_j = dS_j @ K_tile[:, :, j2:j2+kernel_size, :]
                dQ[:, :, i+j:i+j+1, :] = dQ_j


        # Just do the naive thing for dK and dV. Can add tiling later.
        dK = torch.zeros(B, N, T, C, dtype=Q.dtype, device=Q.device)
        dV = torch.zeros(B, N, T, C, dtype=Q.dtype, device=Q.device)
        for i in range(T):
            ni = get_backward_window_start(i, kernel_size)
            ne = get_backward_window_end(i, T, kernel_size)
            for xi in range(ni, ne):
                oni = get_window_start(xi, T, kernel_size)
                si = i - oni

                Q_slice = Q[:, :, xi, :]
                S_xi = Q_slice.unsqueeze(2) @ K[:, :, oni:oni+kernel_size, :].transpose(-1, -2)
                P_xi = torch.softmax(S_xi, dim=-1)
                P_xi_jacobian = softmax_jacobian(P_xi)
                dP_xi = dO[:, :, xi:xi+1, :] @ V[:, :, oni:oni+kernel_size, :].transpose(-1, -2)
                dS_xi = torch.matmul(P_xi_jacobian, dP_xi.transpose(-1, -2)).transpose(-1, -2)
                dS_slice = dS_xi[:, :, 0, si].unsqueeze(-1)

                dK[:, :, i, :] += Q_slice * dS_slice
                dV[:, :, i, :] += P_xi[:, :, 0, si].unsqueeze(-1) * dO[:, :, xi, :]

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

        grid = (B*N, H*W)
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

        # dQ has the same neighborhood pattern, but dK and dV need us to invert the
        # neighborhood mapping, which is not symmetric. 

        dQ = torch.zeros_like(Q)
        
        # Split dO into tiles of size Q_TILE_SIZE. For each q tile, we load corresponding k and v tiles
        # of size Q_TILE_SIZE + kernel_size - 1. 
        k_tile_size = TILE_SIZE + kernel_size - 1

        for x in range(0, H, TILE_SIZE):
            for y in range(0, W, TILE_SIZE):
                dO_tile = dO[:, :, x:x+TILE_SIZE, y:y+TILE_SIZE, :]
                Q_tile = Q[:, :, x:x+TILE_SIZE, y:y+TILE_SIZE, :]

                # Load K and V tiles.
                x_kv_start = get_window_start(x, H, kernel_size)
                y_kv_start = get_window_start(y, W, kernel_size)
                K_tile = K[:, :, x_kv_start:x_kv_start+k_tile_size, y_kv_start:y_kv_start+k_tile_size, :]
                V_tile = V[:, :, x_kv_start:x_kv_start+k_tile_size, y_kv_start:y_kv_start+k_tile_size, :]

                # Process each element in the query tile.
                x_iter_max = min(TILE_SIZE, Q_tile.shape[2])
                y_iter_max = min(TILE_SIZE, Q_tile.shape[3])
                for xi in range(0, x_iter_max):
                    for yi in range(0, y_iter_max):
                        xj = tile_start(x, xi, H, x_kv_start, kernel_size)
                        yj = tile_start(y, yi, W, y_kv_start, kernel_size)
                        Q_ij = Q_tile[:, :, xi:xi+1, yi:yi+1, :].view(B, N, 1, C)
                        K_ij = K_tile[:, :, xj:xj+kernel_size, yj:yj+kernel_size, :].reshape(B, N, kernel_size**2, C)
                        S_j = Q_ij @ K_ij.transpose(-1, -2)
                        P_j = torch.softmax(S_j, dim=-1)
                        P_j_jacobian = softmax_jacobian(P_j)
                        dO_ij = dO_tile[:, :, xi:xi+1, yi:yi+1, :].view(B, N, 1, C)
                        V_ij = V_tile[:, :, xj:xj+kernel_size, yj:yj+kernel_size, :].reshape(B, N, kernel_size**2, C)
                        dP_j = dO_ij @ V_ij.transpose(-1, -2)
                        dS_j = torch.matmul(P_j_jacobian, dP_j.transpose(-1, -2)).transpose(-1, -2)
                        dQ_j = dS_j @ K_ij
                        dQ[:, :, x+xi:x+xi+1, y+yi:y+yi+1, :] = dQ_j.view(B, N, 1, 1, C)

        # Just do the naive thing for dK and dV. Can add tiling later.
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        for x in range(H):
            xni = get_backward_window_start(x, kernel_size)
            xne = get_backward_window_end(x, H, kernel_size)
            for y in range(W):
                yni = get_backward_window_start(y, kernel_size)
                yne = get_backward_window_end(y, W, kernel_size)
                for xi in range(xni, xne):
                    xon = get_window_start(xi, H, kernel_size)
                    for yi in range(yni, yne):
                        yon = get_window_start(yi, W, kernel_size)
                        xsi = x - xon
                        ysi = y - yon

                        Q_slice = Q[:, :, xi, yi, :]
                        K_xiyi = K[:, :, xon:xon+kernel_size, yon:yon+kernel_size, :].reshape(B, N, kernel_size**2, C)
                        V_xiyi = V[:, :, xon:xon+kernel_size, yon:yon+kernel_size, :].reshape(B, N, kernel_size**2, C)
                        S_xi = Q_slice.unsqueeze(2) @ K_xiyi.transpose(-1, -2)
                        P_xi = torch.softmax(S_xi, dim=-1)
                        P_xi_jacobian = softmax_jacobian(P_xi)
                        dP_xi = dO[:, :, xi, yi, :].unsqueeze(-2) @ V_xiyi.transpose(-1, -2)
                        dS_xi = torch.matmul(P_xi_jacobian, dP_xi.transpose(-1, -2)).transpose(-1, -2)

                        dS_xi = dS_xi.reshape(B, N, kernel_size, kernel_size)
                        dS_slice = dS_xi[:, :, xsi, ysi].unsqueeze(-1)
                        dK[:, :, x, y, :] += Q_slice * dS_slice

                        P_xi = P_xi.reshape(B, N, kernel_size, kernel_size)
                        dV[:, :, x, y, :] += P_xi[:, :, xsi, ysi].unsqueeze(-1) * dO[:, :, xi, yi, :]

        return dQ, dK, dV, None

natten2d = Natten2d.apply