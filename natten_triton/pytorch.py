# Reference implementation of the algorithms in PyTorch (for easier debugging)

import torch

TILE_SIZE = 3

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
    jacobian = torch.zeros(B, H, C, C, dtype=x.dtype)
    for b in range(B):
        for h in range(H):  # Head dimension
            # Extract the softmax vector for this batch and head
            xi = x[b, h, 0, :]
            diag_p = torch.diag(xi)
            outer_p = xi.unsqueeze(-1) @ xi.unsqueeze(0)
            jacobian[b, h, :, :] = diag_p - outer_p
    return jacobian


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
        
        # Split q into tiles of size Q_TILE_SIZE. For each q tile, we load corresponding k and v tiles
        # of size Q_TILE_SIZE + kernel_size - 1. The output of each tile is stored in o.
        o = torch.zeros(B, N, T, C)
        k_tile_size = TILE_SIZE + kernel_size - 1

        for i in range(0, T, TILE_SIZE):
            Q_tile = q[:, :, i:i+TILE_SIZE, :]
            kv_start = get_window_start(i, T, kernel_size)
            iter_max = min(TILE_SIZE, Q_tile.shape[2])
            
            # Load K and V tiles.
            K_tile = k[:, :, kv_start:kv_start+k_tile_size, :]
            V_tile = v[:, :, kv_start:kv_start+k_tile_size, :]
            
            # For each element in the query tile, compute QK^T using the relevant slice of the key tile.
            for j in range(0, iter_max):
                j2 = tile_start(i, j, T, kv_start, kernel_size)
                S_j = Q_tile[:, :, j:j+1, :] @ K_tile[:, :, j2:j2+kernel_size, :].transpose(-1, -2)
                P_j = torch.softmax(S_j, dim=-1)
                o_j = P_j @ V_tile[:, :, j2:j2+kernel_size, :]
                o[:, :, i+j:i+j+1, :] = o_j


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

        dQ = torch.zeros(B, N, T, C)
        
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
        dK = torch.zeros(B, N, T, C)
        dV = torch.zeros(B, N, T, C)
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
    def forward(ctx, q, K, v, kernel_size=7):
        """
        Compute 1D attention using Triton kernels.
        
        Args:
            q: Query tensor of shape (B, H, T, C).
            k: Key tensor of shape (B, H, T, C).
            v: Value tensor of shape (B, H, T, C).
            softmax_scale: Softmax scale.
            kernel_size: Kernel size.
        """
        B, N, H, W, C = q.shape
        
        # Split q into tiles of size Q_TILE_SIZE. For each q tile, we load corresponding k and v tiles
        # of size Q_TILE_SIZE + kernel_size - 1. The output of each tile is stored in o.
        o = torch.zeros(B, N, H, W, C)
        k_tile_size = TILE_SIZE + kernel_size - 1

        for x in range(0, H, TILE_SIZE):
            for y in range(0, W, TILE_SIZE):
                Q_tile = q[:, :, x:x+TILE_SIZE, y:y+TILE_SIZE, :]

                x_kv_start = get_window_start(x, H, kernel_size)
                y_kv_start = get_window_start(y, W, kernel_size)
                x_iter_max = min(TILE_SIZE, Q_tile.shape[2])
                y_iter_max = min(TILE_SIZE, Q_tile.shape[3])
                
                # Load KV tiles and allocate space for P.
                K_tile = K[:, :, x_kv_start:x_kv_start+k_tile_size, y_kv_start:y_kv_start+k_tile_size, :]
                V_tile = v[:, :, x_kv_start:x_kv_start+k_tile_size, y_kv_start:y_kv_start+k_tile_size, :]
                
                # For each element in the query tile, compute QK^T using the relevant slice of the key tile.
                for xi in range(0, x_iter_max):
                    for yi in range(0, y_iter_max):
                        xj = tile_start(x, xi, H, x_kv_start, kernel_size)
                        yj = tile_start(y, yi, W, y_kv_start, kernel_size)
                        Q_ij = Q_tile[:, :, xi:xi+1, yi:yi+1, :].view(B, N, 1, C)
                        K_ij = K_tile[:, :, xj:xj+kernel_size, yj:yj+kernel_size, :].reshape(B, N, kernel_size**2, C)
                        V_ij = V_tile[:, :, xj:xj+kernel_size, yj:yj+kernel_size, :].reshape(B, N, kernel_size**2, C)
                        S_ij = Q_ij @ K_ij.transpose(-1, -2)
                        P_ij = torch.softmax(S_ij, dim=-1)
                        o_ij = P_ij @ V_ij
                        o[:, :, x+xi:x+xi+1, y+yi:y+yi+1, :] = o_ij.view(B, N, 1, 1, C)

        ctx.save_for_backward(q, K, v, o)
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

        dQ = torch.zeros(B, N, H, W, C)
        
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
        dK = torch.zeros(B, N, H, W, C)
        dV = torch.zeros(B, N, H, W, C)
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