import torch

Q_TILE_SIZE = 3

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
        B, H, T, C = q.shape
        
        # Split q into tiles of size Q_TILE_SIZE. For each q tile, we load corresponding k and v tiles
        # of size Q_TILE_SIZE + kernel_size - 1. The output of each tile is stored in o.
        o = torch.zeros(B, H, 0, C)
        k_tile_size = Q_TILE_SIZE + kernel_size - 1
        _p = torch.zeros(B, H, 0, kernel_size)

        for i in range(0, T, Q_TILE_SIZE):
            Q_tile = q[:, :, i:i+Q_TILE_SIZE, :]
            kv_start = get_window_start(i, T, kernel_size)
            
            # Load k tile and allocate space for s. Load v tile as well to avoid stalling later.
            K_tile = k[:, :, kv_start:kv_start+k_tile_size, :]
            V_tile = v[:, :, kv_start:kv_start+k_tile_size, :]
            P_tile = torch.zeros((B, H, 0, kernel_size))
            
            # For each element in the query tile, compute QK^T using the relevant slice of the key tile.
            iter_max = min(Q_TILE_SIZE, Q_tile.shape[2])
            for j in range(0, iter_max):
                if i <= kernel_size // 2:
                    j2 = 0
                elif i + j >= T - kernel_size // 2 - 1:
                    j2 = T - kv_start - kernel_size
                else:
                    j2 = j

                S_j = Q_tile[:, :, j:j+1, :] @ K_tile[:, :, j2:j2+kernel_size, :].transpose(-1, -2)
                P_j = torch.softmax(S_j, dim=-1)
                P_tile = torch.cat((P_tile, P_j), dim=2)
                _p = torch.cat((_p, P_j), dim=2)

            for j in range(0, iter_max):
                if i <= kernel_size // 2:
                    j2 = 0
                elif i + j >= T - kernel_size // 2 - 1:
                    j2 = T - kv_start - kernel_size
                else:
                    j2 = j
                o_j = P_tile[:, :, j:j+1, :] @ V_tile[:, :, j2:j2+kernel_size, :]
                o = torch.cat((o, o_j), dim=2)

        ctx.save_for_backward(q, k, v, _p)
        ctx.kernel_size = kernel_size

        return _p

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, v, O = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        B, H, T, C = Q.shape

        dO = grad_output

        # Precompute to assist with calculating dsoftmax later.
        D = torch.sum(dO * O, dim=-1)

        dP = grad_output
        
        # dQ has the same neighborhood pattern, but dK and dV need us to invert the
        # neighborhood mapping, which is not symmetric. 

        dQ = torch.zeros(B, H, 0, C)
        
        # Split dO into tiles of size Q_TILE_SIZE. For each q tile, we load corresponding k and v tiles
        # of size Q_TILE_SIZE + kernel_size - 1. 
        k_tile_size = Q_TILE_SIZE + kernel_size - 1
        for i in range(0, T, Q_TILE_SIZE):
            dP_tile = dP[:, :, i:i+Q_TILE_SIZE, :]
            Q_tile = Q[:, :, i:i+Q_TILE_SIZE, :]

            # Load k tile.
            kv_start = get_window_start(i, T, kernel_size)
            K_tile = K[:, :, kv_start:kv_start+k_tile_size, :]

            # Process each element in the query tile.
            iter_max = min(Q_TILE_SIZE, dP_tile.shape[2])
            for j in range(0, iter_max):
                if i <= kernel_size // 2:
                    j2 = 0
                elif i + j >= T - kernel_size // 2 - 1:
                    j2 = T - kv_start - kernel_size
                else:
                    j2 = j

                P_j = Q_tile[:, :, j:j+1, :] @ K_tile[:, :, j2:j2+kernel_size, :].transpose(-1, -2)
                dP_j = dP_tile[:, :, j:j+1, :]
                dS_j = P_j * (dP_j - D[:, :, j:j+1].unsqueeze(2))
                dQ_j = dS_j @ K_tile[:, :, j2:j2+kernel_size, :]
                dQ = torch.cat((dQ, dQ_j), dim=2)


        # Just do the naive thing for dK and dV. Can add tiling later.
        dK = torch.zeros(B, H, T, C)
        for i in range(T):
            ni = get_backward_window_start(i, kernel_size)
            ne = get_backward_window_end(i, T, kernel_size)
            for xi in range(ni, ne):
                oni = get_window_start(xi, T, kernel_size)
                si = oni - i
                Q_slice = Q[:, :, xi, :]
                P_xi = Q_slice.unsqueeze(2) @ K[:, :, oni:oni+kernel_size, :].transpose(-1, -2)
                dP_xi = dP[:, :, xi:xi+1, :]
                dS_xi = P_xi * (dP_xi - D[:, :, xi:xi+1].unsqueeze(-1))
                dS_slice = dS_xi[:, :, 0, si].unsqueeze(-1)
                dK[:, :, i, :] += Q_slice * dS_slice

        return dQ, dK, dQ, None

natten1d = Natten1d.apply