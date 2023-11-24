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
        _qk = torch.zeros(B, H, 0, kernel_size)

        for i in range(0, T, Q_TILE_SIZE):
            q_tile = q[:, :, i:i+Q_TILE_SIZE, :]
            kv_start = get_window_start(i, T, kernel_size)
            
            # Load k tile and allocate space for s. Load v tile as well to avoid stalling later.
            k_tile = k[:, :, kv_start:kv_start+k_tile_size, :]
            v_tile = v[:, :, kv_start:kv_start+k_tile_size, :]
            s = torch.zeros((B, H, 0, kernel_size))
            
            # For each element in the query tile, compute QK^T using the relevant slice of the key tile.
            iter_max = min(Q_TILE_SIZE, q_tile.shape[2])
            for j in range(0, iter_max):
                if i <= kernel_size // 2:
                    j2 = 0
                elif i + j >= T - kernel_size // 2 - 1:
                    j2 = T - kv_start - kernel_size
                else:
                    j2 = j

                s_j = q_tile[:, :, j:j+1, :] @ k_tile[:, :, j2:j2+kernel_size, :].transpose(-1, -2)
                s = torch.cat((s, s_j), dim=2)
                _qk = torch.cat((_qk, s_j), dim=2)
            
            # Compute softmax.
            p = torch.softmax(s, dim=-1)

            for j in range(0, iter_max):
                if i <= kernel_size // 2:
                    j2 = 0
                elif i + j >= T - kernel_size // 2 - 1:
                    j2 = T - kv_start - kernel_size
                else:
                    j2 = j
                o_j = p[:, :, j:j+1, :] @ v_tile[:, :, j2:j2+kernel_size, :]
                o = torch.cat((o, o_j), dim=2)

        ctx.save_for_backward(q, k, v, o)
        ctx.kernel_size = kernel_size

        return _qk

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        B, H, T, C = q.shape

        dS = grad_output
        
        # dQ has the same neighborhood pattern, but dK and dV need us to invert the
        # neighborhood mapping, which is not symmetric. Let's start with dQ since
        # it's easier -- we're basically going to take the same tiled approach as
        # the forward pass.

        dQ = torch.zeros(B, H, 0, C)
        dK = torch.zeros(B, H, 0, C)
        
        # Split dO into tiles of size Q_TILE_SIZE. For each q tile, we load corresponding k and v tiles
        # of size Q_TILE_SIZE + kernel_size - 1. 
        k_tile_size = Q_TILE_SIZE + kernel_size - 1
        for i in range(0, T, Q_TILE_SIZE):
            # Load k and v tiles.
            kv_start = get_window_start(i, T, kernel_size)
            k_tile = k[:, :, kv_start:kv_start+k_tile_size, :]
            v_tile = v[:, :, kv_start:kv_start+k_tile_size, :]

            # Load a little extra O and Q to account for the assymmetric neighborhood mapping.
            oq_start = get_backward_window_start(kv_start, kernel_size)
            oq_offset = oq_start - i
            oq_end = get_backward_window_end(kv_start + k_tile_size - 1, T, kernel_size)
            q_tile = q[:, :, oq_start:oq_end, :]
            dS_tile = dS[:, :, i:i+Q_TILE_SIZE, :]

            # Process each element in the query tile.
            iter_max = min(Q_TILE_SIZE, dS_tile.shape[2])
            for j in range(0, iter_max):
                if i <= kernel_size // 2:
                    j2 = 0
                elif i + j >= T - kernel_size // 2 - 1:
                    j2 = T - kv_start - kernel_size
                else:
                    j2 = j
                dQ_j = dS_tile[:, :, j:j+1, :] @ k_tile[:, :, j2:j2+kernel_size, :]
                dQ = torch.cat((dQ, dQ_j), dim=2)

        return dQ, dQ, dQ, None

natten1d = Natten1d.apply