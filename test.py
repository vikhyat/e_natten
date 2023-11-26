import torch
from natten.functional import natten1dqk, natten1dav
from natten_triton import natten1d
from test2 import custom_av

if __name__ == '__main__':
    ## 1D attention test

    kernel_size = 5
    q, k, v = torch.randn(3, 2, 3, 6, 2, dtype=torch.double) # 3 B H T C
    q = q.clone().requires_grad_(True)
    q.retain_grad()
    k = k.clone().requires_grad_(True)
    k.retain_grad()

    # natten
    s_1 = natten1dqk(q, k, kernel_size, 1)
    s_1.retain_grad()
    p_1 = torch.softmax(s_1, dim=-1)
    p_1.retain_grad()
    o_1 = natten1dav(p_1, v, kernel_size, 1)

    # natten_triton
    q_2 = q.clone().detach()
    q_2.requires_grad = True
    q_2.retain_grad()
    k_2 = k.detach().clone()
    k_2.requires_grad = True
    k_2.retain_grad()
    v_2 = v.detach().clone()

    o_2 = natten1d(q_2, k_2, v_2, kernel_size)

    print('1D forward pass:', torch.allclose(o_1, o_2, atol=1e-5))

    # Check backward pass.
    loss = torch.sum(o_1 ** 2)
    loss.backward()
    loss_2 = torch.sum(o_2 ** 2)
    loss_2.backward()
    print('1D backward pass (Q):', torch.allclose(q.grad, q_2.grad, atol=1e-5))
    print('1D backward pass (K):', torch.allclose(k.grad, k_2.grad, atol=1e-5))