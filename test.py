import torch
from natten.functional import natten1dqk, natten1dav
from natten_triton.pytorch import natten1d

def test(input_shape, kernel_size):
    q, k, v = torch.randn(input_shape)
    q = q.clone().requires_grad_(True)
    q.retain_grad()
    k = k.clone().requires_grad_(True)
    k.retain_grad()
    v = v.clone().requires_grad_(True)
    v.retain_grad()

    # natten
    s_1 = natten1dqk(q, k, kernel_size, 1)
    s_1.retain_grad()
    p_1 = torch.softmax(s_1, dim=-1)
    p_1.retain_grad()
    o_1 = natten1dav(p_1, v, kernel_size, 1)

    # natten_triton
    q_2 = q.detach().clone()
    q_2.requires_grad = True
    q_2.retain_grad()
    k_2 = k.detach().clone()
    k_2.requires_grad = True
    k_2.retain_grad()
    v_2 = v.detach().clone()
    v_2.requires_grad = True
    v_2.retain_grad()

    o_2 = natten1d(q_2, k_2, v_2, kernel_size)

    print('Forward pass:', torch.allclose(o_1, o_2, atol=1e-5))

    # Check backward pass.
    loss = torch.sum(o_1 ** 2)
    loss.backward()
    loss_2 = torch.sum(o_2 ** 2)
    loss_2.backward()
    print('Backward pass (Q):', torch.allclose(q.grad, q_2.grad, atol=1e-5))
    print('Backward pass (K):', torch.allclose(k.grad, k_2.grad, atol=1e-5))
    print('Backward pass (V):', torch.allclose(v.grad, v_2.grad, atol=1e-5))

if __name__ == '__main__':
    print('# 1D attention')
    test((3, 2, 3, 4, 2), 3)