import torch
from torch.autograd import gradcheck
from natten_triton.pytorch import natten1d

# Define an input tensor for gradcheck
q, k, v = torch.randn((3, 2, 3, 8, 2), dtype=torch.double)
q = q.clone().requires_grad_(True)
q.retain_grad()
k = k.clone().requires_grad_(True)
k.retain_grad()
v = v.clone().requires_grad_(True)
v.retain_grad()

# Use gradcheck to verify the gradients of natten1d
gradcheck_result = gradcheck(natten1d, (q, k, v, 7), eps=1e-6, atol=1e-4)
print("Gradient check result:", gradcheck_result)
