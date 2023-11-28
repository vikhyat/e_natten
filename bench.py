import torch
import triton
from e_natten import natten1d
from natten.functional import natten1dqk, natten1dav

configs = []
configs.append(triton.testing.Benchmark(
    x_names=['Sequence Length'],
    x_vals=[2**i for i in range(8, 16)],
    line_arg="provider",
    line_vals=['triton', 'natten'],
    line_names=['Natten (triton)', 'Natten (original)'],
    ylabel='time (ms)',
    args={'B': 4, 'N': 12, 'C': 1024, 'kernel_size': 5},
    plot_name=f"1d-fwd",
))

@triton.testing.perf_report(configs)
def bench_1d_fwd(B, N, T, C, kernel_size, provider):
    q, k, v = torch.randn((3, B, N, T, C)).cuda()
    if provider == 'triton':
        fn = lambda: natten1d(q, k, v, kernel_size)
    elif provider == 'natten':
        fn = lambda: natten1dav(torch.softmax(natten1dqk(q, k, kernel_size, 1), dim=-1), v, kernel_size, 1)
    warmup = 25
    rep = 1000
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms

bench_1d_fwd.run(save_path=".", print_data=True)