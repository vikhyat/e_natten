import torch
import triton
from e_natten import natten2d
from natten.functional import natten2dqk, natten2dav

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['D'],
        x_vals=[2**i for i in range(4, 10)],
        line_arg="provider",
        line_vals=['triton', 'natten'],
        line_names=['e_natten (fused)', 'natten (original)'],
        ylabel='time (ms)',
        args={'B': 4, 'N': 4, 'C': 128, 'kernel_size': 5},
        plot_name=f"2d-fwd",
    )
])
def bench_fwd(B, N, D, C, kernel_size, provider):
    q, k, v = torch.randn((3, B, N, D, D, C)).cuda()
    if provider == 'triton':
        fn = lambda: natten2d(q, k, v, kernel_size)
    elif provider == 'natten':
        fn = lambda: natten2dav(torch.softmax(natten2dqk(q, k, kernel_size, 1), dim=-1), v, kernel_size, 1)

    warmup = 200
    rep = 1000
    with torch.no_grad():
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['D'],
        x_vals=[2**i for i in range(4, 9)],
        line_arg="provider",
        line_vals=['triton', 'natten'],
        line_names=['e_natten (fused)', 'natten (original)'],
        ylabel='time (ms)',
        args={'B': 4, 'N': 4, 'C': 128, 'kernel_size': 5},
        plot_name=f"2d-bwd",
    )
])
def bench_bwd(B, N, D, C, kernel_size, provider):
    q, k, v = torch.randn((3, B, N, D, D, C)).requires_grad_().cuda()
    if provider == 'triton':
        def fn():
            out = natten2d(q, k, v, kernel_size)
            loss = torch.sum(out ** 2)
            loss.backward()
    elif provider == 'natten':
        def fn():
            s = natten2dqk(q, k, kernel_size, 1)
            p = torch.softmax(s, dim=-1)
            o = natten2dav(p, v, kernel_size, 1)
            loss = torch.sum(o ** 2)
            loss.backward()
    warmup = 100
    rep = 200
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms

bench_fwd.run(save_path="assets", print_data=True)
bench_bwd.run(save_path="assets", print_data=True)