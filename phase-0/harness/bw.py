import torch, time
n = 256 * 1024 * 1024            # elements, 512 MB in bf16
x = torch.empty(n, dtype=torch.bfloat16, device='cuda')
y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
torch.cuda.synchronize(); t = time.perf_counter()
reps = 20
for _ in range(reps): y.copy_(x)
torch.cuda.synchronize(); dt = time.perf_counter() - t
gb = x.numel() * 2 * 2 * reps / 1e9    # read + write
print('device copy GB/s:', gb / dt)
h = torch.empty(n, dtype=torch.bfloat16).pin_memory()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(reps): x.copy_(h, non_blocking=True)
torch.cuda.synchronize(); dt = time.perf_counter() - t
print('host to device GB/s:', x.numel() * 2 * reps / 1e9 / dt)
