"""Cross-node NCCL all_reduce over EFA. Run BEFORE any multi-node model load.

A 555GB checkpoint takes ~10 minutes to load; discovering that ranks cannot
talk to each other AFTER that is ten minutes wasted per attempt. This exercises
exactly the collective vLLM's tensor parallel depends on, in seconds.
"""
import os, torch, torch.distributed as dist

lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()

t = torch.full((1024, 1024), float(r), device=f"cuda:{lr}")
dist.all_reduce(t)
expect = sum(range(w))
ok = abs(t[0, 0].item() - expect) < 1e-3
if r == 0:
    print(f"world={w}  all_reduce sum={t[0,0].item():.0f} expect={expect}  {'OK' if ok else 'WRONG'}",
          flush=True)
dist.barrier()
dist.destroy_process_group()
