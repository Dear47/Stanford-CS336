#%%
from timeit import default_timer as timer
import pandas as pd
import itertools
import contextlib
import torch
import torch.cuda.nvtx as nvtx
from annotated_attention import NVTXWrappedMHA
from cs336_basics.model import BasicsTransformerLM, CausalMultiHeadSelfAttention
from cs336_basics.optimizer import AdamW
# %%
SIZE2CFG = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
VOCAB_SIZE = 10_000

def build_model(size:str,ctx_len:int,device:str):
    """
    Initial the model
    """
    cfg = SIZE2CFG[size]
    model = BasicsTransformerLM(
        vocab_size = VOCAB_SIZE,
        context_length = ctx_len,
        d_model = cfg["d_model"],
        num_layers = cfg["num_layers"],
        num_heads = cfg["num_heads"],
        d_ff = cfg["d_ff"],
        rope_theta = 10000,
    ).to(device)
    return model

def device_sync_if_needed(device: str):
    """
    Wait for all GPU kernels to complete
    """
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def ifnvtxwrap_mha_if_needed(model:torch.nn.Module):
    """
    Wrap the MHA procedure with nvtx
    """
    for name, module in model.named_children():
        if isinstance(module, CausalMultiHeadSelfAttention):
            setattr(model, name, NVTXWrappedMHA(module))
        else:
            ifnvtxwrap_mha_if_needed(module)

def set_matmul_precision(fp32_mode: str):
    """
    Controls TF32 usage on Ampere+/Ada+ for FP32 matmuls
    """
    torch.set_float32_matmul_precision(fp32_mode)

def autocast(device:str, precision:str):
    """
    Modify the precision automatically
    """
    if device.startswith('cuda') and precision in {'fp16', 'bf16'}:
        dtype = torch.float16 if precision=='fp16' else torch.bfloat16
        return torch.autocast(device_type=device, dtype=dtype)
    return contextlib.nullcontext()

def run_benchmark(config:dict):
    """
    Run a single benchmark configuration and return results as a dict
    """
    size = config["size"]
    ctx = config["ctx"]
    bsz = config["bsz"]
    mode = config["mode"]
    warm_up = config["warm_up"]
    steps = config["steps"]
    seed = config["seed"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    ifnvtx = config["nvtx"]
    wrap_mha = config["wrap_mha"]
    precision = config["precision"]
    mem_snapshot = config["mem_snapshot"]
    mem_snapshot_out = config["mem_snapshot_out"]
    fp32_matmul = config["fp32_matmul"]

    torch.manual_seed(seed)
    set_matmul_precision(fp32_matmul)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    model = build_model(size, ctx, device)
    if wrap_mha:
        ifnvtxwrap_mha_if_needed(model=model)
    model.train()
    
    x = torch.randint(VOCAB_SIZE, (bsz, ctx), device=device, dtype=torch.long)
    y = torch.randint(VOCAB_SIZE, (bsz, ctx), device=device, dtype=torch.long)

    optim = None
    if mode == "fwd_bwd":
        optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # GradScaler for fp16 training
    scaler = None
    if mode == "fwd_bwd" and device.startswith("cuda") and precision == "fp16":
        scaler = torch.amp.GradScaler(enabled=True)
    
    fwd_times = []
    bwd_times = []
    opt_times = []

    # warm up
    print(f"Runnung {warm_up} warm-up steps on {device}")
    if ifnvtx:
        nvtx.range_push("warm_up")
    for _ in range(warm_up):
        with autocast(device=device,precision=precision):
            if mode == "fwd":
                with nvtx.range(f"warmup_fwd") if ifnvtx else contextlib.nullcontext():
                    _ = model(x)
            else:
                with nvtx.range(f"warmup_fwd") if ifnvtx else contextlib.nullcontext():
                    logits = model(x)
                loss = logits.mean()
                if scaler is not None:
                    with nvtx.range(f"warmup_bwd") if ifnvtx else contextlib.nullcontext():
                        scaler.scale(loss).backward()
                    with nvtx.range(f"warmup_opt") if ifnvtx else contextlib.nullcontext():
                        with nvtx.range(f"optimizer.step") if ifnvtx else contextlib.nullcontext():
                            scaler.step(optim)
                            scaler.update()
                        with nvtx.range(f"optimizer.zero_grad") if ifnvtx else contextlib.nullcontext():
                            optim.zero_grad()
                else:
                    with nvtx.range(f"warmup_bwd") if ifnvtx else contextlib.nullcontext():
                        loss.backward()
                    with nvtx.range(f"warmup_opt") if ifnvtx else contextlib.nullcontext():
                        with nvtx.range(f"optimizer.step") if ifnvtx else contextlib.nullcontext():
                            optim.step()
                        with nvtx.range(f"optimizer.zero_grad") if ifnvtx else contextlib.nullcontext():
                            optim.zero_grad()
    if ifnvtx:
        nvtx.range_pop()
    device_sync_if_needed(device=device)  # 等待warm up完成

    if mem_snapshot:
         torch.cuda.memory._record_memory_history(max_entries=1000000)

    # measure loop
    print(f"\nRunning {steps} measure-loop ({mode}) steps on {device}...")
    start_time = timer()
    if ifnvtx:
        nvtx.range_push("measure_loop")
    for _ in range(steps):
        device_sync_if_needed(device=device)  # 确保GPU已空闲
        t0 = timer()
        with autocast(device=device,precision=precision):
            with nvtx.range(f"mearsure_fwd") if ifnvtx else contextlib.nullcontext():
                logits = model(x)
            device_sync_if_needed(device=device)  # 等待fwd完成
            t1 = timer()
            fwd_ms = (t1 - t0) * 1000
            fwd_times.append(fwd_ms)
            if mode == "fwd":
                bwd_times.append(0.0)
                opt_times.append(0.0)
            else:
                device_sync_if_needed(device=device)  # 确保GPU已空闲
                t2 = timer()
                loss = logits.mean()
                with nvtx.range(f"mearsure_bwd") if ifnvtx else contextlib.nullcontext():
                    loss.backward()
                device_sync_if_needed(device=device)  # 等待bwd完成
                t3 = timer()
                bwd_ms = (t3 - t2) * 1000
                bwd_times.append(bwd_ms)

                with nvtx.range(f"mearsure_opt") if ifnvtx else contextlib.nullcontext():
                    device_sync_if_needed(device=device)  # 确保GPU已空闲
                    t4 = timer()
                    with nvtx.range(f"optimizer.step") if ifnvtx else contextlib.nullcontext():
                        optim.step()
                    with nvtx.range(f"optimizer.zero_grad") if ifnvtx else contextlib.nullcontext():
                        optim.zero_grad()
                    device_sync_if_needed(device=device)  # 等待opt完成
                    t5 = timer()
                    opt_ms = (t5 - t4) * 1000
                    opt_times.append(opt_ms)
    if ifnvtx:
        nvtx.range_pop()
    device_sync_if_needed(device=device)  # 等待measure loop完成
    end_time = timer()

    if mem_snapshot:
        torch.cuda.memory._dump_snapshot(mem_snapshot_out)
        torch.cuda.memory._record_memory_history(enabled=None)

    total_time = (end_time - start_time)
    total_fwd = sum(fwd_times)
    total_bwd = sum(bwd_times)
    total_opt = sum(opt_times)

    # Clean up to avoid OOM in loop
    del model, x, y, optim
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "size": size,
        "ctx": ctx,
        "bsz": bsz,
        "mode": mode,
        "precision": precision,
        "total_time_ms": round(total_time * 1000, 4),
        "total_fwd_time_ms": round(total_fwd, 4),
        "total_bwd_time_ms":round(total_bwd, 4),
        "total_opt_time_ms":round(total_opt, 4)
    }

def main():
    # Define parameter grid
    param_grid = {
        "size": ["large"],  # "small", "medium", "large"
        "ctx": [256],
        "bsz": [4],
        "mode": ["fwd_bwd"],  # "fwd, fwd_bwd"
        "precision": ["fp16"],  # "fp32, fp16, bf16"
        "warm_up": [5],
        "steps": [10],
        "seed": [47],
        "lr": [1e-3],
        "weight_decay": [0.01],
        "nvtx": [True],
        "wrap_mha": [True],
        "mem_snapshot":[True],
        "mem_snapshot_out":["memory_snapshot.pickle"],  # https://pytorch.org/memory_viz
        "fp32_matmul": ["high"],  # "medium", "high", "highest"
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    total = len(configs)
    print(f"Running {total} benchmark configurations...")

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total}] Running: {config}")
        try:
            result = run_benchmark(config)
            results.append(result)
            print(f"✅ Completed!")
        except Exception as e:
            print(f"❌ Failed: {e}")
            # Optionally append failed run with NaNs
            fallback = {k: config.get(k, None) for k in ["size", "ctx", "bsz", "mode", "precision"]}
            fallback.update({
                "total_time_ms": None,
            })
            results.append(fallback)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Optional: Reorder columns for readability
    col_order = [
        "size", "ctx", "bsz", "mode", "precision",
        "total_time_ms", "total_fwd_time_ms", "total_bwd_time_ms", "total_opt_time_ms",
    ]
    df = df[col_order]

    print("\n" + "="*80)
    print("BENCHMARK RESULTS (DataFrame):")
    print("="*80)
    print(df.to_string(index=False))

    # Save to LaTeX and Markdown
    latex_str = df.to_latex(index=False, float_format="%.2f")
    markdown_str = df.to_markdown(index=False, tablefmt="github")

    with open("benchmark_results.tex", "w") as f:
        f.write(latex_str)
    with open("benchmark_results.md", "w") as f:
        f.write(markdown_str)

    print("\n✅ Results saved to:")
    print("  - benchmark_results.tex")
    print("  - benchmark_results.md")

if __name__ == "__main__":
    main()
# %%
