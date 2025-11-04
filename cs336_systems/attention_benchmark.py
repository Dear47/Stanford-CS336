#%%
import torch
from torch import Tensor
from einops import einsum
import torch.cuda.nvtx as nvtx
from timeit import default_timer as timer
import math
from jaxtyping import Float
import pandas as pd
import itertools

#%%
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
        d_k = Q.shape[-1]
        attention_score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attention_score, dim=-1)  # softmax over "keys"

        return einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
        
#%%
def device_sync_if_needed(device: str):
    """
    Wait for all GPU kernels to complete
    """
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def set_matmul_precision(fp32_mode: str):
    """
    Controls TF32 usage on Ampere+/Ada+ for FP32 matmuls
    """
    torch.set_float32_matmul_precision(fp32_mode)

def get_qkv(batch_size, seq_len, d_model, device, dtype, requires_grad=False):
    Q = torch.rand(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    K = torch.rand(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    V = torch.rand(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    return Q, K, V

def run_attention(config):
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    d_model = config["d_model"]
    if config["dtype"] == "fp32":
        dtype = torch.float32
    elif config["dtype"] == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    warm_up = config["warm_up"]
    steps = config["steps"]
    seed = config["seed"]
    fp32_matmul = config["fp32_matmul"]
    ifcompile = config["compile"]

    torch.manual_seed(seed)
    set_matmul_precision(fp32_matmul)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    attn = ScaledDotProductAttention().to(device)
    if ifcompile:
        attn = torch.compile(attn, dynamic=True)
    attn.eval()

    fwd_times = []
    bwd_times = []

    try:
        # -------------- forward pass ----------------
        Q,K,V = get_qkv(batch_size, seq_len, d_model, device, dtype)  # without retain graph
        for _ in range(warm_up):
            with torch.no_grad():
                _ = attn(Q, K, V)
            device_sync_if_needed(device=device)

        for _ in range(steps):
            start_time = timer()
            logits = attn(Q, K, V)
            device_sync_if_needed(device=device)
            fwd_time = (timer() - start_time) * 1000
            fwd_times.append(fwd_time)
        
        # ------------- measure memory before backward pass --------------------
        if device.startswith("cuda"):
            mem_before_bwd = torch.cuda.memory_allocated(device=device) / (1024**2)  # MB
        
        # ------------- forward-backward pass -----------------
        for _ in range(warm_up):
            Qg, Kg, Vg = get_qkv(batch_size, seq_len, d_model, device, dtype, True)  # with retain graph
            logits = attn(Qg, Kg, Vg)
            loss = logits.mean()
            loss.backward()
            del Qg, Kg, Vg, logits, loss
            device_sync_if_needed(device=device)

        for _ in range(steps):
            Qg, Kg, Vg = get_qkv(batch_size, seq_len, d_model, device, dtype, True)  # reconstruct its own graph on each step
            logits = attn(Qg, Kg, Vg)
            loss = logits.mean()
            start_time = timer()  # only measure backward pass time
            loss.backward()
            del Qg, Kg, Vg, logits, loss
            device_sync_if_needed(device=device)
            bwd_time = (timer() - start_time) * 1000
            bwd_times.append(bwd_time)

        return {
            "batch_size":batch_size,
            "seq_len":seq_len,
            "d_model":d_model,
            "dtype":config["dtype"],
            "compile":ifcompile,
            "total_fwd_time":sum(fwd_times),
            "total_bwd_time":sum(bwd_times),
            "mem_before_bwd":mem_before_bwd
        }
    
    except Exception as e:
        if "out of memory" in str(e):
            print(f"Error: Out of Memory!")
            return {
            "batch_size":batch_size,
            "seq_len":seq_len,
            "d_model":d_model,
            "dtype":config["dtype"],
            "compile":ifcompile,
            "total_fwd_time":"OOM",
            "total_bwd_time":"OOM",
            "mem_before_bwd":"OOM"
            }
        else:
            raise e
        

def main():
    param_grid = {
        "batch_size":[8],
        "seq_len":[256, 1024, 4096],  # 256, 1024, 4096, 8192, 16384
        "d_model":[16, 32, 64, 128],  # 16, 32, 64, 128
        "dtype":["fp32"],
        "warm_up":[5],
        "steps":[100],
        "seed": [47],
        "fp32_matmul": ["high"],  # "medium, high, highest"
        "compile":[False, True],
    }

    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    total = len(configs)
    print(f"Running {total} benchmark configurations...")
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total}] Running: {config}")
        try:
            result = run_attention(config)
            results.append(result)
            print(f"✅ Completed!")
        except Exception as e:
            print(f"❌ Failed: {e}")
            fallback = {k:config.get(k, None) for k in ["batch_size", "seq_len", "d_model", "dtype", "compile"]}
            fallback.update({
                "total_fwd_time":"OOM",
                "total_bwd_time":"OOM",
                "mem_before_bwd":"OOM"
            })
            results.append(fallback)

    df = pd.DataFrame(results)

    col_order = [
        "batch_size", "seq_len", "d_model", "dtype", "compile",
        "total_fwd_time", "total_bwd_time", "mem_before_bwd"
    ]
    df = df[col_order]
    print("\n" + "="*80)
    print("ATTENTION BENCHMARK RESULTS (DataFrame):")
    print("="*80)
    print(df.to_string(index=False))

    latex_str = df.to_latex(index=False, float_format="%.2f")
    markdown_str = df.to_markdown(index=False, tablefmt="github")

    with open("attention_benchmark_results.tex", "w") as f:
        f.write(latex_str)
    with open("attention_benchmark_results.md", "w") as f:
        f.write(markdown_str)

    print("\n✅ Results saved to:")
    print("  - attention_benchmark_results.tex")
    print("  - attention_benchmark_results.md")
    
if __name__ == "__main__":
    main()