from cs336_systems.flash_attention_triton import FlashAttentionTriton
from cs336_systems.flash_attention_pytorch import FlashAttentionPytorch
import triton
import triton.language as tl
import torch
import pandas as pd
import itertools
import warnings

def flash_attention_benckmark(config):
    batchsize = config["batch_size"]
    seq_len = config["seq_len"]
    d_model = config["d_model"]
    is_causal = config["is_causal"]
    dtype_str = config["dtype_str"]
    seed = config["seed"]

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        warnings.warn("CUDA 不可用，跳过Triton基准测试。")
        return {
            "batch_size": batchsize, "seq_len": seq_len, "d_model": d_model, "is_causal": is_causal, "dtype_str": dtype_str,
            "fwd_pytorch (ms)": "N/A (CPU)", "fwd_triton (ms)": "N/A (CPU)",
            "bwd_pytorch (ms)": "N/A (CPU)", "bwd_triton (ms)": "N/A (CPU)",
            "e2e_pytorch (ms)": "N/A (CPU)", "e2e_triton (ms)": "N/A (CPU)"
        }
    torch.cuda.set_device(device)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(dtype_str)
    if dtype == torch.float32 and torch.cuda.get_device_capability()[0] >= 8:
        warnings.warn("FP32 在 Ampere 或更高版本的 GPU 上可能很慢或不受支持，建议使用 FP16/BF16。")
    torch.set_float32_matmul_precision('high')
    
    torch.manual_seed(seed)

    Q =  torch.rand(batchsize, seq_len, d_model, device=device, dtype=dtype)
    K =  torch.rand(batchsize, seq_len, d_model, device=device, dtype=dtype)
    V =  torch.rand(batchsize, seq_len, d_model, device=device, dtype=dtype)
    dO = torch.rand(batchsize, seq_len, d_model, device=device, dtype=dtype)
    
    warmup = config.get("warm_up", 10)
    rep = config.get("steps", 50)
    
    # ----------------- forward benchmark ----------------------
    def forward_pytorch():
        FlashAttentionPytorch.apply(Q, K, V, is_causal)

    def forward_triton():
        FlashAttentionTriton.apply(Q, K, V, is_causal)

    time_forward_pytorch = triton.testing.do_bench(forward_pytorch, warmup=warmup, rep=rep, quantiles=[0.5])
    time_forward_triton = triton.testing.do_bench(forward_triton, warmup=warmup, rep=rep, quantiles=[0.5])
    
    # ----------------- backward benckmark ----------------------
    Q_py = Q.clone().requires_grad_()
    K_py = K.clone().requires_grad_()
    V_py = V.clone().requires_grad_()
    O_py = FlashAttentionPytorch.apply(Q_py, K_py, V_py, is_causal)
    def backward_pytorch():
        O_py.backward(gradient=dO, retain_graph=True)
    time_backward_pytorch = triton.testing.do_bench(backward_pytorch, warmup=warmup, rep=rep, quantiles=[0.5])
    del Q_py, K_py, V_py, O_py

    Q_tri = Q.clone().requires_grad_()
    K_tri = K.clone().requires_grad_()
    V_tri = V.clone().requires_grad_()
    O_tri = FlashAttentionTriton.apply(Q_tri, K_tri, V_tri, is_causal)
    def backward_triton():
        O_tri.backward(gradient=dO, retain_graph=True)
    time_backward_triton = triton.testing.do_bench(backward_triton, warmup=warmup, rep=rep, quantiles=[0.5])
    del Q_tri, K_tri, V_tri, O_tri

    # ----------------- fwd_bwd_benchmark ------------------------
    Q_fbb = Q.clone().requires_grad_()
    K_fbb = K.clone().requires_grad_()
    V_fbb = V.clone().requires_grad_()
    def fwd_bwd_pytorch():
        O = FlashAttentionPytorch.apply(Q_fbb, K_fbb, V_fbb, is_causal)
        loss = torch.mean(O)
        loss.backward()
        Q_fbb.grad = None
        K_fbb.grad = None
        V_fbb.grad = None
    time_fwd_bwd_pytorch = triton.testing.do_bench(fwd_bwd_pytorch, warmup=warmup, rep=rep, quantiles=[0.5])
    
    Q_fbb.grad = None
    K_fbb.grad = None
    V_fbb.grad = None
    def fwd_bwd_triton():
        O = FlashAttentionTriton.apply(Q_fbb, K_fbb, V_fbb, is_causal)
        loss = torch.mean(O)
        loss.backward()
        Q_fbb.grad = None
        K_fbb.grad = None
        V_fbb.grad = None
    time_fwd_bwd_triton = triton.testing.do_bench(fwd_bwd_triton, warmup=warmup, rep=rep, quantiles=[0.5])
    del Q_fbb, K_fbb, V_fbb

    return {
        "batch_size": batchsize,
        "seq_len": seq_len,
        "d_model": d_model,
        "is_causal": is_causal,
        "dtype_str": dtype_str,
        "fwd_pytorch (ms)": time_forward_pytorch,
        "fwd_triton (ms)": time_forward_triton,
        "bwd_pytorch (ms)": time_backward_pytorch,
        "bwd_triton (ms)": time_backward_triton,
        "e2e_pytorch (ms)": time_fwd_bwd_pytorch,
        "e2e_triton (ms)": time_fwd_bwd_triton,
    }

def main():
    param_grid = {
        "batch_size":[1],
        "seq_len":[128, 256, 512],  # 128, 256, 512, 1024, 2048, 4096, 8192, 16384
        "d_model":[32, 64],  # 16, 32, 64, 128
        "is_causal":[True],
        "dtype_str":["fp32"],
        "warm_up":[10],
        "steps":[50],
        "seed": [47],
    }

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        param_grid["dtype_str"].append("bf16")

    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    total = len(configs)
    print(f"Running {total} benchmark configurations...")

    for i, config in enumerate(configs, 1):
        config_str = ", ".join(f"{k}={v}" for k,v in config.items() if k not in ['warm_up', 'steps', 'seed'])
        print(f"\n[{i}/{total}] 运行中: {config_str}")
        try:
            result = flash_attention_benckmark(config)
            results.append(result)
            fwd_tri = result.get('fwd_triton (ms)', 'N/A')
            fwd_py = result.get('fwd_pytorch (ms)', 'N/A')
            if isinstance(fwd_tri, float) and isinstance(fwd_py, float):
                print(f"✅ 完成! Triton Fwd: {fwd_tri:.4f} ms, PyTorch Fwd: {fwd_py:.4f} ms")
            else:
                 print(f"✅ 完成! (Triton: {fwd_tri}, PyTorch: {fwd_py})")
                 
        except Exception as e:
            print(f"❌ 失败: {e}")
            fallback = config.copy()
            fallback.update({
                "fwd_pytorch (ms)": "Error", "fwd_triton (ms)": "Error",
                "bwd_pytorch (ms)": "Error", "bwd_triton (ms)": "Error",
                "e2e_pytorch (ms)": "Error", "e2e_triton (ms)": "Error",
            })
            results.append(fallback)

    df = pd.DataFrame(results)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n\n--- 最终基准测试结果 ---")
    
    # 计算加速比
    for col_prefix in ['fwd', 'bwd', 'e2e']:
        py_col = f'{col_prefix}_pytorch (ms)'
        tri_col = f'{col_prefix}_triton (ms)'
        
        if py_col in df.columns and tri_col in df.columns:
            df[py_col] = pd.to_numeric(df[py_col], errors='coerce')
            df[tri_col] = pd.to_numeric(df[tri_col], errors='coerce')
            
            # 计算加速比
            df[f'{col_prefix}_speedup'] = df[py_col] / df[tri_col]

    print("\n" + "="*80)
    print("FLASHATTENTION BENCHMARK RESULTS (DataFrame):")
    print("="*80)
    
    # 定义列的最终顺序
    col_order = [
        "batch_size", "seq_len", "d_model", "dtype_str", "is_causal",
        "fwd_pytorch (ms)", "fwd_triton (ms)", "fwd_speedup",
        "bwd_pytorch (ms)", "bwd_triton (ms)", "bwd_speedup",
        "e2e_pytorch (ms)", "e2e_triton (ms)", "e2e_speedup"
    ]
    
    # 过滤 col_order，只保留 df 中实际存在的列（例如，如果 speedup 计算失败或在 CPU 上运行）
    existing_cols = [col for col in col_order if col in df.columns]
    df_to_save = df[existing_cols]
    
    # 打印到控制台
    print(df_to_save.to_string(index=False, float_format="%.4f"))
    
    # 保存到文件
    try:
        # 使用 "%.4f" 以保持与控制台输出一致的精度
        latex_str = df_to_save.to_latex(index=False, float_format="%.4f")
        markdown_str = df_to_save.to_markdown(index=False, tablefmt="github")

        # 使用 utf-8 编码保存文件
        with open("flash_attention_benchmark_results.tex", "w", encoding="utf-8") as f:
            f.write(latex_str)
        with open("flash_attention_benchmark_results.md", "w", encoding="utf-8") as f:
            f.write(markdown_str)

        print("\n✅ 结果已保存到:")
        print("   - flash_attention_benchmark_results.tex")
        print("   - flash_attention_benchmark_results.md")
    except Exception as e:
        print(f"\n❌ 保存文件失败: {e}")

if __name__ == "__main__":
    main()