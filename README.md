# CS336
使用方法:<br>
将原项目的cs336_basic文件夹和tests/adapters.py替换掉即可<br>

cs336-assignment1:<br>
/cs336_basic/tokenizer:实现作业1的BPE训练和分词<br>
/cs336_basic/transformer:实现作业1的transformer_lm的各个模块，并完成了最终训练<br>
```
cs336_basics
├── tokenizer     # 基于字节对编码(BPE)的分词器实现
|   ├── __init__.py
│   ├── pretokenizer.py   # 预分词器
│   ├── tokenizer.py      # BPE编解码实现
│   └── trainer.py        # BPE训练器实现和训练脚本
├── transformer       # Transformer语言模型实现
|   ├── __init__.py
|   ├── module.py         # 模型架构(transformer及其各个模块)
|   ├── transformer_utils.py      # 训练相关工具(loss、优化器等)
|   ├── trainer.py          # 模型训练脚本
|   └── inference.py      # 推理脚本
└── data              # 存放数据集、vocab、merges和模型权重文件
    ├── merges        # 合并规则
    ├── model         # 模型权重/检查点
    ├── token         # .npy文件
    ├── vocab         # 字典
```
![image](training_results.png)
cs336-assignment2:<br>
用pytorch和triton分别实现flashattention的前向传播和反向传播算法<br>
```
cs336_systems
   ├── annotated_attention.py     # 用nvtx包装attention计算过程
   ├── attention_benchmark.py     # 实现不同大小的attention前向传播和反向传播的benchmark脚本
   ├── benchmark.py      # 端到端的benchmark脚本, 实现nvtx包装、内存快照、混合精度等要求
   ├── flash_attention_benchmark.py    # 不同大小和计算精度的flash attention benchmark, 用于比较triton实现和pytorch实现
   ├── flash_attention_pytorch.py   # 基于pytorch实现的flash attention
   ├── flash_attention_triton.py    # 基于triton实现的flash attention
   ├── __init__.py
   ├── mixed_precision_accumulation.py    # 混合精度累加测试
   └── weighted_sum_func.py   #  一个用triton语法实现torch.atuograd.Function的示例
```
持续更新中...
