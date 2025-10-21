# CS336
```
cs336_basics
├── tokenizer     # 基于字节对编码(BPE)的分词器实现
|   ├── __init__.py
│   ├── pretokenizer.py   # 预分词器
│   ├── tokenizer.py      # BPE编解码实现
│   └── trainer.py        # BPE训练器实现和训练脚本
├── transformer       # Transformer语言模型实现
|   ├── __init__.py
|   ├── module.py         # 模型架构（transformer及其各个模块）
|   ├── transformer_utils.py      # 训练相关工具（loss、优化器等）
|   ├── trainer.py          # 模型训练脚本
|   └── inference.py      # 推理脚本
└── data              # 存放数据集、vocab、merges和模型权重文件
    ├── merges        # 合并规则
    ├── model         # 模型权重/检查点
    ├── token         # .npy文件
    ├── vocab         # 字典
```
cs336-assignment1:<br>
/cs336_basic/tokenizer:实现作业1的BPE训练和分词<br>
/cs336_basic/transformer: 实现作业1的transformer_lm的各个模块，并完成了最终训练<br>
将原项目的cs336_basic文件夹和tests/adapters.py替换掉即可<br>
持续更新中...
