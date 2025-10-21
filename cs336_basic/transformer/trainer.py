#%%
import torch
import numpy as np
import numpy.typing as npt
import wandb
import pathlib
from tqdm import tqdm
from cs336_basics.utils.logger import get_logger
from cs336_basics.transformer.transformer_utils import *
from cs336_basics.transformer.module import *
logger = get_logger(__name__,'LM_trainer_v1.txt')
#%%
if __name__ == '__main__':
    model_config = {
        "vocab_size": 10000,      # 词汇表大小
        "context_length": 256,    #上下文长度
        "num_layers": 4,          # TransformerBlock数
        "num_heads": 16,          # 注意力头数
        "d_model": 512,           # 嵌入空间维度
        "d_ff": 1344,             # 前馈网络维度, 512 * 2.66 ≈ 1366, 采用最接近64整数倍的1344(提高GPU利用率)
        "theta": 10000,           # RoPE参数
    }

    optim_config = {
        "lr": 3e-4,               # 学习率
        "weight_decay": 1e-2,     # 权重衰减
        "betas": (0.9, 0.999),    # beta参数
        "max_l2_norm": 1.0,       # 梯度裁剪的最大范数
    }

    train_config = {
        "batch_size": 16,         # 批次大小
        "total_epochs": 1,      # 训练轮数
        "checkpoint_freq": 2000,  # 每隔多少步保存一次检查点
        "log_freq": 50,           # 每隔多少步记录一次日志
        "val_freq": 500,          # 每隔多少步在验证集上评估
        "val_batch_size": 16,     # 验证时每个批次的大小
        "val_batches": 20,        # 验证时使用的批次数量
    }

    PATH = pathlib.Path(__file__).resolve().parent.parent
    data_paths = {
        "training_dataset_path": PATH/"data/token/TinyStoriesV2-GPT4_train.npy",  # 训练集路径
        "validation_dataset_path": PATH/"data/token/TinyStoriesV2-GPT4_valid.npy",  # 验证集路径
        "checkpoint_load_path": None,  # 模型检查点路径
        "checkpoint_save_format": PATH/"data/model/checkpoint_v1_{}.pt",  # 检查点保存路径格式
        "final_model_path": PATH/"data/model/final_model_v1.pt",  # 最终模型保存路径
    }

    run = wandb.init(
        project="cs336-assignment-1",
        name="train_v1",
        config={
            "model": model_config,
            "optimizer": optim_config,
            "training": train_config,
        }
    )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    logger.info("⚙️ Initialize model...")
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        d_ff=model_config["d_ff"],
        theta=model_config["theta"],
        device=device,
    )
    logger.info("✅ Model initializtion has finished!")

    logger.info("⚙️ Initialize optimizer...")
    optim = AdamW(
        params=model.parameters(),
        lr=optim_config["lr"],
        weight_decay=optim_config["weight_decay"],
        betas=optim_config["betas"],
    )
    logger.info("✅ Optimizer initialization has finisded!")


    start_iter = 1
    if data_paths["checkpoint_load_path"]:
        logger.info(f"⚙️ Load checkpoint:{data_paths['checkpoint_load_path']}...")
        start_iter = load_checkpoint(data_paths["checkpoint_load_path"], model=model, optimizer=optim)
        start_iter += 1
        logger.info(f"✅ Load checkpoint successfully, current iteration:{start_iter}.")
    else:
        logger.info("🤔 No checkpoint, train the model from scratch!")

    logger.info(f"⚙️ Loading training dataset:{data_paths['training_dataset_path']}...")
    training_dataset = np.memmap(data_paths["training_dataset_path"], mode='r',dtype=np.uint16)
    logger.info("✅ Load training dataset successfully!")

    validation_dataset = None
    if data_paths["validation_dataset_path"]:
        logger.info(f"⚙️ Loading validation dataset:{data_paths['validation_dataset_path']}...")
        validation_dataset = np.memmap(data_paths["validation_dataset_path"],mode='r',dtype=np.uint16)
        logger.info("✅ Load validation dataset successfully!")
    
    total_tokens = training_dataset.shape[0]
    total_steps = int(train_config['total_epochs'] * total_tokens)//(train_config['batch_size']*model_config['context_length'])
    logger.info(f"Total tokens:{total_tokens}, train epochs:{train_config['total_epochs']}, batch size:{train_config['batch_size']}, context length:{model_config['context_length']}")
    logger.info(f"Total steps:{total_steps}")

    logger.info("🛠️ Start training the model...")
    for step in tqdm(range(start_iter, total_steps+1), desc="training", unit='step'):
        optim.zero_grad()

        lr = cosine_annealing_schedule(
            it=step,
            max_learning_rate=optim_config['lr'],
            min_learning_rate=optim_config["lr"]*0.01,
            warmup_iters=int(0.05 * total_steps),
            cosine_cycle_iters=int(0.95 * total_steps),
        )

        for group in optim.param_groups:
            group['lr'] = lr
        
        inputs, targets = get_batch(
            dataset=training_dataset,
            batch_size=train_config['batch_size'],
            context_length=model_config['context_length'],
            device=device   
        )

        logits = model(inputs)

        loss = cross_entropy(inputs=logits, targets=targets)

        loss.backward()

        gradient_clipping(parameters=model.parameters(),max_l2_norm=optim_config['max_l2_norm'])
        optim.step()

        if step % train_config["log_freq"] == 0:
            grad_norm = compute_grad_norm(model.parameters())
            logger.info(f"🤖 Step:{step}, Loss:{loss.item()}, Grad_l2_norm:{grad_norm}")
            run.log({"train_loss":loss.item(), "lr":lr, "grad_l2_norm":grad_norm, "step":step})
        
        if validation_dataset is not None and step % train_config["val_freq"] == 0:
            logger.info("🎯 Eval the model on validation dataset...")
            val_loss = evaluate_model(
                model=model,
                dataset=validation_dataset,
                batch_size=train_config['val_batch_size'],
                context_length=model_config["context_length"],
                num_batches=train_config["val_batches"],
                device=device
            )
            logger.info(f"🤖 Loss on validation dataset:{val_loss}")
            run.log({'val_loss':val_loss, 'step':step})

        if step % train_config["checkpoint_freq"]==0:
            checkpoint_save_path = str(data_paths["checkpoint_save_format"]).format(step)
            logger.info(f"⚙️ Save the checkpoint to: {checkpoint_save_path}...")
            save_checkpoint(
                model=model,
                optimizer=optim,
                iteration=step,
                out=checkpoint_save_path
            )
            logger.info("✅ Save the checkpoint successfully!")

    logger.info("✅ Train the model successfully!")

    logger.info(f"⚙️ Save the final model to:{data_paths['final_model_path']}...")
    save_checkpoint(
        model=model,
        optimizer=optim,
        iteration=step,
        out=data_paths["final_model_path"]
    )
    logger.info("✅ Save the final model successfully!")

    run.finish()