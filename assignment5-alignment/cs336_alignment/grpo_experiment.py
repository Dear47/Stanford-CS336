#%%
import wandb
import os
import argparse
import gc
import torch
import torch.distributed as dist
from tqdm import tqdm
from typing import TYPE_CHECKING, Callable, List, Dict, Literal
from dataclasses import dataclass, asdict
from unittest.mock import patch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
if TYPE_CHECKING:
    from transformers import AutoTokenizer
    from vllm import LLM

# Import helpers
from cs336_alignment.data_utils import *
from cs336_alignment.sft_helper import (
    tokenize_and_prompt_and_output, 
    get_response_log_probs,
    save_tokenizer_and_model,
)
from cs336_alignment.grpo_helper import *
from cs336_alignment.overwatch import ExperimentLogger
from cs336_alignment.paths import (
    checkpoint_path,
    log_path,
    model_path,
    prompt_template_path,
    result_path,
    test_data_path,
    train_data_path,
)

# Configure logger
script_dir = os.path.dirname(os.path.abspath(__file__))
logger = None

def model_load_kwargs(train_device: str) -> dict:
    kwargs = {
        "use_cache": False,
        "trust_remote_code": True,
    }
    if str(train_device).startswith("cuda"):
        kwargs.update({
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        })
    return kwargs

def clear_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def require_vllm_cuda(device: str):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("GRPO requires vLLM on CUDA for rollout generation/evaluation.")

# --- Configuration ---
@dataclass
class TrainConfig:
    model_name: str = "Qwen2.5-Math-1.5B"
    data_name: str = "MATH-500"
    
    # GRPO 参数
    n_grpo_steps: int = 200
    eval_steps: int = 5
    eval_samples: int = 1024
    lr: float = 1e-6
    advantage_eps: float = 1e-6
    cliprange: float = 0.2
    use_std_normalization: bool = True 

    # rollout 配置
    rollout_batch_size: int = 256
    group_size: int = 8
    inference_batch_size: int = 4

    # 生成配置
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024

    # train-loop 配置
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    loss_normalization: Literal["token_mean", "constant"] = "token_mean"
    
    # 系统配置
    gpu_memory_utilization: float = 0.85
    max_grad_norm: float = 1.0
    train_device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
        assert self.rollout_batch_size % self.group_size == 0, "rollout_batch_size must be divisible by group_size"
        assert self.train_batch_size >= self.group_size, "train_batch_size must be greater than or equal to group_size"

        # 训练时的 Micro Batch 是决定显存占用的关键!    
        self.micro_train_batch_size = self.train_batch_size // self.gradient_accumulation_steps
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size
        self.n_microbatches_per_rollout_batch = self.rollout_batch_size // self.micro_train_batch_size
        
        self.model_path: str = str(model_path(self.model_name))
        self.data_path: str = str(train_data_path(self.data_name))
        self.prompt_template_path: str = str(prompt_template_path())

    def validate_paths(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        if not os.path.exists(self.prompt_template_path):
            logger.warning(f"Prompt file not found at {self.prompt_template_path}. Please check the path.")
    

@dataclass
class EvalConfig:
    data_name:str = "MATH-500"
    eval_device: str = "cuda"
    seed: int = 42
    max_length: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self):        
        self.data_path: str = str(test_data_path(self.data_name))
        self.prompt_template_path: str = str(prompt_template_path())

    def validate_paths(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        if not os.path.exists(self.prompt_template_path):
            logger.warning(f"Prompt file not found at {self.prompt_template_path}. Please check the path.")

# --- vLLM Helpers ---
def init__vllm(
        model_id: str,
        device: str,
        seed: int,
        gpu_memory_utilization: float = 0.85,
) -> "LLM":
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    # Patching to allow running in potentially constrained environments
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1) 
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)  
    
    with world_size_patch, profiling_patch: 
        return LLM(  
            model=model_id, 
            device=device, 
            dtype="bfloat16", 
            enable_prefix_caching=True, 
            gpu_memory_utilization=gpu_memory_utilization, 
            tensor_parallel_size=1,
            trust_remote_code=True  # Qwen needs this param
        )

def load_policy_into_vllm_instance(policy: torch.nn.Module, vllm: "LLM", tie_weights:bool=False):
    """
    Transfer weights from the training policy model to the vLLM engine for evaluation.
    """
    logger.info("⚙️ Loading policy weights into vLLM...")
    try:
        # vLLM internal access to model
        llm_model = vllm.llm_engine.model_executor.driver_worker.model_runner.model
        if not tie_weights:
            llm_model.load_weights(policy.state_dict().items())
        else:
            # CPU Offload strategy
            policy.eval()
            policy.tie_weights()
            cpu_sd = {k: v.detach().to("cpu") for k, v in policy.state_dict().items()}
            llm_model.load_weights(cpu_sd.items())
            policy.train()
        logger.info("✅ Policy loaded into vLLM.")
    except Exception as e:
        logger.error(f"❌ to load weights into vLLM: {e}")
        raise e

# --- Dataset & Collate ---
class GRPODataset(Dataset):
    def __init__(
        self, 
        prompts: List[str], 
        cots: List[str], 
        groundtruths: List[str], 
        advantages: torch.Tensor, 
        raw_rewards: torch.Tensor,
        old_log_probs: torch.Tensor
    ):
        self.prompts = prompts
        self.cots = cots
        self.groundtruths = groundtruths
        self.advantages = advantages
        self.raw_rewards = raw_rewards
        self.old_log_probs = old_log_probs # Shape: (Total_Samples, Max_Seq_Len)

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int):
        # 返回字典，方便 collate_fn 处理
        return {
            "prompt": self.prompts[idx],
            "cot": self.cots[idx],
            "groundtruth": self.groundtruths[idx],
            "advantage": self.advantages[idx],  # scalar
            "raw_rewards": self.raw_rewards[idx],  # scalar
            "old_log_prob": self.old_log_probs[idx]  # tensor 1D
        }

def get_grpo_collate_fn(tokenizer) -> Dict:
    def collate_fn(batch) -> Dict:
        prompts = [item['prompt'] for item in batch]
        cots = [item['cot'] for item in batch]
        
        # 1. 处理文本数据 (Tokenize, Padding, Masking)
        tokenized_data = tokenize_and_prompt_and_output(prompts, cots, tokenizer)
        
        # 2. 处理 Advantage (堆叠为 Tensor)
        advantages = torch.tensor([item['advantage'] for item in batch], dtype=torch.float32)
        raw_rewards = torch.tensor([item['raw_rewards'] for item in batch], dtype=torch.float32)
        
        # 3. 处理 Old Log Probs (Padding)
        # 必须 Pad 到当前 batch 的最大长度，通常使用 0.0 (因为 mask 掉的部分不会用到)
        old_log_probs_list = [item['old_log_prob'] for item in batch]
        old_log_probs = pad_sequence(old_log_probs_list, batch_first=True, padding_value=0.0)
        
        return {
            "input_ids": tokenized_data['input_ids'],       # (B, L)
            "labels": tokenized_data['labels'],             # (B, L) - shifted
            "response_mask": tokenized_data['response_mask'], # (B, L)
            "advantages": advantages,                       # (B,)
            "raw_rewards": raw_rewards,                     # (B,)
            "old_log_probs": old_log_probs,                 # (B, L)
        }
    return collate_fn

# --- Training Functions ---

@torch.no_grad()
def obtain_grpo_dataset(
    policy_model: torch.nn.Module,
    vllm_model: "LLM",
    tokenizer: "AutoTokenizer",
    reward_fn: Callable,
    prompts: List[str],
    groundtruths: List[str],
    train_config: TrainConfig
) -> Dataset:
    """
    Rollout 阶段：vLLM 生成 -> CPU 算分 -> Policy Model 算基准概率 -> 构造 Dataset
    """
    from cs336_alignment.math_baseline import get_response
    # 1. vLLM 生成 (Cuda:1)
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=train_config.sampling_temperature,
        max_tokens=train_config.sampling_max_tokens,
        min_tokens=train_config.sampling_min_tokens,
        n=train_config.group_size,
        seed=train_config.seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    # prompts 只有 n_prompts_per_rollout_batch 个，需要生成 expand 后的 responses
    outputs:List[List[str]] = get_response(vllm_model, prompts, sampling_params)  # n_prompts_per_rollout_batch * group_size
    
    flat_prompts, flat_responses, flat_groundtruths = [], [], []
    for prompt, groundtruth, output_group in zip(prompts, groundtruths, outputs):
        for output_text in output_group:
            flat_prompts.append(prompt)  # n_prompts_per_rollout_batch * group_size
            flat_responses.append(output_text)  # n_prompts_per_rollout_batch * group_size
            flat_groundtruths.append(groundtruth)  # n_prompts_per_rollout_batch * group_size
            
    # 2. 计算 Rewards (CPU)
    advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
        reward_fn,
        flat_responses,
        flat_groundtruths,
        train_config.group_size,
        train_config.advantage_eps,
        train_config.use_std_normalization
    )
    
    # 3. 计算 Old Log Probs (Cuda:0)
    logger.info("🧮 [Rollout] Computing reference log probs...")
    policy_model.eval()
    old_log_probs_list:List[torch.Tensor] = []
    
    # 分批推理以防 OOM
    inference_batch_size = train_config.inference_batch_size
    total_samples = len(flat_prompts)
    
    for i in range(0, total_samples, inference_batch_size):
        batch_prompts = flat_prompts[i : i + inference_batch_size]
        batch_responses = flat_responses[i : i + inference_batch_size]
        
        # Tokenize & Align
        batch_data = tokenize_and_prompt_and_output(batch_prompts, batch_responses, tokenizer)
        input_ids = batch_data['input_ids'].to(policy_model.device)  # (B, L)
        labels = batch_data['labels'].to(policy_model.device)  # (B, L)
        
        # Forward Pass 获取 Log Probs
        log_probs_dict = get_response_log_probs(policy_model, input_ids, labels)  # (B, L)
        old_log_probs_list.append(log_probs_dict['log_probs'].cpu()) # 移回 CPU, List 长度为 rollout_batch_size // inference_batch_size

    # 统一 Padding (不同 batch 长度可能不同)
    all_sequences:List[torch.Tensor] = [seq for batch in old_log_probs_list for seq in batch]  # List 长度为 rollout_batch_size
    old_log_probs_tensor:torch.Tensor = pad_sequence(all_sequences, batch_first=True, padding_value=0.0) # (rollout_batch_size, max_seq_len)
    
    # Logging
    wandb.log(reward_metadata)

    return GRPODataset(
        prompts=flat_prompts,
        cots=flat_responses,
        groundtruths=flat_groundtruths,
        advantages=advantages,
        raw_rewards=raw_rewards,
        old_log_probs=old_log_probs_tensor
    )  # (rollout_batch_size,)

def train_grpo_model(
    model: torch.nn.Module,
    tokenizer: "AutoTokenizer",
    grpo_dataset: Dataset,
    train_config: TrainConfig,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    """
    Run the GRPO training loop.
    """
    model.train()
    
    # Create DataLoader
    collate_fn = get_grpo_collate_fn(tokenizer)
    dataloader = DataLoader(
        dataset=grpo_dataset, 
        batch_size=train_config.micro_train_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    device = train_config.train_device

    # Init accumulative variable
    running_loss = 0.0
    running_entropy = 0.0
    running_kl = 0.0
    running_len = 0.0

    optimizer.zero_grad()
    for _ in range(train_config.epochs_per_rollout_batch):
        progress_bar = tqdm(dataloader, desc=f"Training ({len(grpo_dataset)} samples)")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            response_mask = batch['response_mask'].to(device)
            labels = batch['labels'].to(device)
            advantages = batch['advantages'].to(device)         # (B,)
            raw_rewards = batch['raw_rewards'].to(device)        # (B,)
            old_log_probs = batch['old_log_probs'].to(device)   # (B, L)

            # Compute log probs for the specific tokens (requires forward pass)          
            log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
            policy_log_probs = log_probs_dict['log_probs']  # (B, L)
            token_entropy = log_probs_dict["token_entropy"]  # (B, l)

            # 由于在构建 Dataset 时，将所有样本的旧概率 Padding 到了全局最大长度
            # 在训练循环中，DataLoader 取出的一个 Batch，重新 Tokenize 并 Padding 到该 Batch 的最大长度
            # 这两个最大长度很可能是不同的，且后者一定不大于前者，这就需要将前者裁剪到与后者相同的长度
            if old_log_probs.shape[1] > policy_log_probs.shape[1]:
                old_log_probs = old_log_probs[:, :policy_log_probs.shape[1]]

            # Compute KL div
            with torch.no_grad():
                log_ratio = policy_log_probs - old_log_probs
                approx_kl = (log_ratio.exp() - 1) - log_ratio
                mean_kl = masked_mean(approx_kl, response_mask)  # scalar
            
            # Compute average response length
            with torch.no_grad():
                avg_len = response_mask.sum(dim=1).float().mean()  # scalar    

            # Compute average entropy of the response
            mean_entropy = masked_mean(token_entropy, response_mask)

            # Compute Loss
            loss, metadata = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                loss_type=train_config.loss_type,
                raw_rewards=raw_rewards.unsqueeze(1),
                advantages=advantages.unsqueeze(1),
                old_log_probs=old_log_probs,
                cliprange=train_config.cliprange,
                loss_normalization=train_config.loss_normalization,
                normalize_constant=train_config.sampling_max_tokens,
            )
            
            running_loss += loss.item() * train_config.gradient_accumulation_steps # scale back for display
            running_entropy += mean_entropy.item()
            running_kl += mean_kl.item()
            running_len += avg_len.item()

            is_last_batch = step == len(dataloader) - 1
            if (step + 1) % train_config.gradient_accumulation_steps == 0 or is_last_batch:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                accumulated_steps = (step % train_config.gradient_accumulation_steps) + 1
                avg_loss = running_loss / accumulated_steps
                avg_entropy = running_entropy / accumulated_steps
                avg_kl = running_kl / accumulated_steps
                avg_len = running_len / accumulated_steps
                
                wandb.log({
                    "train/loss": avg_loss,
                    "train/entropy": avg_entropy,
                    "train/approx_kl": avg_kl,
                    "train/avg_response_len": avg_len,
                    "train/clip_fraction": metadata.get('clip_fraction', 0),
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/global_step": global_step
                })

                # Reset accumulative variable
                running_loss = 0.0
                running_entropy = 0.0
                running_kl = 0.0
                running_len = 0.0

                global_step += 1
                
    return model, global_step

def evaluate_grpo_model(vllm: "LLM", eval_config: EvalConfig, current_label: str, limit=200, save:bool=True) -> float:
    """
    Evaluates the vLLM model after several grpo steps on the test set.
    """
    logger.info(f"🧪 Evaluating {current_label}...")
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    from cs336_alignment.math_baseline import evaluate_vllm
    from vllm import SamplingParams

    _, prompts, cots, groundtruths = load_and_format_prompts(
        eval_config.data_path,
        eval_config.prompt_template_path,
        limit,
        eval_config.seed)
    
    # Generate
    # Stop at </answer> to prevent hallucination
    sampling_params = SamplingParams(
        temperature=eval_config.temperature, 
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_length,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    output_path = result_path(eval_config.data_name, "grpo", f"{current_label}.jsonl")
    acc = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, cots, groundtruths, sampling_params, output_path, save)
    return acc
    
# --- Main Experiment ---
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="Qwen2.5-Math-1.5B")
    parser.add_argument("--data_name",type=str,default="MATH-500")
    parser.add_argument("--groupsize", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline", help="Literal[no_baseline, reinforce_with_baseline, grpo_clip]")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--exp_id", type=str, default="default", help="Identifier for this experiment run")
    parser.add_argument("--use_std_normalization", action="store_true")
    parser.add_argument("--loss_normalization", type=str, default="token_mean", choices=["token_mean", "constant"])
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    global logger
    log_file_name = f"{args.exp_id}.log"
    logger = ExperimentLogger(log_file=str(log_path(args.data_name, "grpo", log_file_name)))

    train_config = TrainConfig(
        model_name=args.model_name,
        data_name=args.data_name,
        lr=args.lr,
        n_grpo_steps=args.steps,
        group_size=args.groupsize,
        loss_type=args.loss_type, 
        train_device=args.train_device,
        use_std_normalization=args.use_std_normalization,
        loss_normalization=args.loss_normalization,
    )
    eval_config = EvalConfig(
        data_name=args.data_name,
        eval_device=args.eval_device
    )
    train_config.validate_paths()
    eval_config.validate_paths()
    require_vllm_cuda(eval_config.eval_device)
    
    run_name = f"{args.data_name}-{args.loss_type}-{'std'if args.use_std_normalization else'no_std'}-G{args.groupsize}-STEP{args.steps}-lr{args.lr}"
    wandb.init(
        project="cs336-a5-grpo_experiment",
        config=asdict(train_config),
        name=run_name,
        mode="offline"
    )

    # Initialize Model and Tokenizer
    # We will reload the model for each experiment iteration to start fresh
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Initialize vLLM (holds memory, we will update weights by 'load_policy_into_vllm_instance' later)
    logger.info(f"⚙️ Initializing vLLM on {eval_config.eval_device}...") 
    llm = init__vllm(train_config.model_path, eval_config.eval_device, eval_config.seed, train_config.gpu_memory_utilization)
    logger.info("✅ Initialized vLLM successfully.") 

    # Initialize Policy Model
    logger.info(f"⚙️ Initializing Policy Model {train_config.train_device}...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        train_config.model_path,
        **model_load_kwargs(train_config.train_device),
    ).to(train_config.train_device)
    policy_model.gradient_checkpointing_enable()
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("✅ Initialized Policy Model successfully.")

    global_step = 0

    total_training_steps = (train_config.n_grpo_steps * train_config.epochs_per_rollout_batch * train_config.rollout_batch_size // train_config.micro_train_batch_size) // train_config.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=train_config.lr, weight_decay=0.0, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_training_steps, 1))
    
    # Expert Iteration Loop
    for grpo_step in range(train_config.n_grpo_steps):
        logger.info(f"\n{'='*10} GRPO Step {grpo_step+1}/{train_config.n_grpo_steps} {'='*10}")

        # clean fragmented memory
        gc.collect()
        clear_device_cache()

        # Sample n prompts(questions)
        current_seed = train_config.seed + grpo_step  # 防止每次采样的种子是一样的 
        _, prompts, _, groundtruths = load_and_format_prompts(
            train_config.data_path,
            train_config.prompt_template_path,
            train_config.n_prompts_per_rollout_batch,
            current_seed,
        )

        # Set the old policy model (Policy -> vLLM)
        load_policy_into_vllm_instance(policy_model, llm)

        grpo_dataset = obtain_grpo_dataset(
            policy_model,
            llm,
            tokenizer, 
            r1_zero_reward_fn,
            prompts,
            groundtruths,
            train_config
        )

        #===================================
        # 以上产生一个 rollout_batch 所需的数据，
        # 下面在 train-loop 中利用这一个 rollout
        #===================================
        policy_model, global_step = train_grpo_model(
            policy_model, 
            tokenizer, 
            grpo_dataset,
            train_config,
            global_step,
            optimizer,
            scheduler,
        )

        # Free up dataset memory
        del grpo_dataset
        gc.collect()
        clear_device_cache()

        # Evaluate
        if (grpo_step + 1) % train_config.eval_steps == 0:
            # Set the old policy model
            load_policy_into_vllm_instance(policy_model, llm)
            
            save = False if (grpo_step+1)%50 else True
            acc = evaluate_grpo_model(llm, eval_config, f"{args.exp_id}_step{grpo_step+1}", train_config.eval_samples, save=save)

            wandb.log({
                "eval/step": grpo_step + 1,
                "eval/accuracy": acc
            })
            logger.info(f"📊 Step {grpo_step+1} Accuracy: {acc:.2%}")


    logger.info(f"🎉 GRPO Finished!")
    wandb.finish()
    # Save tokenzier and model
    save_path = checkpoint_path(args.data_name, "grpo", args.exp_id)
    save_tokenizer_and_model(policy_model, tokenizer, save_path)

    if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
    # output_path = os.path.join(script_dir, f"results/gsm8k_sft_full.jsonl")
    # analyze_results(output_path)
