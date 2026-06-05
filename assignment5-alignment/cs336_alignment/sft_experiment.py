#%%
import wandb
import os
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from typing import TYPE_CHECKING, Union, Tuple, List, Dict
from dataclasses import dataclass, asdict
from unittest.mock import patch
from torch.utils.data import Dataset, DataLoader
if TYPE_CHECKING:
    from transformers import AutoTokenizer
    from vllm import LLM

# Import helpers
from cs336_alignment.data_utils import *
from cs336_alignment.sft_helper import (
    tokenize_and_prompt_and_output, 
    get_response_log_probs,
    sft_microbatch_train_step, 
    save_tokenizer_and_model,
    masked_normalize,
)
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

def can_use_vllm(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()

# --- Configuration ---
@dataclass
class TrainConfig:
    model_name: str = "Qwen2.5-Math-1.5B"
    data_name: str = "MATH-500"
    
    batch_size: int = 2
    lr: float = 1e-5
    gradient_accumulation: int = 16
    max_grad_norm: float = 1.0
    train_device: str = "cuda"
    epochs: int = 1
    seed: int = 42
    sizes_to_eval: Union[int, str] = "full"

    def __post_init__(self):      
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
        gpu_memory_utilization: float = 0.6,
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

class MathSFTDataset(Dataset):
    def __init__(self, prompts: List[str], cots: List[str], groundtruths: List[str]):
        self.prompts = prompts
        self.cots = cots
        self.groundtruths = groundtruths

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        prompt = self.prompts[idx]
        cot = self.cots[idx]
        groundtruth = self.groundtruths[idx]

        return prompt, cot, groundtruth
    
def get_collate_fn(tokenizer) -> Dict:
    def collate_fn(batch) -> Dict:
        prompts, cots, groundtruths = zip(*batch)
        # prompts are the raw question text with prompt template
        # cots are the raw answer text
        return {
            **tokenize_and_prompt_and_output(prompts, cots, tokenizer),
            "groundtruths": groundtruths
            }
    return collate_fn

# --- Training Functions ---
def train_sft_model(
    model: torch.nn.Module,
    tokenizer: "AutoTokenizer",
    train_dataset: Dataset,
    train_config: TrainConfig,
):
    """
    Run the SFT training loop.
    """
    model.train()
    
    # Create DataLoader
    collate_fn = get_collate_fn(tokenizer)
    dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    
    # Learning-rate scheduler
    num_training_steps = len(dataloader) * train_config.epochs // train_config.gradient_accumulation
    if num_training_steps == 0: num_training_steps = 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    device = train_config.train_device
    
    global_step = 0
    # Init accumulative variable
    running_loss = 0.0
    running_entropy = 0.0
    running_len = 0.0
    
    optimizer.zero_grad()
    for _ in range(train_config.epochs):
        progress_bar = tqdm(dataloader, desc=f"Training ({len(train_dataset)} samples)")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            response_mask = batch['response_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Compute log probs for the specific tokens (requires forward pass)          
            log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
            policy_log_probs = log_probs_dict['log_probs']
            token_entropy = log_probs_dict["token_entropy"]
            
            # Compute average response length
            with torch.no_grad():
                avg_len = response_mask.sum(dim=1).float().mean()  # scalar
            
            # Compute average entropy of the response
            mean_entropy = (token_entropy * response_mask).sum() / response_mask.sum().clamp_min(1)

            # Compute Loss
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=train_config.gradient_accumulation
            )
            
            running_loss += loss.item() * train_config.gradient_accumulation # scale back for display
            running_entropy += mean_entropy.item()
            running_len += avg_len.item()

            is_last_batch = step == len(dataloader) - 1
            if (step + 1) % train_config.gradient_accumulation == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                accumulated_steps = (step % train_config.gradient_accumulation) + 1
                avg_loss = running_loss / accumulated_steps
                avg_entropy = running_entropy / accumulated_steps
                avg_len = running_len / accumulated_steps

                # progress_bar.update(1)
                # progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                wandb.log({
                    "train/loss": avg_loss,
                    "train/entropy": avg_entropy,
                    "train/avg_response_len": avg_len,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/global_step": global_step
                })

                # Reset accumulative variable
                running_loss = 0.0
                running_entropy = 0.0
                running_len = 0.0

                global_step += 1
                
                # Cleanup
                del policy_log_probs, log_probs_dict, loss, token_entropy

    del optimizer, scheduler, dataloader
    return model

def evaluate_sft_model(vllm: "LLM", eval_config: EvalConfig, current_label: str, limit=200) -> float:
    """
    Evaluates the vLLM model on the test set after sft.
    """
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

    output_path = result_path(eval_config.data_name, "sft", f"{current_label}.jsonl")
    acc = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, cots, groundtruths, sampling_params, output_path)
    return acc

# --- Main Experiment ---
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Run SFT Experiment with specific data size and epochs.")
    parser.add_argument("--model_name",type=str,default="Qwen2.5-Math-1.5B")
    parser.add_argument("--data_name",type=str,default="MATH-500")
    parser.add_argument("--size_to_eval", type=str, default="full", help="Size of dataset before eval: 'full' or an integer (e.g., 128)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--exp_id", type=str, default="default", help="Identifier for this experiment run")
    args = parser.parse_args()

    global logger
    log_file_name = f"{args.exp_id}.log"
    logger = ExperimentLogger(log_file=str(log_path(args.data_name, "sft", f"{log_file_name}.log")))

    data_size_val = "full"
    if args.size_to_eval != "full":
        try:
            data_size_val = int(args.size_to_eval)
        except ValueError:
            logger.error("Invalid size_to_eval provided. Must be 'full' or an integer.")
            return
    train_config = TrainConfig(
        model_name=args.model_name,
        data_name=args.data_name,
        train_device=args.train_device,
        epochs=args.epochs,
        sizes_to_eval=data_size_val
    )
    eval_config = EvalConfig(data_name=args.data_name, eval_device=args.eval_device)
    train_config.validate_paths()
    eval_config.validate_paths()
    
    run_name = f"{args.data_name}-SFT-Size{train_config.sizes_to_eval}-E{args.epochs}"
    wandb.init(
        project="cs336-a5-math_sft_experiment",
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
        
    llm = None
    if can_use_vllm(eval_config.eval_device):
        # Initialize vLLM (holds memory, we will update weights by 'load_policy_into_vllm_instance' later)
        logger.info("⚙️ Initializing vLLM...")
        llm = init__vllm(train_config.model_path, eval_config.eval_device, eval_config.seed)
        logger.info("✅ Initialized vLLM successfully.")
    else:
        logger.warning("Skipping vLLM initialization/evaluation because CUDA is unavailable or eval_device is not CUDA.")

    
    limit_size = -1 if train_config.sizes_to_eval == "full" else train_config.sizes_to_eval
    datasets, prompts, cots, groundtruths = load_and_format_prompts(
        train_config.data_path,
        train_config.prompt_template_path,
        limit_size,
        train_config.seed,
    )
    logger.info(f"📚 Loaded {len(datasets)} training examples.")

    # Re-initialize Policy Model
    logger.info("⚙️ Initializing Policy Model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        train_config.model_path,
        **model_load_kwargs(train_config.train_device),
    ).to(train_config.train_device)
    policy_model.gradient_checkpointing_enable()
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("✅ Initialized Policy Model successfully.")
    
    # Load sft training data
    logger.info("⚙️ Loading Math-SFT-Training Data...")
    train_dataset = MathSFTDataset(prompts, cots, groundtruths)
    logger.info(f"✅ Loaded {len(train_dataset)} Math-SFT-Training Data successfully.")
    
    # Train loop
    train_sft_model(policy_model, tokenizer, train_dataset, train_config)

    # Save tokenzier and model
    save_path = checkpoint_path(args.data_name, "sft", args.exp_id)
    save_tokenizer_and_model(policy_model, tokenizer, save_path)

    if llm is not None:
        # Sync weights to vLLM
        load_policy_into_vllm_instance(policy_model, llm)
        
        # Evaluate
        acc = evaluate_sft_model(llm, eval_config, args.exp_id, -1)

        wandb.log({
            "eval/size_to_eval": len(datasets),
            "eval/accuracy": acc
        })

        logger.info(f"🎉 SFT Finished. Accuracy: {acc:.2%}")
    else:
        logger.info("🎉 SFT Finished. Evaluation skipped.")
    wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
    # output_path = os.path.join(script_dir, f"results/gsm8k_sft_full.jsonl")
    # analyze_results(output_path)
