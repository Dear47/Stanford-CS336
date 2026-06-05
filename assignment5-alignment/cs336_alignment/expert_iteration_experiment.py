#%%
import wandb
import os
import torch
import torch.distributed as dist
import gc
import argparse
from tqdm import tqdm
from typing import TYPE_CHECKING, Callable, Tuple, List, Dict
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

def clear_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def require_vllm_cuda(device: str):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Expert Iteration requires vLLM on CUDA for rollout generation/evaluation.")

# --- Configuration ---
@dataclass
class TrainConfig:
    model_name: str = "Qwen2.5-Math-1.5B"
    data_name: str = "MATH-500"
    
    batch_size: int = 2
    lr: float = 1e-5
    gradient_accumulation: int = 8
    max_grad_norm: float = 1.0
    train_device: str = "cuda"
    sft_epochs: int = 1
    seed: int = 42

    # Expert Iteration hyperparameters
    n_ei_steps: int = 5
    ei_batchsize: int = 1024
    G: int = 4
    ei_temperature: float = 1.0  # generate diverse outputs
    ei_top_p: float = 1.0
    ei_min_tokens: int = 4
    ei_max_tokens: int = 1024

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
    sft_dataset: Dataset,
    train_config: TrainConfig,
):
    """
    Run the SFT training loop.
    """
    model.train()
    
    # Create DataLoader
    collate_fn = get_collate_fn(tokenizer)
    dataloader = DataLoader(
        dataset=sft_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    
    # Learning-rate scheduler
    num_training_steps = len(dataloader) * train_config.sft_epochs // train_config.gradient_accumulation
    if num_training_steps == 0: num_training_steps = 1

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    device = train_config.train_device

    global_step = 0
    # Init accumulative variable
    running_loss = 0.0
    running_entropy = 0.0
    running_len = 0.0
    
    optimizer.zero_grad()
    for _ in range(train_config.sft_epochs):
        progress_bar = tqdm(dataloader, desc=f"Training ({len(sft_dataset)} samples)")
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
    Evaluates the vLLM model after sft on the test set.
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

    output_path = result_path(eval_config.data_name, "ei", f"{current_label}.jsonl")
    acc = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, cots, groundtruths, sampling_params, output_path)
    return acc

@torch.no_grad()
def ei_obtain_useful_dataset(
    old_policy_model: "LLM",
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    groundtruths: List[str],
    train_config: TrainConfig
) -> Dataset:
    """
    """
    from cs336_alignment.math_baseline import get_response
    # Sample G outputs for each prompts
    from vllm import SamplingParams

    ei_sampling_params = SamplingParams(
        temperature=train_config.ei_temperature,
        max_tokens=train_config.ei_max_tokens,
        min_tokens=train_config.ei_min_tokens,
        n=train_config.G,
        seed=train_config.seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    logger.info(f"🎲 Generating {train_config.G} rollouts per question...")
    outputs:List[List[str]] = get_response(old_policy_model, prompts, ei_sampling_params)
    
    useful_prompts, useful_responses, useful_groundtruths = [], [], []
    total_generated = 0
    total_correct = 0
    # Compute rewards for each sampled output by running reward function
    # and filter out wrong outputs to obtain a sft dataset of correct prompt-response pairs
    for prompt, groundtruth, output_list in zip(prompts, groundtruths, outputs):
        for output in output_list:
            total_generated += 1
            metrics = reward_fn(output, groundtruth)
            if metrics.get("reward", 0.0) == 1:
                total_correct += 1
                useful_prompts.append(prompt)
                useful_responses.append(output)
                useful_groundtruths.append(groundtruth)
    sft_dataset = MathSFTDataset(useful_prompts, useful_responses, useful_groundtruths)
    logger.info(f"✅ Generated {total_generated} sequences, found {total_correct} correct (SR: {total_correct/total_generated:.2%})")
    return sft_dataset
    
# --- Main Experiment ---
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="Qwen2.5-Math-1.5B")
    parser.add_argument("--data_name",type=str,default="MATH-500")
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--ei_batchsize", type=int, default=1024)
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--exp_id", type=str, default="default", help="Identifier for this experiment run")
    args = parser.parse_args()

    global logger
    log_file_name = f"{args.exp_id}.log"
    logger = ExperimentLogger(log_file=str(log_path(args.data_name, "ei", log_file_name)))

    train_config = TrainConfig(
        model_name=args.model_name,
        data_name=args.data_name,
        G=args.G, 
        sft_epochs=args.sft_epochs, 
        ei_batchsize=args.ei_batchsize,
        train_device=args.train_device,
    )
    eval_config = EvalConfig(data_name=args.data_name, eval_device=args.eval_device)
    train_config.validate_paths()
    eval_config.validate_paths()
    require_vllm_cuda(eval_config.eval_device)
    
    run_name = f"{args.data_name}-EI-G{args.G}-E{args.sft_epochs}-B{args.ei_batchsize}"
    wandb.init(
        project="cs336-a5-expert_iteration_experiment",
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
    logger.info("⚙️ Initializing vLLM...") 
    llm = init__vllm(train_config.model_path, eval_config.eval_device, eval_config.seed)
    logger.info("✅ Initialized vLLM successfully.") 

    # Initialize Policy Model
    logger.info("⚙️ Initializing Policy Model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        train_config.model_path,
        **model_load_kwargs(train_config.train_device),
    ).to(train_config.train_device)
    policy_model.gradient_checkpointing_enable()
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("✅ Initialized Policy Model successfully.")

    # Expert Iteration Loop
    for n_ei_step in range(train_config.n_ei_steps):
        logger.info(f"\n{'='*10} EI Step {n_ei_step+1}/{train_config.n_ei_steps} {'='*10}")

        # clean fragmented memory
        gc.collect()
        clear_device_cache()

        # Sample ei_batchsize prompts(questions)
        current_seed = train_config.seed + n_ei_step
        _, prompts, _, groundtruths = load_and_format_prompts(
            train_config.data_path,
            train_config.prompt_template_path,
            train_config.ei_batchsize,
            current_seed,
        )
        
        # Obtain SFT Dataset
        sft_dataset = ei_obtain_useful_dataset(llm, r1_zero_reward_fn, prompts, groundtruths, train_config)

        # Train loop
        if len(sft_dataset) > 0:
            train_sft_model(policy_model, tokenizer, sft_dataset, train_config)

            # Set the old policy model
            load_policy_into_vllm_instance(policy_model, llm)

            # Save tokenzier and model
            save_path = checkpoint_path(args.data_name, "ei", args.exp_id, f"step_{n_ei_step+1}")
            save_tokenizer_and_model(policy_model, tokenizer, save_path)
        else:
            logger.warning("⚠️ No useful data collected in this step. Skipping training.")

        # Free up dataset memory
        del sft_dataset
        gc.collect()
        clear_device_cache()

        # Evaluate
        acc = evaluate_sft_model(llm, eval_config, f"{args.exp_id}_step{n_ei_step+1}", 500)
        wandb.log({
            "eval/step": n_ei_step + 1,
            "eval/accuracy": acc
        })
        logger.info(f"📊 Step {n_ei_step+1} Accuracy: {acc:.2%}")

    logger.info(f"🎉 EI Finished. Final Accuracy: {acc:.2%}")
    wandb.finish()

    if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
    # output_path = os.path.join(script_dir, f"results/gsm8k_sft_full.jsonl")
    # analyze_results(output_path)
