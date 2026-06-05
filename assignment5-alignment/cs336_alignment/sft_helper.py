#%%
import os
from typing import TYPE_CHECKING, List, Dict
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import logging
from cs336_alignment.paths import model_path

if TYPE_CHECKING:
    from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)
#%%
"""
huggingface 的 TRL 库有许多类似的实现，可供参考
"""

def tokenize_and_prompt_and_output(
        prompt_strs:List[str], 
        output_strs:List[str], 
        tokenizer: "PreTrainedTokenizer"
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes prompts and outputs and prepares training tensors.

    Args
    ----------
    prompt_strs : list[str]
        List of prompt strings.
    output_strs : list[str]
        List of output strings.
    tokenizer : PreTrainedTokenizer
        Tokenizer to use for tokenization.

    Returns
    -------
    input_ids : torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
        the tokenized prompt and output strings, with the final token sliced off.
    labels : torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
        shifted input ids, i.e., the input ids without the first token.
    response_mask : torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
        a mask on the response tokens in the labels.
    """
    # 确保 tokenizer 有 pad_token，如果没有则通常将其设为 eos_token 或 0
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    input_ids_list:List[List] = []
    mask_list:List[List] = []

    # 1. 逐条处理数据
    for prompt, output in zip(prompt_strs, output_strs):
        # Prompt 部分：add_special_tokens=True，负责添加句首的 BOS (如果有)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]  # (prompt_lens,)
        
        # Output 部分：add_special_tokens=False，避免在中间再次插入 BOS
        output_input_ids = tokenizer(output, return_tensors="pt", add_special_tokens=False).input_ids[0]  # (output_lens,)
        
        full_input_ids = torch.cat([prompt_input_ids, output_input_ids])  # (prompt_and_output_lens,)
        
        prompt_len = len(prompt_input_ids)
        
        # 2. 创建 Mask (0 表示 Prompt/Padding, 1 表示 Output/Response)
        # 初始化全 0
        mask = torch.zeros_like(full_input_ids)
        # 将 Response 部分设为 1。
        # Prompt 部分长度为 prompt_len，从 prompt_len 开始是 Output
        if len(full_input_ids) > prompt_len:
            mask[prompt_len:] = 1
            
        input_ids_list.append(full_input_ids)
        mask_list.append(mask)

    # 3. Padding 对齐
    # input_ids 使用 pad_token_id 填充
    padded_input_ids = pad_sequence(
        input_ids_list, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    
    # mask 使用 0 填充 (padding 部分也不计算 loss)
    padded_mask = pad_sequence(
        mask_list, 
        batch_first=True, 
        padding_value=0
    )

    # 4. 错位切片 (Shift)    
    input_ids = padded_input_ids[:, :-1]
    labels = padded_input_ids[:, 1:].clone()
    response_mask = padded_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Args
    ---------
    logits : torch.Tensor 
        Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits.

    Returns
    ---------
    entropy : torch.Tensor 
        Shape (batch_size, sequence_length). The entropy for each next-token prediction.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -torch.sum(probs * log_probs, dim=-1).to(logits.dtype)

def get_response_log_probs(
        model: "PreTrainedModel",
        input_ids: torch.Tensor, 
        labels: torch.Tensor, 
        return_token_entropy: bool = False, 
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities (given the previous tokens) from a causal language model,
    and optionally the entropy of the model’s next-token distribution.
    
    Args
    ---------
    model: PreTrainedModel 
        HuggingFace model used for scoring (placed on the correct device and in inference mode if gradients should not be computed).  
    input_ids: torch.Tensor 
        shape (batch_size, sequence_length), concatenated prompt + response tokens as produced by your tokenization method.  
    labels: torch.Tensor 
        shape (batch_size, sequence_length), labels as produced by your tokenization method.  
    return_token_entropy: bool 
        If True, also return per-token entropy by calling compute_entropy.  

    Returns
    ----------
    log_probs: torch.Tensor
        shape (batch_size, sequence_length), conditional log-probabilities log pθ(xt | x<t).  
    token_entropy: Optional[torch.Tensor]
        shape (batch_size, sequence_length), per-token entropy for each position (present only if return_token_entropy=True).
    """
    # 1. 前向传播获取 Logits
    # model 输出 shape: (batch_size, seq_len, vocab_size)
    logits = model(input_ids).logits

    # 2. 计算 Log Softmax
    # 我们先计算所有词表的 log_probs，shape: (batch_size, seq_len, vocab_size)
    all_log_probs = F.log_softmax(logits, dim=-1)

    # 3. 提取目标 Token (Labels) 对应的 Log Probability
    # labels shape: (batch_size, seq_len)
    # 我们需要从最后一维 (vocab) 中 gather 出 labels 指定的那个 token 的概率。
    
    # 【防御性编程】处理 padding 或 ignore_index (-100)
    gather_indices = labels.clone()
    gather_indices[gather_indices == -100] = 0

    # torch.gather 要求 index 维度与 input 一致，所以需要 unsqueeze 最后一维
    # input: (B, L, V), index: (B, L, 1) -> output: (B, L, 1)
    token_log_probs = torch.gather(
        all_log_probs, 
        dim=-1, 
        index=gather_indices.unsqueeze(-1)
    ).squeeze(-1) # 变回 (B, L)

    result = {
        "log_probs": token_log_probs
    }

    # 4. Optional：计算熵
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result

def masked_normalize(
        tensor: torch.Tensor, 
        mask: torch.Tensor, 
        normalize_constant: float = 1.0, 
        dim: int | None = None, 
) -> torch.Tensor:
    """
    Sum over tensor elements and normalizes by a constant while respecting a boolean mask.

    Args
    ------------
    tensor: torch.Tensor 
        The tensor to sum and normalize.  
    mask: torch.Tensor 
        Same shape as tensor; positions with 1 are included in the sum.  
    normalize_constant: float 
        the constant to divide by for normalization.  
    dim: int | None 
        the dimension to sum along before normalization. If None, sum over all dimensions.

    Returns
    ------------
    torch.Tensor 
        the normalized sum, where masked elements (mask == 0) don’t contribute to the sum.
    """
    tensor_sum = torch.sum(tensor * mask, dim=dim)
    return tensor_sum / normalize_constant

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch for SFT.

    Args
    ----------
    policy_log_probs: Tensor
        shape (batch_size, sequence_length), per-token log-probilities from the SFT policy being trained.
    response_mask: Tensor
        shape (batch_size, sequence_length), 1 for sequence tokens, 0 for prompt/padding.
    gradient_accumulation_steps: int
        Number of microbatches per optimizer step.
    normalize_constant: float
        The constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns
    ----------
    loss: Scalar Tensor
        The microbatch loss, adjusted for gradient accmulation. We return this so we can log it.
    metadata: dict
        Dict with metadata from the underlying loss call, and any other statistics you might want to log.
    """
    metadata = {}
    num_response_tokens = torch.sum(response_mask)
    metadata["num_response_tokens"] = num_response_tokens.detach()

    # 1. 计算负对数似然
    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant)

    # 2. 梯度累积和反传
    # 按batch_size平均！
    loss = -loss / policy_log_probs.size(0) / gradient_accumulation_steps
    loss.backward()
    metadata["sft_loss"] = loss.detach()

    return loss, metadata

def save_tokenizer_and_model(
    model: torch.nn.Module, 
    tokenizer: "AutoTokenizer",
    output_dir: str
):
    """
    Save model weights and tokenizer to specific dir
    """
    logger.info(f"💾 Saving model and tokenizer to {output_dir}...")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save model
        # safe_serialization=True 会保存为 .safetensors 格式（加载速度更快且更安全）
        model.save_pretrained(output_dir, safe_serialization=True)
        
        # 2. Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✅ Model and tokenizer saved successfully at {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        raise e

if __name__ =='__main__':
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = model_path("Qwen2.5-Math-1.5B")

    prompt_strs = [
        "Hello, world!",
        "This is a test.",
        "This is another test.",
    ]

    output_strs = [
        "Hello, world!",
        "This is a test.",
        "This is another test.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    
    input_ids, labels, response_mask = tokenize_and_prompt_and_output(prompt_strs, output_strs, tokenizer)
