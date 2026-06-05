#%%
import os
from typing import Callable, List, Dict, Tuple, Literal
import torch
import logging

from cs336_alignment.sft_helper import masked_normalize

script_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

#%%
def compute_group_normalized_rewards(
        reward_fn:Callable[[str, str], Dict[str, float]],
        rollout_responses:List[str],
        repeated_ground_truths:List[str],
        group_size:int,
        advantage_eps:float,
        normalize_by_std:bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args
    --------
    reward_fn: Callable[[str, str], dict[str, float]]
        Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
    rollout_responses: list[str] 
        Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    repeated_ground_truths: list[str] 
        The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
    group_size: int 
        Number of responses per question (group).
    advantage_eps: float 
        Small constant to avoid division by zero in normalization.
    normalize_by_std: bool 
        If True, divide by the per-group standard deviation; otherwise subtract only the group mean.
    
    Returns
    --------
    advantages: torch.Tensor 
        shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
    raw_rewards: torch.Tensor 
        shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
    metadata: dict[str, float]
        your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    if group_size <= 1:
        raise ValueError("group_size must be greater than 1 for group-normalized rewards")

    rewards = []
    for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(rollout_response, ground_truth)["reward"]
        rewards.append(reward)

    raw_rewards = torch.tensor(rewards)  # rollout_batch_size = n_prompts_per_rollout_batch * group_size
    raw_rewards_per_group = raw_rewards.reshape((-1, group_size))  # (n_prompts_per_rollout_batch, group_size)
    advantages =  raw_rewards_per_group - raw_rewards_per_group.mean(-1,keepdim=True) # (n_prompts_per_rollout_batch, group_size)
    if normalize_by_std:
        advantages /= (advantage_eps + raw_rewards_per_group.std(-1, keepdim=True))
    
    advantages = advantages.flatten()
    metadata = {
        'reward_mean': raw_rewards.mean().item(),
        'reward_std': raw_rewards.std().item(),
        # 'reward_max': raw_rewards.max().item(),
        # 'reward_min': raw_rewards.min().item(), 
    }
    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
        raw_reward_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, 
    where `raw_rewards_or_advantages` is either the raw reward or an already-normalized advantage
    
    Args
    ----------
    raw_rewards_or_advantages: torch.Tensor 
        Shape (batch_size, 1), scalar reward/advantage for each rollout response.
    policy_log_probs: torch.Tensor 
        Shape (batch_size, sequence_length), logprobs for each token.

    Returns
    ----------
    torch.Tensor 
        Shape (batch_size, sequence_length), the per-token policy-gradient loss (to be aggregated across the batch and sequence dimensions in the training loop).
    """
    return -raw_reward_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange:float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Args
    ---------
    advantages: torch.Tensor 
        Shape (batch_size, 1), per-example advantages A.
    policy_log_probs: torch.Tensor 
        Shape (batch_size, sequence_length), per-token log probs from the policy being trained.
    old_log_probs: torch.Tensor 
        Shape (batch_size, sequence_length), per-token log probs from the old policy.
    cliprange: float 
        Clip parameter ε (e.g. 0.2).
    
    Returns
    ----------
    loss: torch.Tensor 
        torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss. 
    metadata: dict[str, torch.Tensor]
        dict containing whatever you want to log. 
        We suggest logging whether each token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of the min was lower than the LHS.
    """
    log_ratio = policy_log_probs - old_log_probs
    # log_ratio = torch.clip(log_ratio, -10, 10)
    ratio = torch.exp(log_ratio)
    unclipped_obj = ratio * advantages
    clipped_ratio = torch.clip(ratio, 1-cliprange, 1+cliprange)
    clipped_obj = clipped_ratio * advantages
    loss = -torch.min(unclipped_obj, clipped_obj)
    with torch.no_grad():
        adv_pos = advantages > 0
        adv_neg = advantages < 0
        # when advantage > 0, clipped if ratio > 1+cliprange
        clipped_pos = adv_pos & (ratio > (1 + cliprange))
        # when advantage < 0, clipped if ratio < 1-cliprange
        clipped_neg = adv_neg & (ratio < (1 - cliprange))
        # merge mask
        clipped_mask = clipped_pos | clipped_neg
        # 计算 fraction: 被 clip 的 token 总数 / 总 token 数
        clip_fraction = clipped_mask.float().mean()
    metadata = {
        'clip_fraction': clip_fraction,
        'clipped_mask': clipped_mask,
    }
    return loss, metadata

def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None, 
        advantages: torch.Tensor | None = None, 
        old_log_probs: torch.Tensor | None = None, 
        cliprange: float | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args
    ----------
    policy_log_probs: torch.Tensor 
        shape (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
    loss_type: Literal[str]
        One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
    raw_rewards: torch.Tensor | None
        Required if loss_type == "no_baseline"; shape (batch_size, 1).
    advantages: torch.Tensor | None
        Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
    old_log_probs: torch.Tensor | None
        Required for "grpo_clip"; shape (batch_size, sequence_length).
    cliprange: float | None
        Required for "grpo_clip"; scalar ε used for clipping.     
    Returns
    ----------
    loss: torch.Tensor
        shape (batch_size, sequence_length), per-token loss.  
    metadata: dict[str, torch.Tensor]
        statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    raise ValueError(f"Unknown loss_type: {loss_type}")

def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where `mask == 1`.
    
    Args
    ---------
    tensor: torch.Tensor
        The data to be averaged.  
    mask: torch.Tensor 
        Same shape as tensor; positions with 1 are included in the mean.  
    dim: int | None 
        Dimension over which to average. If `None`, compute the mean over all masked elements.
    
    Returns
    ---------
    torch.Tensor 
        The masked mean; shape matches tensor.mean(dim) semantics.
    """
    mask_bool = mask.bool()
    masked_tensor = torch.where(mask_bool, tensor, torch.zeros_like(tensor))
    denominator = mask.sum(dim) if dim is not None else mask.sum()
    return masked_tensor.sum(dim) / denominator

def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        loss_normalization: Literal["token_mean", "constant"] = "token_mean",
        normalize_constant: float | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args
    ---------
    policy_log_probs: torch.Tensor
        (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
    response_mask: torch.Tensor
        (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
    gradient_accumulation_steps: int
        Number of microbatches per optimizer step.
    loss_type: Literal[str] 
        One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
    raw_rewards: torch.Tensor
        Needed when loss_type == "no_baseline"; shape (batch_size, 1).
    advantages: torch.Tensor
        Needed when loss_type != "no_baseline"; shape (batch_size, 1).    
    old_log_probs: torch.Tensor
        Required for GRPO-Clip; shape (batch_size, sequence_length).
    cliprange: float
        Clip parameter ε for GRPO-Clip.
    
    Returns
    ---------
    loss scalar: torch.Tensor
        The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
    metadata: Dict[str, torch.Tensor]
        metadata from the underlying loss call, and any other statistics you might want to log.
    """
    policy_gradient_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    if loss_normalization == "token_mean":
        loss = masked_mean(policy_gradient_loss, response_mask)
    elif loss_normalization == "constant":
        if normalize_constant is None:
            normalize_constant = policy_log_probs.shape[1]
        per_example_loss = masked_normalize(
            policy_gradient_loss,
            response_mask,
            normalize_constant=normalize_constant,
            dim=1,
        )
        loss = per_example_loss.mean()
    else:
        raise ValueError(f"Unknown loss_normalization: {loss_normalization}")

    clipped_mask = metadata.pop("clipped_mask", None)
    if clipped_mask is not None:
        metadata["clip_fraction"] = masked_mean(clipped_mask.float(), response_mask)

    loss /=  gradient_accumulation_steps  # batchsize 在 masked_mean 中已经被 average 了
    # metadata["policy_log_probs_grad"] = policy_log_probs.grad

    loss.backward()
    metadata["grpo_loss"] = loss.detach()
    
    num_response_tokens = torch.sum(response_mask)
    metadata["num_response_tokens"] = num_response_tokens.detach()

    return loss, metadata
