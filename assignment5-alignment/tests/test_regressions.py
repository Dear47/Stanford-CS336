import math
import types

import pytest
import torch

from cs336_alignment import grpo_helper, sft_helper
from tests import adapters


def test_compute_entropy_is_finite_for_large_logits():
    logits = torch.tensor([[[1000.0, 999.0, 998.0]]])

    entropy = sft_helper.compute_entropy(logits)

    assert torch.isfinite(entropy).all()


def test_group_normalized_rewards_zero_variance_has_zero_advantages():
    def reward_fn(response, ground_truth):
        return {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}

    advantages, raw_rewards, _ = grpo_helper.compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=["a", "b", "c", "d"],
        repeated_ground_truths=["x", "x", "y", "y"],
        group_size=2,
        advantage_eps=1e-6,
        normalize_by_std=True,
    )

    assert torch.equal(raw_rewards, torch.ones(4))
    assert torch.equal(advantages, torch.zeros(4))


def test_grpo_microbatch_supports_constant_length_normalization():
    policy_log_probs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    advantages = torch.ones((2, 1))

    loss, _ = grpo_helper.grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=1,
        loss_type="reinforce_with_baseline",
        advantages=advantages,
        loss_normalization="constant",
        normalize_constant=3.0,
    )

    expected = -((1.0 + 2.0) / 3.0 + (4.0 + 5.0 + 6.0) / 3.0) / 2.0
    assert torch.isclose(loss, torch.tensor(expected))


def test_grpo_adapter_passes_constant_length_normalization():
    policy_log_probs = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    advantages = torch.ones((1, 1))

    loss, _ = adapters.run_grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=1,
        loss_type="reinforce_with_baseline",
        advantages=advantages,
        loss_normalization="constant",
        normalize_constant=3.0,
    )

    assert torch.isclose(loss, torch.tensor(-1.0))


def test_grpo_clip_fraction_uses_response_mask_only():
    policy_log_probs = torch.log(torch.tensor([[10.0, 10.0, 1.0, 1.0]], requires_grad=True))
    old_log_probs = torch.zeros_like(policy_log_probs)
    response_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    advantages = torch.ones((1, 1))

    _, metadata = grpo_helper.grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=1,
        loss_type="grpo_clip",
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=0.2,
    )

    assert metadata["clip_fraction"].item() == 0.0


def test_sft_train_loop_iterates_dataloader_batches(monkeypatch):
    import cs336_alignment.sft_experiment as sft_experiment

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return "prompt", "response", "gt"

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids):
            logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            return types.SimpleNamespace(logits=logits + self.weight)

    def fake_tokenize(prompts, cots, tokenizer):
        return {
            "input_ids": torch.tensor([[0, 1]]),
            "labels": torch.tensor([[1, 2]]),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        }

    def fake_log_probs(model, input_ids, labels, return_token_entropy=False):
        log_probs = model.weight.expand_as(input_ids).float()
        return {"log_probs": log_probs, "token_entropy": torch.zeros_like(log_probs)}

    monkeypatch.setattr(sft_experiment, "tokenize_and_prompt_and_output", fake_tokenize)
    monkeypatch.setattr(sft_experiment, "get_response_log_probs", fake_log_probs)
    monkeypatch.setattr(sft_experiment.wandb, "log", lambda *args, **kwargs: None)

    config = sft_experiment.TrainConfig(
        model_name="unused",
        data_name="unused",
        batch_size=1,
        gradient_accumulation=1,
        train_device="cpu",
        epochs=1,
    )
    config.model_path = "."
    config.data_path = "."
    config.prompt_template_path = "."

    sft_experiment.train_sft_model(TinyModel(), object(), TinyDataset(), config)


def test_grpo_train_loop_reuses_optimizer(monkeypatch):
    import cs336_alignment.grpo_experiment as grpo_experiment

    class CountingAdamW(torch.optim.SGD):
        init_count = 0

        def __init__(self, params, *args, **kwargs):
            CountingAdamW.init_count += 1
            super().__init__(params, lr=kwargs.get("lr", 0.1))

    monkeypatch.setattr(grpo_experiment.torch.optim, "AdamW", CountingAdamW)

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    assert CountingAdamW.init_count == 1

    # The train function should accept and use the existing optimizer instead of
    # constructing another one internally.
    assert "optimizer" in grpo_experiment.train_grpo_model.__code__.co_varnames
    assert "scheduler" in grpo_experiment.train_grpo_model.__code__.co_varnames


def test_grpo_train_loop_runs_collate_and_step_on_cpu(monkeypatch):
    import cs336_alignment.grpo_experiment as grpo_experiment

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = [1] if add_special_tokens else []
            ids.extend([2] * max(len(text.split()), 1))
            return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids):
            logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            return types.SimpleNamespace(logits=logits + self.weight)

    monkeypatch.setattr(grpo_experiment.wandb, "log", lambda *args, **kwargs: None)

    dataset = grpo_experiment.GRPODataset(
        prompts=["p0", "p1"],
        cots=["r0", "r1 longer"],
        groundtruths=["g0", "g1"],
        advantages=torch.ones(2),
        raw_rewards=torch.ones(2),
        old_log_probs=torch.zeros((2, 8)),
    )
    config = grpo_experiment.TrainConfig(
        model_name="unused",
        data_name="unused",
        rollout_batch_size=2,
        group_size=2,
        train_batch_size=2,
        gradient_accumulation_steps=1,
        train_device="cpu",
        loss_type="reinforce_with_baseline",
    )
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    _, global_step = grpo_experiment.train_grpo_model(
        model,
        TinyTokenizer(),
        dataset,
        config,
        global_step=0,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert global_step == 1


def test_experiment_model_load_kwargs_are_cpu_safe():
    import cs336_alignment.expert_iteration_experiment as expert_iteration_experiment
    import cs336_alignment.grpo_experiment as grpo_experiment
    import cs336_alignment.sft_experiment as sft_experiment

    assert sft_experiment.can_use_vllm("cpu") is False

    for module in (sft_experiment, expert_iteration_experiment, grpo_experiment):
        cpu_kwargs = module.model_load_kwargs("cpu")
        assert "attn_implementation" not in cpu_kwargs
        assert "torch_dtype" not in cpu_kwargs

        cuda_kwargs = module.model_load_kwargs("cuda")
        assert cuda_kwargs["attn_implementation"] == "flash_attention_2"
        assert cuda_kwargs["torch_dtype"] is torch.bfloat16

    with pytest.raises(RuntimeError, match="requires vLLM on CUDA"):
        expert_iteration_experiment.require_vllm_cuda("cpu")
    with pytest.raises(RuntimeError, match="requires vLLM on CUDA"):
        grpo_experiment.require_vllm_cuda("cpu")


def test_load_and_format_prompts_detects_schema_not_path(tmp_path):
    from cs336_alignment.data_utils import load_and_format_prompts

    data_path = tmp_path / "renamed.jsonl"
    prompt_path = tmp_path / "prompt.txt"
    data_path.write_text('{"problem":"1+1?","solution":"2","answer":"2"}\n', encoding="utf-8")
    prompt_path.write_text("Question: {question}", encoding="utf-8")

    _, prompts, cots, groundtruths = load_and_format_prompts(str(data_path), str(prompt_path))

    assert prompts == ["Question: 1+1?"]
    assert cots == ["2 </think> <answer> 2 </answer>"]
    assert groundtruths == ["2"]
