from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cs336_alignment.sft_helper import get_response_log_probs, tokenize_and_prompt_and_output


class PackedSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def get_packed_sft_dataset(tokenizer, dataset_path: str | Path, seq_length: int, shuffle: bool) -> Dataset:
    with open(dataset_path, encoding="utf-8") as f:
        documents = [json.loads(line) for line in f if line.strip()]

    if shuffle:
        rng = random.Random(42)
        rng.shuffle(documents)

    token_ids: list[int] = []
    eos_token_id = tokenizer.eos_token_id
    for doc in documents:
        text = f"{doc['prompt']}\n{doc['response']}"
        token_ids.extend(tokenizer.encode(text, add_special_tokens=True))
        if eos_token_id is not None:
            token_ids.append(eos_token_id)

    examples = []
    for start in range(0, len(token_ids) - seq_length, seq_length):
        chunk = torch.tensor(token_ids[start : start + seq_length + 1], dtype=torch.long)
        if chunk.numel() != seq_length + 1:
            continue
        examples.append({"input_ids": chunk[:-1].clone(), "labels": chunk[1:].clone()})

    return PackedSFTDataset(examples)


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    option_letters = "ABCD"[: len(mmlu_example.get("options", []))]
    patterns = [
        r"(?:correct answer is|answer is|answer:)\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?(?:[.\):\s]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, model_output, flags=re.IGNORECASE)
        if match:
            choice = match.group(1).upper()
            return choice if choice in option_letters else None
    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    matches = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _sequence_log_prob(model, tokenizer, prompt: str, response: str) -> torch.Tensor:
    tokenized = tokenize_and_prompt_and_output([prompt], [response], tokenizer)
    device = next(model.parameters()).device
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)
    log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
    return (log_probs * response_mask).sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    policy_chosen = _sequence_log_prob(lm, tokenizer, prompt, response_chosen)
    policy_rejected = _sequence_log_prob(lm, tokenizer, prompt, response_rejected)
    with torch.no_grad():
        ref_chosen = _sequence_log_prob(lm_ref, tokenizer, prompt, response_chosen)
        ref_rejected = _sequence_log_prob(lm_ref, tokenizer, prompt, response_rejected)

    policy_log_ratio = policy_chosen - policy_rejected
    ref_log_ratio = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (policy_log_ratio - ref_log_ratio))
