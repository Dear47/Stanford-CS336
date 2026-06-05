import json
import logging
import math

import torch
import pytest

from .adapters import get_packed_sft_dataset, run_iterate_batches
from .common import FIXTURES_PATH

logger = logging.getLogger(__name__)


@pytest.fixture
def llama_tokenizer():
    tokenizer_path = FIXTURES_PATH / "Meta-Llama-3-8B"
    if not tokenizer_path.exists():
        pytest.skip("Meta-Llama tokenizer fixture is unavailable")

    try:
        from transformers import AutoTokenizer
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"transformers is unavailable in this environment: {exc}")
    return AutoTokenizer.from_pretrained(tokenizer_path)


def test_packed_sft_dataset(llama_tokenizer):
    sft_sample_path = FIXTURES_PATH / "sft_sample.jsonl"
    tokenized_sample_path = FIXTURES_PATH / "tokenized_sft_sample.json"
    if not sft_sample_path.exists() or not tokenized_sample_path.exists():
        pytest.skip("packed SFT fixtures are unavailable")

    tokenizer = llama_tokenizer
    seq_length = 32
    packed_sft_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=False,
    )

    with open(tokenized_sample_path) as f:
        expected_examples = json.load(f)

    assert len(packed_sft_dataset) == len(expected_examples)

    for example, expected_example in zip(packed_sft_dataset, expected_examples):
        assert example["input_ids"].tolist() == expected_example["input_ids"]
        assert example["labels"].tolist() == expected_example["labels"]

    # Check that shuffling works by ensuring that it returns different data
    # than the unshuffled dataset. Note that there's a small chance that it
    # just happens that the shuffling preserves the original order.
    shuffled_packed_sft_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=True,
    )
    all_unshuffled_examples = []
    all_shuffled_examples = []
    for example, shuffled_example in zip(
        packed_sft_dataset, shuffled_packed_sft_dataset
    ):
        all_unshuffled_examples.append({k: v.tolist() for k, v in example.items()})
        all_shuffled_examples.append(
            {k: v.tolist() for k, v in shuffled_example.items()}
        )
    assert all_unshuffled_examples != all_shuffled_examples


def test_iterate_batches(llama_tokenizer):
    sft_sample_path = FIXTURES_PATH / "sft_sample.jsonl"
    if not sft_sample_path.exists():
        pytest.skip("SFT sample fixture is unavailable")

    tokenizer = llama_tokenizer
    seq_length = 32
    batch_size = 8
    packed_sft_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=True,
    )
    train_dataloader = run_iterate_batches(
        dataset=packed_sft_dataset, batch_size=batch_size, shuffle=True
    )
    assert len(train_dataloader) == math.ceil(75 / batch_size)
    for batch_idx, batch in enumerate(train_dataloader):
        # Make sure each of input_ids and labels is a (batch_size, seq_length) tensor, except
        # for the last batch (which can be less than batch_size items)
        if batch_idx != len(train_dataloader) - 1:
            assert batch["input_ids"].shape == (batch_size, seq_length)
            assert batch["labels"].shape == (batch_size, seq_length)

        assert (
            batch["input_ids"].dtype == torch.long
            or batch["input_ids"].dtype == torch.int64
        )
        assert (
            batch["labels"].dtype == torch.long or batch["labels"].dtype == torch.int64
        )
