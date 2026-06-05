import torch
import pytest

from .adapters import run_compute_per_instance_dpo_loss as compute_per_instance_dpo_loss
from .common import FIXTURES_PATH


def test_per_instance_dpo_loss():
    tiny_model_path = FIXTURES_PATH / "tiny-gpt2"
    tiny_ref_path = FIXTURES_PATH / "tiny-gpt2-ref"
    if not tiny_model_path.exists() or not tiny_ref_path.exists():
        pytest.skip("tiny GPT-2 fixtures are unavailable")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"transformers is unavailable in this environment: {exc}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = AutoModelForCausalLM.from_pretrained(tiny_model_path)
    model_ref = AutoModelForCausalLM.from_pretrained(tiny_ref_path)

    prompt = "The quick brown fox jumps over"
    good_response = "the lazy dog."
    bad_response = "their crazy frog."

    loss = compute_per_instance_dpo_loss(
        lm=model,
        lm_ref=model_ref,
        tokenizer=tokenizer,
        beta=0.5,
        prompt=prompt,
        response_chosen=good_response,
        response_rejected=bad_response,
    )

    assert torch.isclose(loss, torch.tensor(0.5785), atol=1e-4)
