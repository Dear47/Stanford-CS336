#%%
import numpy as np
import pathlib
import torch
from cs336_basics.transformer.transformer_utils import load_checkpoint
from cs336_basics.transformer.module import TransformerLM
from cs336_basics.tokenizer.bpetokenizer import BPETokenizer
PATH = pathlib.Path(__file__).resolve().parent.parent
#%%
if __name__ == "__main__":
    prompts = ["The quick brown fox jumps over the lazy dog",
               "Once upon a time,",
               "Tom and Lily are best friends.",]
    
    max_new_tokens = 128
    temperature = 1.2
    top_k = 3
    top_p = 0.9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    context_length = 256

    end_token = "<|endoftext|>"

    tokenizer = BPETokenizer.from_files(
        vocab_path=PATH/"data/vocab/TinyStoriesV2-GPT4_vocab.json",
        merges_path=PATH/"data/merges/TinyStoriesV2-GPT4_merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    model = TransformerLM(
        vocab_size=10000,
        context_length=context_length,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        theta=10000,
        device=device
    )

    inputs_ids = []
    len_inputs_ids = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.int32).to(device)
        inputs_ids.append(input_ids)
        len_inputs_ids.append(len(input_ids))
    
    len_inputs_ids = torch.tensor(len_inputs_ids, dtype=torch.int64, device=device)  # (batch_size,)
    eos_token_id = tokenizer.encode(end_token)[0]

    _ = load_checkpoint(PATH/"data/model/final_model_v1.pt",model,None)
    model.to(device)
    model.eval()

    output_texts = []
    for ids in inputs_ids:
        generate_token_ids=model.generate(
            x=ids,
            max_new_token=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id)
        output_ids = generate_token_ids.cpu().numpy()[0]
        output_texts.append(tokenizer.decode(output_ids))
    
    for i,text in enumerate(output_texts):
        print(f"PROMPT{i}:\n{prompts[i]+text}")
