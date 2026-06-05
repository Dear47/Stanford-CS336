#%%
import os
import json
import regex as re
import random
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)
#%%
def load_jsonl_to_list(file_path: str) -> List[Dict]:
    data_list = []

    if not os.path.exists(file_path):
        logger.error(f"❌ File {file_path} not found.")
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    return data_list

def wrap_prompt(text: str, prompt_template_path: str):
    """
    Wrap the text by the provided prompt
    """
    if not os.path.exists(prompt_template_path):
        logger.error(f"❌ File {prompt_template_path} not found.")

    with open(prompt_template_path, "r") as file:
        prompt = file.read()
    return prompt.format(question=text)

def format_cot_to_think_answer(text: str) -> str:
    """
    Convert the chain-of-thought style answer 
    to the format: "{cot}</think> <answer>{ans}</answer>"
    """
    # try to capture the number behind four delimiters at the end of text
    m = re.search(r"####\s*([^\n]+)\s*$", text)
    if m:
        ans = m.group(1).strip()
        cot = text[: m.start()].strip()
        return f"{cot} </think> <answer>{ans}</answer>"
    
    # try to capture a trailing number at the end of text
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", text)
    if m:
        ans = m.group(1)
        cot = text[: m.start()].rstrip()
        return f"{cot} </think> <answer>{ans}</answer>"
    
    logger.warning(f"⚠️ Can't format CoT, return text directly!")
    return text

def extract_gsm8k_groundtruth(answer: str) -> str:
    """
    Extract the groundtruth from gsm8k dataset
    """
    ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    
    logger.warning(f"⚠️ Can't extract Gsm8k GroundTruth, return [invalid]")
    return "[invalid]"

def extract_MATH500_groundtruth(answer: str) -> str:
    """
    Extract the groundtruth from MATH-500 dataset
    """
    return answer

def load_and_format_prompts(
        data_path: str, 
        prompt_template_path: str,
        sample_size: int = -1,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[str], List[str], List[str]]:
    """
    Load JSONL and prompt template and return four lists:
    raw datasets, formatted prompts, formatted chain-of-thoughts, gsm8k groundtruths
    """
    prompts = []
    cots = []
    groundtruths = []

    all_data = load_jsonl_to_list(data_path)
    if sample_size > 0 and sample_size < len(all_data):
        random.seed(seed)
        datasets = random.sample(all_data, sample_size)
        logger.info(f"Randomly sampled {len(datasets)} datas.")
    else:
        datasets = all_data
        logger.info(f"Using all {len(datasets)} datas.")

    for entry in datasets:
        try:
            if {"problem", "solution", "answer"}.issubset(entry):
                groundtruths.append(extract_MATH500_groundtruth(entry["answer"]))
                cots.append(f"{entry['solution']} </think> <answer> {entry['answer']} </answer>")
                prompts.append(wrap_prompt(entry["problem"], prompt_template_path))
            elif {"question", "answer"}.issubset(entry):
                groundtruths.append(extract_gsm8k_groundtruth(entry["answer"]))
                cots.append(format_cot_to_think_answer(entry["answer"]))
                prompts.append(wrap_prompt(entry["question"], prompt_template_path))
            else:
                logger.error(f"Entry has unsupported schema keys: {sorted(entry.keys())}")
        except KeyError as e:
            logger.error(f"Entry contains unknown placeholder {e}")

    return datasets, prompts, cots, groundtruths

def extract_reference_answer(response: str) -> str:
    """
    extract the reference answer from the LLM response
    """
    from cs336_alignment.drgrpo_grader import extract_answer

    model_answer = response.split("<answer>")[-1].replace("</answer>", "")
    if "\\boxed" in model_answer:
        model_answer = extract_answer(model_answer)

    # logger.warning("⚠️ Can't extract reference answer, return model answer directly!")
    return model_answer

if __name__ == "__main__":
    # dataname = "gsm8k"
    dataname = 'MATH-500'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(os.path.dirname(script_dir), f"data/{dataname}/test.jsonl")
    PROMPT_TEMPLATE_PATH = os.path.join(script_dir, "prompts/r1_zero.prompt")
    data = load_jsonl_to_list(DATA_PATH)
    print("data:",data[0])
    # gt = extract_gsm8k_groundtruth(data[0]["answer"])
    gt = extract_MATH500_groundtruth(data[0]['answer'])
    print("gt:",gt)
    print(type(gt))

    print(wrap_prompt(data[0]["problem"], PROMPT_TEMPLATE_PATH))
    print(f"{data[0]['solution']} </think> <answer> {data[0]['answer']} </answer>")
