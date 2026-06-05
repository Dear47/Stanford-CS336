#%%
import json
import os
from typing import TYPE_CHECKING, Union, Callable, List, Dict, Tuple
import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.data_utils import *
from cs336_alignment.overwatch import ExperimentLogger
from cs336_alignment.paths import log_path, model_path, prompt_template_path, result_path, test_data_path

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

# dataname = "gsm8k" 
dataname = "MATH-500"
script_dir = os.path.dirname(os.path.abspath(__file__))
logger = ExperimentLogger(log_file=str(log_path(dataname, f"math_baseline_{dataname}.log")))
#%%
def get_response(vllm_model: "LLM", prompts:List[str], sampling_params: "SamplingParams") -> Union[List[str], List[List[str]]]:
    results = vllm_model.generate(prompts, sampling_params)
    is_multiple_samples = getattr(sampling_params, 'n', 1) > 1
    if is_multiple_samples:
        all_outputs = []
        for output in results:
            prompt_generations = [o.text.strip() for o in output.outputs]
            all_outputs.append(prompt_generations)
        return all_outputs
    else:
        outputs = [output.outputs[0].text.strip() for output in results]
        return outputs

def evaluate_vllm(
    vllm_model: "LLM",
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    cots: List[str],
    groundtruths: List[str],
    eval_sampling_params: "SamplingParams",
    output_filepath: str,
    save:bool = True,
) -> Dict:
    """
    Evaluate model on prompts, calulate metrics and serialize results to disk 
    """
    logger.info(f"🤖 Starting generation for {len(prompts)} prompts...")
    
    # 1. 批量生成 (vLLM 会自动处理批次)
    # responses 的顺序保证与 prompts 的输入顺序一致
    responses = get_response(vllm_model, prompts, eval_sampling_params)
    
    results = []
    correct_count = 0
    total_count = len(prompts)

    logger.info("🔢 Calculating metrics and serializing results...")

    # 2. 遍历结果并评分
    for prompt, cot, response, groundtruth in zip(prompts, cots, responses, groundtruths):
        ref_ans = extract_reference_answer(response)

        # 3. 计算奖励/分数
        metrics = reward_fn(response, groundtruth)

        # 统计正确率 (Answer Reward = 1 表示正确)
        if metrics.get("answer_reward", 0.0) == 1.0:
            correct_count += 1

        # 4. 构建结果对象
        result_entry = {
            "prompt": prompt,
            # "cot": cot,
            "response": response,
            "groundtruth": groundtruth,
            "reference_answer": ref_ans,
            "metrics": metrics
        }
        results.append(result_entry)

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    logger.info(f"📈 Evaluation Accuracy: {accuracy:.2%}")

    if save:
        # 5. 序列化结果到磁盘
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Results saved to {output_filepath}")

    return accuracy
    

def analyze_results(filepath:str):
    if not os.path.exists(filepath):
        logger.error(f"❌ File not found in {filepath}")
        return

    logger.info(f"📏 Analyze {filepath} ...")

    cat1_correct_format_correct_answer = 0
    cat2_correct_format_wrong_answer = 0
    cat3_wrong_format_wrong_answer = 0
    
    # 储存 10 个错误结果
    format_failures = []
    answer_failures = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            metrics = data.get('metrics', {})
            
            format_score = metrics.get('format_reward', 0)
            answer_score = metrics.get('answer_reward', 0) 
            
            if format_score == 1 and answer_score == 1:
                cat1_correct_format_correct_answer += 1
            elif format_score == 1 and answer_score == 0:
                cat2_correct_format_wrong_answer += 1
                if len(answer_failures) < 10:
                    answer_failures.append(data)
            elif format_score == 0:
                # 通常情况下 format_score = 0 也认为 answer_score = 0
                cat3_wrong_format_wrong_answer += 1
                if len(format_failures) < 10:
                    format_failures.append(data)


    logger.info(f"""
                {"="*40}
                STATISTICS
                {"="*40}
                (1) Correct Format + Correct Answer: {cat1_correct_format_correct_answer}
                (2) Correct Format + Wrong Answer:   {cat2_correct_format_wrong_answer}
                (3) Wrong Format + Wrong Answer:     {cat3_wrong_format_wrong_answer}
                Total: {cat1_correct_format_correct_answer + cat2_correct_format_wrong_answer + cat3_wrong_format_wrong_answer}""")

    logger.info(f"""
                {"="*40}
                ANALYSIS: FORMAT FAILURES (Format=0)
                Look for: Missing <think>/<answer> tags, empty generation, or chatty intro text.
                {"="*40}""")
    for i, item in enumerate(format_failures):
        logger.info(f"""
                    {"-"*20} Case {i+1} {"-"*20}
                    [Response]: {repr(item['response'])}
                    [GroundTruth]: {item['groundtruth']}""")

    logger.info(f"""
                {"="*40}
                ANALYSIS: ANSWER FAILURES (Format=1, Answer=0)
                Look for: Calculation errors, hallucination, or Parser matching issues (e.g. 1/2 vs 0.5).
                {"="*40}""")
    for i, item in enumerate(answer_failures):
        logger.info(f"""
                    {"-"*20} Case {i+1} {"-"*20}
                    [Response (Snippet)]: ...{item['response'][-100:]}
                    [GroundTruth]: {item['groundtruth']}""")

if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    MODEL_PATH = model_path("Qwen2.5-Math-1.5B")
    DATA_PATH = test_data_path(dataname)
    PROMPT_TEMPLATE_PATH = prompt_template_path()
    OUTPUT_PATH = result_path(f"{dataname}_baseline.jsonl")
    
    assert os.path.exists(MODEL_PATH), f"❗ Model file not found at {MODEL_PATH}. Please check the file."
    assert os.path.exists(DATA_PATH), f"❗ Data file not found at {DATA_PATH}. Please check the path."

    if not os.path.exists(PROMPT_TEMPLATE_PATH):
        logger.warning(f"⚠️ Prompt file not found at {PROMPT_TEMPLATE_PATH}. Please check the path.")
    
    sample_size = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载数据和格式化 Prompt
    datasets, prompts, cots, groundtruths = load_and_format_prompts(DATA_PATH, PROMPT_TEMPLATE_PATH)
    logger.info("✅ Loaded data Successfully!")

    # 2. 设置采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024, 
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    # 3. 初始化 vLLM 模型
    llm = LLM(model=str(MODEL_PATH), dtype="bfloat16", tensor_parallel_size=1, trust_remote_code=True)

    # 4. 运行评估流程
    evaluate_vllm(
        vllm_model = llm,
        reward_fn = r1_zero_reward_fn,
        prompts = prompts,
        cots = cots,
        groundtruths = groundtruths,
        eval_sampling_params = sampling_params,
        output_filepath = OUTPUT_PATH
    )

    # 5. 分析 Qwen 2.5 Math 1.5B Baseline
    analyze_results(OUTPUT_PATH)
