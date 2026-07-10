"""
Image Quality Assessment (IQA) Reward Function for GRPO Training.

This module implements reward functions for evaluating model responses
in the IQA task, including:
1. Format reward: 1 only when format is completely correct, else 0
2. Accuracy reward: Exponential decay reward based on MAE (configurable decay_coef)
3. Rank reward (optional): Batch-level ranking consistency (Spearman) between
   predicted scores and ground truth scores
4. Repetition penalty: Penalize repeated calls to the same tool in one response
5. Tool usage reward: Reward for correctly calling tools; penalize excessive calls
"""

import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# 训练过程中保存生成结果的配置
# ---------------------------------------------------------------------------
# 通过环境变量控制是否保存生成结果，默认不保存
SAVE_GENERATIONS = os.environ.get("IQA_SAVE_GENERATIONS", "0") == "1"
# 保存路径，默认为当前目录下的 generations.jsonl
GENERATIONS_SAVE_PATH = os.environ.get(
    "IQA_GENERATIONS_SAVE_PATH",
    os.path.join(os.path.dirname(__file__), "generations.jsonl")
)
# 全局计数器，用于给每条记录编号
_generation_counter = 0


# Metadata
REWARD_NAME = "iqa"
REWARD_TYPE = "batch"


# ---------------------------------------------------------------------------
# IQA 工具名称列表（用于工具调用识别，与 tools.py 的 ALL_TOOL_NAMES 保持一致）
# ---------------------------------------------------------------------------
_ALL_TOOL_NAMES = [
    "GradientMagnitudeMap",
    "GradientOrientationCoherenceMap",
    "GradientMagnitudeHistogram",
    "HighFrequencyResidualMap",
    "DoGSharpnessMap",
    "FourierMagnitudeSpectrum",
    "NoiseResidualMap",
    "LuminanceHistogram",
    "ExtremeLuminanceMap",
    "PerceptualColorfulnessMap",
    "ChromaticDeviationMap",
    "ColorSaturationClippingMap",
    "MSCNNormalizedLuminanceMap",
    "MSCNDistributionHistogram",
    "LocalGradientDeviationMap",
]


def extract_tool_calls(response: str) -> list[str]:
    """
    从模型回复中按出现顺序提取所有工具调用名称（带尖括号标签形式）。
    
    工具调用格式：<ToolName>，其中 ToolName 必须是已知 IQA 工具之一。
    
    Args:
        response: 模型输出字符串
        
    Returns:
        按出现顺序排列的工具标签列表（可能包含重复项）
    """
    calls: list[tuple[int, str]] = []
    for tool_name in _ALL_TOOL_NAMES:
        tag = f"<{tool_name}>"
        start = 0
        while True:
            pos = response.find(tag, start)
            if pos == -1:
                break
            calls.append((pos, tag))  # 保存完整的工具标签 <ToolName>
            start = pos + len(tag)
    # 按出现位置排序，保证顺序正确
    calls.sort(key=lambda x: x[0])
    return [name for _, name in calls]


def extract_score(response: str) -> float | None:
    """
    Extract the predicted score from model response.
    
    Expected format: <answer_start>{"score": X.XX}<answer_end>
    
    Args:
        response: Model's response string
        
    Returns:
        Extracted score as float, or None if extraction fails
    """
    # Try to match the score pattern
    patterns = [
        r'<answer_start>\s*\{\s*"score"\s*:\s*(\d+\.?\d*)\s*\}\s*<answer_end>',
        r'<answer_start>\s*\{\s*\'score\'\s*:\s*(\d+\.?\d*)\s*\}\s*<answer_end>',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                score = float(match.group(1))
                # Validate score is in reasonable range (1-5 for IQA)
                if 1.0 <= score <= 5.0:
                    return score
            except (ValueError, IndexError):
                continue
    
    return None


def format_reward(response: str) -> float:
    """
    Check if the response follows the required format.
    
    Required format:
    <think>...</think>
    <answer_start>{"score": X.XX}<answer_end>
    
    Args:
        response: Model's response string
        
    Returns:
        1.0 only when format is completely correct, 0.0 otherwise (no partial credit).
    """
    # Remove extra whitespace for matching
    response_cleaned = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    
    # Check for think tags
    think_pattern = re.compile(r"<think>.*</think>", re.DOTALL)
    has_think = bool(think_pattern.search(response_cleaned))
    
    # Check for answer tags with proper JSON format
    answer_pattern = re.compile(
        r'<answer_start>\s*\{\s*"score"\s*:\s*\d+\.?\d*\s*\}\s*<answer_end>',
        re.DOTALL
    )
    has_answer = bool(answer_pattern.search(response_cleaned))
    
    # Full format check: think should come before answer
    full_pattern = re.compile(
        r"<think>.*</think>.*<answer_start>.*<answer_end>",
        re.DOTALL
    )
    correct_order = bool(full_pattern.search(response_cleaned))
    
    # 仅格式完全正确时奖励为 1，否则为 0
    if has_think and has_answer and correct_order:
        return 1.0
    return 0.0


def accuracy_reward(
    response: str,
    ground_truth: str,
    decay_coef: float = 2.0,
) -> float:
    """
    基于 MAE 的指数衰减型准确度奖励。
    
    公式: reward = exp(-decay_coef * MAE)
    - MAE=0 时 reward=1.0
    - MAE 越大 reward 越小，衰减系数 decay_coef 控制曲线陡峭程度：
      decay_coef 越大曲线越陡（小误差惩罚更轻、大误差惩罚更重）
      decay_coef 越小曲线越平缓
    
    Args:
        response: 模型输出字符串
        ground_truth: 真实分数字符串
        decay_coef: 衰减系数，控制 reward 曲线陡峭程度，默认 2.0
        
    Returns:
        准确度奖励，范围 [0.0, 1.0]
    """
    predicted_score = extract_score(response)
    
    if predicted_score is None:
        return 0.0
    
    try:
        gt_score = float(ground_truth)
    except (ValueError, TypeError):
        return 0.0
    
    mae = abs(predicted_score - gt_score)
    reward = math.exp(-decay_coef * mae)
    return max(0.0, min(1.0, reward))


def repetition_penalty_reward(
    response: str,
) -> float:
    """
    重复调用工具惩罚：若同一工具在一条推理链中被多次调用则给予惩罚。
    
    逻辑修改：
    - 只要有任何一个工具被重复调用，即返回 -1；
    - 否则，返回 0。
    
    Args:
        response: 模型输出字符串
        
    Returns:
        惩罚值，范围 [-1.0, 0.0]。0 表示无重复，-1 表示有重复。
    """
    tool_calls = extract_tool_calls(response)  # 提取工具调用（标签形式）
    if not tool_calls:
        return 0.0  # 如果没有工具调用，直接返回 0

    call_counts = Counter(tool_calls)  # 统计每个工具标签的调用次数

    # 如果任何工具标签调用次数大于 1，直接返回 -1（惩罚）
    for count in call_counts.values():
        if count > 1:
            return -1.0  # 只要有一个工具重复调用，直接给予 -1 的惩罚

    return 0.0  # 如果没有工具重复调用，返回 0


def tool_usage_reward(
    response: str,
    max_tool_calls: int = 3,
    decay_coef: float = 0.1, 
) -> float:
    """
    工具调用奖励：正确调用工具给予固定奖励，调用次数过多则给予非线性惩罚。
    
    规则:
    - 无工具调用                          → 0.0
    - 有至少 1 次有效调用且总次数 ≤ max   → 1.0（奖励只给一次，不因多次调用累加）
    - 总调用次数 > max_tool_calls         → 奖励 = 1.0 - decay_factor * (total - max)^2
    
    最终值被裁剪到 [-1.0, 1.0] 范围。
    
    示例 (max_tool_calls=4, excess_penalty_factor=0.2, decay_coef=0.1):
    - 调用 3 次工具 → reward = 1.0
    - 调用 4 次工具 → reward = 0.9
    - 调用 5 次工具 → reward = 0.6
    - 调用 6 次工具 → reward = 0.1
    
    Args:
        response: 模型输出字符串
        max_tool_calls: 允许的最大工具调用次数（默认 4）
        excess_penalty_factor: 超过最大调用次数时的惩罚系数（默认 0.2）
        decay_coef: 衰减系数，控制惩罚增长的速度（默认 0.1）
        
    Returns:
        奖励值，范围 [-1.0, 1.0]。
    """
    tool_calls = extract_tool_calls(response)
    total_calls = len(tool_calls)

    if total_calls == 0:
        return 0.0  # 没有工具调用，奖励为 0

    # 至少有一次有效调用 → 基础奖励 1.0
    reward = 1.0

    # 调用次数超出上限时根据非线性惩罚调整奖励
    if total_calls > max_tool_calls:
        excess = total_calls - max_tool_calls
        # 使用非线性公式（例如平方衰减）
        reward -= decay_coef * (excess ** 2)

    # 将奖励裁剪到 [-1.0, 1.0] 范围
    return max(-1.0, min(1.0, reward))



def _rank_data(values: list[float]) -> list[float]:
    """对数值列表赋秩（1-based，并列取平均秩）。"""
    n = len(values)
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman_corr(pred_scores: list[float], gt_scores: list[float]) -> float:
    """
    计算 Spearman 秩相关系数（-1 到 1）。
    要求 len(pred_scores) == len(gt_scores) 且至少 2 个点。
    """
    n = len(pred_scores)
    if n < 2:
        return 0.0
    rp = _rank_data(pred_scores)
    rg = _rank_data(gt_scores)
    mp = sum(rp) / n
    mg = sum(rg) / n
    num = sum((rp[i] - mp) * (rg[i] - mg) for i in range(n))
    den_p = sum((rp[i] - mp) ** 2 for i in range(n))
    den_g = sum((rg[i] - mg) ** 2 for i in range(n))
    den = (den_p * den_g) ** 0.5
    if den <= 0:
        return 0.0
    return num / den


def rank_reward_batch(pred_scores: list[float], gt_scores: list[float]) -> float:
    """
    基于 batch 内预测分数与真实分数排序一致性的排名奖励。
    
    使用 Spearman 秩相关衡量排序正确性，并映射到 [0, 1]：
    reward = (spearman + 1) / 2
    
    Args:
        pred_scores: 本 batch 各样本的预测分数
        gt_scores: 本 batch 各样本的真实分数
        
    Returns:
        排名奖励，范围 [0.0, 1.0]。样本数 < 2 时返回 0.0。
    """
    if len(pred_scores) < 2 or len(gt_scores) < 2 or len(pred_scores) != len(gt_scores):
        return 0.0
    rho = _spearman_corr(pred_scores, gt_scores)
    return max(0.0, min(1.0, (rho + 1.0) / 2.0))


def _save_generations(reward_inputs: list[dict[str, Any]], scores: list[dict[str, float]]) -> None:
    """
    将生成结果保存到 JSONL 文件中，方便检查格式。
    
    每行包含：
    - id: 唯一编号
    - timestamp: 时间戳
    - response: 原始完整回复（多轮拼接后）
    - ground_truth: 真实标签
    - predicted_score: 从回复中提取的预测分数
    - rewards: 各项奖励分数
    """
    global _generation_counter
    
    try:
        # 确保目录存在
        save_dir = os.path.dirname(GENERATIONS_SAVE_PATH)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        with open(GENERATIONS_SAVE_PATH, "a", encoding="utf-8") as f:
            for reward_input, score in zip(reward_inputs, scores):
                response = reward_input.get("response", "")
                ground_truth = reward_input.get("ground_truth", "")
                predicted_score = extract_score(response)
                
                record = {
                    "id": _generation_counter,
                    "timestamp": timestamp,
                    "response": response,
                    "ground_truth": ground_truth,
                    "predicted_score": predicted_score,
                    "rewards": score,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                _generation_counter += 1
                
    except Exception as e:
        # 保存失败不影响训练，只打印警告
        print(f"[WARNING] Failed to save generations: {e}")

def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.2,
    accuracy_weight: float = 0.8,
    decay_coef: float = 2.0,
) -> list[dict[str, float]]:
    """
    版本 1：计算 batch 内每条样本的综合奖励，仅考虑格式奖励和分数奖励。
    
    Args:
        reward_inputs: 每项包含 "response" 与 "ground_truth"
        format_weight: 格式奖励权重（默认 0.2）
        accuracy_weight: 分数奖励权重（默认 0.8）
        decay_coef: 准确度指数衰减系数（默认 2.0）
        
    Returns:
        每项包含 "overall", "format", "accuracy"
    """
    scores = []
    
    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        
        fmt_score = format_reward(response)
        acc_score = accuracy_reward(response, ground_truth, decay_coef=decay_coef)

        # 基础奖励
        overall_score = format_weight * fmt_score + accuracy_weight * acc_score
        
        entry: dict[str, float] = {
            "overall": overall_score,
            "format": fmt_score,
            "accuracy": acc_score,
        }

        scores.append(entry)
    
    return scores


def compute_score_v2(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.2,
    accuracy_weight: float = 0.7,
    tool_usage_weight: float = 0.1,
    decay_coef_score: float = 2.0,
    max_tool_calls: int = 3,
    decay_coef_tool: float = 0.1,
) -> list[dict[str, float]]:
    """
    版本 2：计算 batch 内每条样本的综合奖励，考虑格式奖励、分数奖励和工具调用奖励。
    
    Args:
        reward_inputs: 每项包含 "response" 与 "ground_truth"
        format_weight: 格式奖励权重（默认 0.2）
        accuracy_weight: 分数奖励权重（默认 0.7）
        tool_usage_weight: 工具调用奖励权重（默认 0.1）
        decay_coef_score: 准确度指数衰减系数（默认 2.0）
        max_tool_calls: 最大工具调用次数（默认 3）
        decay_coef_tool: 工具调用奖励衰减系数（默认 0.1）
        
    Returns:
        每项包含 "overall", "format", "accuracy", "tool_usage"
    """
    scores = []
    
    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        
        fmt_score = format_reward(response)
        acc_score = accuracy_reward(response, ground_truth, decay_coef=decay_coef_score)
        tool_usage_score = tool_usage_reward(response, max_tool_calls=max_tool_calls, decay_coef=decay_coef_tool)
        
        # 基础奖励 + 工具调用奖励
        overall_score = format_weight * fmt_score + accuracy_weight * acc_score + tool_usage_weight * tool_usage_score
        
        entry: dict[str, float] = {
            "overall": overall_score,
            "format": fmt_score,
            "accuracy": acc_score,
            "tool_usage": tool_usage_score,
        }

        scores.append(entry)
    
    return scores


def compute_score_v3(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.2,
    accuracy_weight: float = 0.65,
    tool_usage_weight: float = 0.1,
    repetition_penalty_weight: float = 0.05,
    decay_coef_score: float = 5.0,  
    max_tool_calls: int = 3,
    decay_coef_tool: float = 0.1,
) -> list[dict[str, float]]:
    """
    版本 3：计算 batch 内每条样本的综合奖励，考虑格式奖励、分数奖励、工具调用奖励和重复调用惩罚。
    
    Args:
        reward_inputs: 每项包含 "response" 与 "ground_truth"
        format_weight: 格式奖励权重（默认 0.2）
        accuracy_weight: 分数奖励权重（默认 0.65）
        tool_usage_weight: 工具调用奖励权重（默认 0.1）
        repetition_penalty_weight: 重复调用惩罚权重（默认 0.05）
        decay_coef_score: 准确度指数衰减系数（默认 2.0）
        max_tool_calls: 最大工具调用次数（默认 3）
        decay_coef_tool: 工具调用奖励衰减系数（默认 0.1）
        
    Returns:
        每项包含 "overall", "format", "accuracy", "tool_usage", "repetition_penalty"
    """
    scores = []
    
    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        
        fmt_score = format_reward(response)
        acc_score = accuracy_reward(response, ground_truth, decay_coef=decay_coef_score)
        tool_usage_score = tool_usage_reward(response, max_tool_calls=max_tool_calls, decay_coef=decay_coef_tool)
        repetition_penalty_score = repetition_penalty_reward(response)
        
        # 综合奖励 = 格式奖励 + 分数奖励 + 工具调用奖励 + 重复调用惩罚
        overall_score = (
            format_weight * fmt_score + 
            accuracy_weight * acc_score + 
            tool_usage_weight * tool_usage_score + 
            repetition_penalty_weight * repetition_penalty_score
        )
        
        entry: dict[str, float] = {
            "overall": overall_score,
            "format": fmt_score,
            "accuracy": acc_score,
            "tool_usage": tool_usage_score,
            "repetition_penalty": repetition_penalty_score,
        }

        scores.append(entry)
    
    return scores


# For testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "response": '<think>The image shows good quality with clear details.</think>\n<answer_start>{"score": 3.46}<answer_end>',
            "ground_truth": "3.46"
        },
        {
            "response": '<think>Poor quality image.</think>\n<answer_start>{"score": 2.50}<answer_end>',
            "ground_truth": "3.46"
        },
        {
            "response": 'The score is 3.46',  # Wrong format
            "ground_truth": "3.46"
        },
        # 排名测试：预测顺序与真实顺序一致
        {
            "response": '<think>Good.</think>\n<answer_start>{"score": 4.0}<answer_end>',
            "ground_truth": "4.0"
        },
        {
            "response": '<think>Mid.</think>\n<answer_start>{"score": 3.0}<answer_end>',
            "ground_truth": "3.0"
        },
        {
            "response": '<think>Low.</think>\n<answer_start>{"score": 2.0}<answer_end>',
            "ground_truth": "2.0"
        },
    ]

    # 仅格式 + 准确度，指数衰减
    results = compute_score(test_cases, decay_coef=2.0)
    print("Without rank reward:")
    for i, (case, result) in enumerate(zip(test_cases, results)):
        print(f"  Case {i + 1}: overall={result['overall']:.3f} format={result['format']} accuracy={result['accuracy']:.3f}")

    # 启用排名奖励（权重需与 format_weight + accuracy_weight 协调，例如 0.1）
    results_with_rank = compute_score(
        test_cases,
        format_weight=0.2,
        accuracy_weight=0.7,
        rank_weight=0.1,
        decay_coef=2.0,
    )
    print("\nWith rank reward (rank_weight=0.1):")
    for i, (case, result) in enumerate(zip(test_cases, results_with_rank)):
        r = result.get("rank", None)
        print(f"  Case {i + 1}: overall={result['overall']:.3f} format={result['format']} accuracy={result['accuracy']:.3f} rank={r}")

    # ===================================================================
    # 工具调用相关奖励测试
    # ===================================================================
    print("\n" + "=" * 60)
    print("Tool reward tests")
    print("=" * 60)

    tool_test_cases = [
        # Case 1: 无工具调用
        {
            "response": '<think>The image looks good.</think>\n<answer_start>{"score": 3.50}<answer_end>',
            "ground_truth": "3.50",
        },
        # Case 2: 3 个不同工具调用（无重复）
        {
            "response": (
                '<think>Checking quality via '
                '<GradientMagnitudeMap><|vision_start|><|image_pad|><|vision_end|> '
                '<NoiseResidualMap><|vision_start|><|image_pad|><|vision_end|> '
                '<PerceptualColorfulnessMap><|vision_start|><|image_pad|><|vision_end|> '
                'Good quality.</think>\n<answer_start>{"score": 3.80}<answer_end>'
            ),
            "ground_truth": "3.80",
        },
        # Case 3: 有重复调用（GradientMagnitudeMap 被调用了 2 次）
        {
            "response": (
                '<think>Checking sharpness '
                '<GradientMagnitudeMap><|vision_start|><|image_pad|><|vision_end|> '
                'Let me check again '
                '<GradientMagnitudeMap><|vision_start|><|image_pad|><|vision_end|> '
                '<NoiseResidualMap><|vision_start|><|image_pad|><|vision_end|> '
                'Moderate quality.</think>\n<answer_start>{"score": 2.80}<answer_end>'
            ),
            "ground_truth": "3.00",
        },
        # Case 4: 过多调用（6 次，超过 max_tool_calls=5）
        {
            "response": (
                '<think>'
                '<GradientMagnitudeMap><|vision_start|><|image_pad|><|vision_end|> '
                '<NoiseResidualMap><|vision_start|><|image_pad|><|vision_end|> '
                '<PerceptualColorfulnessMap><|vision_start|><|image_pad|><|vision_end|> '
                '<ExtremeLuminanceMap><|vision_start|><|image_pad|><|vision_end|> '
                '<LuminanceHistogram><|vision_start|><|image_pad|><|vision_end|> '
                '<DoGSharpnessMap><|vision_start|><|image_pad|><|vision_end|> '
                'Low quality.</think>\n<answer_start>{"score": 2.00}<answer_end>'
            ),
            "ground_truth": "2.00",
        },
        # Case 5: 过多调用 + 重复（7 次，其中 GradientMagnitudeMap 出现 3 次）
        {
            "response": (
                '<think>'
                '<GradientMagnitudeMap><|vision_start|><|image_pad|><|vision_end|> '
                '<NoiseResidualMap><|vision_start|><|image_pad|><|vision_end|> '
                '<GradientMagnitudeMap><|vision_start|><|image_pad|><|vision_end|> '
                '<PerceptualColorfulnessMap><|vision_start|><|image_pad|><|vision_end|> '
                '<ExtremeLuminanceMap><|vision_start|><|image_pad|><|vision_end|> '
                '<GradientMagnitudeMap><|vision_start|><|image_pad|><|vision_end|> '
                '<LuminanceHistogram><|vision_start|><|image_pad|><|vision_end|> '
                'Very low.</think>\n<answer_start>{"score": 1.50}<answer_end>'
            ),
            "ground_truth": "1.50",
        },
    ]

    # 单独测试 extract_tool_calls
    print("\n--- extract_tool_calls ---")
    for i, case in enumerate(tool_test_cases):
        calls = extract_tool_calls(case["response"])
        print(f"  Case {i + 1}: {len(calls)} calls → {calls}")

    # 单独测试 repetition_penalty_reward
    print("\n--- repetition_penalty_reward ---")
    for i, case in enumerate(tool_test_cases):
        score = repetition_penalty_reward(case["response"])
        print(f"  Case {i + 1}: {score:.2f}")

    # 单独测试 tool_usage_reward
    print("\n--- tool_usage_reward (max_tool_calls=5) ---")
    for i, case in enumerate(tool_test_cases):
        score = tool_usage_reward(case["response"], max_tool_calls=5)
        print(f"  Case {i + 1}: {score:.2f}")

    # 综合测试：启用所有工具奖励
    results_with_tools = compute_score(
        tool_test_cases,
        format_weight=0.2,
        accuracy_weight=0.6,
        repetition_penalty_weight=0.1,
        tool_usage_weight=0.1,
        decay_coef=2.0,
        penalty_per_repeat=0.5,
        max_tool_calls=5,
        excess_penalty_per_call=0.2,
    )
    print("\n--- compute_score with tool rewards ---")
    for i, (case, result) in enumerate(zip(tool_test_cases, results_with_tools)):
        rep = result.get("repetition_penalty", "N/A")
        tu = result.get("tool_usage", "N/A")
        rep_str = f"{rep:.2f}" if isinstance(rep, float) else rep
        tu_str = f"{tu:.2f}" if isinstance(tu, float) else tu
        print(
            f"  Case {i + 1}: overall={result['overall']:.3f} "
            f"format={result['format']:.1f} accuracy={result['accuracy']:.3f} "
            f"rep_penalty={rep_str} tool_usage={tu_str}"
        )
