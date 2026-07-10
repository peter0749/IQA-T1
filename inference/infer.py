"""
Usage:
    python infer.py --image_path /path/to/image.jpg
    python infer.py --image_path /path/to/image.jpg --model_path /path/to/model --device cuda
"""

import os
import re
import json
import argparse
import shutil
import torch
import numpy as np
from transformers import (
    AutoProcessor,
    GenerationConfig,
    set_seed,
    Qwen3VLForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

# 导入工具函数（从 scripts 子目录）
import sys
_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPT_DIR))

from tools import (
    ALL_TOOL_NAMES,
    TOOL_RESIZE_CONFIG,
    resize_tool_image,
    GradientMagnitudeMap,
    GradientOrientationCoherenceMap,
    GradientMagnitudeHistogram,
    HighFrequencyResidualMap,
    DoGSharpnessMap,
    FourierMagnitudeSpectrum,
    NoiseResidualMap,
    LuminanceHistogram,
    ExtremeLuminanceMap,
    PerceptualColorfulnessMap,
    ChromaticDeviationMap,
    ColorSaturationClippingMap,
    MSCNNormalizedLuminanceMap,
    MSCNDistributionHistogram,
    LocalGradientDeviationMap,
)


# ================== 工具调用相关 ==================

# 工具名到函数的映射
TOOL_FUNCTIONS = {
    "GradientMagnitudeMap": GradientMagnitudeMap,
    "GradientOrientationCoherenceMap": GradientOrientationCoherenceMap,
    "GradientMagnitudeHistogram": GradientMagnitudeHistogram,
    "HighFrequencyResidualMap": HighFrequencyResidualMap,
    "DoGSharpnessMap": DoGSharpnessMap,
    "FourierMagnitudeSpectrum": FourierMagnitudeSpectrum,
    "NoiseResidualMap": NoiseResidualMap,
    "LuminanceHistogram": LuminanceHistogram,
    "ExtremeLuminanceMap": ExtremeLuminanceMap,
    "PerceptualColorfulnessMap": PerceptualColorfulnessMap,
    "ChromaticDeviationMap": ChromaticDeviationMap,
    "ColorSaturationClippingMap": ColorSaturationClippingMap,
    "MSCNNormalizedLuminanceMap": MSCNNormalizedLuminanceMap,
    "MSCNDistributionHistogram": MSCNDistributionHistogram,
    "LocalGradientDeviationMap": LocalGradientDeviationMap,
}

# 工具调用检测正则表达式
TOOL_PATTERN = re.compile(r'<(' + '|'.join(ALL_TOOL_NAMES) + r')>')


def extract_tool_call(text):
    """从文本中提取第一个工具调用标签"""
    match = TOOL_PATTERN.search(text)
    return match.group(1) if match else None


class ToolCallStoppingCriteria(StoppingCriteria):
    """
    实时检测工具调用标签的 StoppingCriteria

    在生成过程中检测是否出现 <ToolName> 标签，一旦检测到立即停止生成。
    """

    def __init__(self, tokenizer, tool_names, prompt_length, check_interval=3):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.tool_patterns = [f"<{name}>" for name in tool_names]
        self.check_interval = check_interval
        self.detected_tool = None
        self.call_count = 0
        self.last_checked_len = 0

    def reset(self):
        self.detected_tool = None
        self.call_count = 0
        self.last_checked_len = 0

    def __call__(self, input_ids, scores, **kwargs):
        if self.detected_tool is not None:
            return True
        self.call_count += 1
        if self.call_count > 5 and self.call_count % self.check_interval != 0:
            return False
        new_tokens = input_ids[0, self.prompt_length:]
        if new_tokens.shape[0] == 0:
            return False
        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        for pattern in self.tool_patterns:
            if pattern in new_text:
                self.detected_tool = pattern[1:-1]
                return True
        return False


def get_sample_tool_dir(base_dir, image_path):
    """获取样本专属的工具图像保存目录"""
    from pathlib import Path
    image_name_no_ext = Path(image_path).stem
    sample_dir = os.path.join(base_dir, image_name_no_ext)
    os.makedirs(sample_dir, exist_ok=True)
    return sample_dir


def generate_tool_image(tool_name, image_path, save_dir, cache=None):
    """根据工具名生成对应的工具图像（包含 resize 步骤）"""
    from pathlib import Path

    if cache is not None and tool_name in cache:
        cached_path = cache[tool_name]
        if os.path.exists(cached_path):
            return cached_path

    image_name_no_ext = Path(image_path).stem
    expected_path = os.path.join(save_dir, image_name_no_ext, f"{tool_name}.png")

    if os.path.exists(expected_path):
        if cache is not None:
            cache[tool_name] = expected_path
        return expected_path

    func = TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        raise ValueError(f"Unknown tool: {tool_name}")

    result = func(image_path, save_dir)
    if result is None:
        raise RuntimeError(f"Tool {tool_name} failed to generate image")

    raw_save_path = result["save_path"]

    if tool_name in TOOL_RESIZE_CONFIG:
        raw_path = Path(raw_save_path)
        resize_tool_image(raw_path, raw_path, tool_name)

    if cache is not None:
        cache[tool_name] = raw_save_path

    return raw_save_path


def generate_with_tools(
    model,
    processor,
    initial_messages,
    image_path,
    gen_config,
    device,
    tool_save_dir,
    max_tool_calls=5,
    stopping_check_interval=3,
):
    """支持工具调用的多轮生成函数（内联图像版本）"""
    from PIL import Image as PILImage

    tool_images = []
    tools_used = []
    full_response = ""
    tool_cache = {}
    all_images = []

    base_text = processor.apply_chat_template(
        initial_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    for msg in initial_messages:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'image' or 'image' in item:
                    img_data = item.get('image')
                    if img_data is None:
                        continue
                    if isinstance(img_data, PILImage.Image):
                        all_images.append(img_data)
                    elif isinstance(img_data, str):
                        if img_data.startswith('file://'):
                            all_images.append(PILImage.open(img_data[7:]).convert('RGB'))
                        else:
                            all_images.append(PILImage.open(img_data).convert('RGB'))

    VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    for round_idx in range(max_tool_calls + 1):
        if round_idx == 0:
            current_text = base_text
        else:
            current_text = base_text + full_response

        inputs = processor(
            text=[current_text],
            images=all_images if all_images else None,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(device)

        prompt_length = inputs.input_ids.shape[1]
        tool_stopping_criteria = ToolCallStoppingCriteria(
            tokenizer=processor.tokenizer,
            tool_names=ALL_TOOL_NAMES,
            prompt_length=prompt_length,
            check_interval=stopping_check_interval,
        )
        stopping_criteria_list = StoppingCriteriaList([tool_stopping_criteria])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                stopping_criteria=stopping_criteria_list,
                use_cache=True
            )

        output_trimmed = outputs[0][prompt_length:]
        response = processor.decode(output_trimmed, skip_special_tokens=False)
        tool_name = tool_stopping_criteria.detected_tool

        if tool_name is None:
            full_response += response
            break

        tool_tag = f"<{tool_name}>"
        tag_idx = response.find(tool_tag)
        if tag_idx != -1:
            response_until_tool = response[:tag_idx + len(tool_tag)]
        else:
            response_until_tool = response
        full_response += response_until_tool

        try:
            tool_image_path = generate_tool_image(
                tool_name, image_path, tool_save_dir, cache=tool_cache
            )
            tool_images.append(tool_image_path)
            tools_used.append(tool_name)
            tool_img = PILImage.open(tool_image_path).convert("RGB")
            all_images.append(tool_img)
            full_response += VISION_PLACEHOLDER
        except Exception as e:
            print(f"[Warning] Tool {tool_name} generation failed: {e}")
            break

    return full_response, tools_used, tool_images


# ================== IQA 推理核心类 ==================

SYSTEM_PROMPT = (
    "You are an expert image quality assessment (IQA) assistant. "
    "You will analyze images using various visual quality analysis tools and provide quality scores.\n\n"
    "## Output Format Requirements\n"
    "Your response MUST follow this exact format:\n\n"
    "<think>\n"
    "[Your detailed analysis of the image quality, referencing the visual analysis maps/histograms provided. "
    "Describe observations about:\n"
    "- Luminance and exposure characteristics\n"
    "- Sharpness and blur levels\n"
    "- Color quality and saturation\n"
    "- Noise and artifacts\n"
    "- Overall perceptual quality]\n"
    "</think>\n"
    "<answer_start>{\"score\": X.XX}<answer_end>\n\n"
    "## Scoring Guidelines\n"
    "- Score range: 1.00 to 5.00\n"
    "- Higher scores indicate better perceptual quality\n"
    "- Consider all visual quality dimensions in your assessment\n\n"
    "## Available Analysis Tools\n"
    "You may reference these visual analysis tools in your reasoning:\n"
    "- GradientMagnitudeMap, GradientOrientationCoherenceMap, GradientMagnitudeHistogram\n"
    "- HighFrequencyResidualMap, DoGSharpnessMap, LocalGradientDeviationMap\n"
    "- NoiseResidualMap, FourierMagnitudeSpectrum\n"
    "- LuminanceHistogram, ExtremeLuminanceMap\n"
    "- PerceptualColorfulnessMap, ChromaticDeviationMap, ColorSaturationClippingMap\n"
    "- MSCNNormalizedLuminanceMap, MSCNDistributionHistogram"
)

USER_PROMPT = "What is your overall rating on the quality of this picture?"


def extract_score(response_text):
    """从模型回复中提取分数（鲁棒）"""
    # 优先匹配 JSON 格式的 score
    m = re.search(r'"score"\s*:\s*([0-9]+\.?[0-9]*)', response_text)
    if m:
        return float(m.group(1))
    # 回退：匹配 1~5 之间的数字
    nums = re.findall(r'([0-9]+\.?[0-9]*)', response_text)
    for n in nums:
        v = float(n)
        if 1 <= v <= 5:
            return v
    return None


class IQAInference:
    """
    IQA 单图推理器

    封装模型加载与推理流程，对外提供简洁的 infer() 接口。

    Examples
    --------
    >>> inferencer = IQAInference(model_path="/path/to/model")
    >>> result = inferencer.infer("/path/to/image.jpg")
    >>> print(result["score"])
    >>> print(result["raw_response"])
    """

    def __init__(
        self,
        model_path,
        device="cuda",
        seed=42,
        enable_tools=True,
        max_tool_calls=6,
        tool_save_dir="result",
        keep_tool_images=True,
    ):
        """
        Parameters
        ----------
        model_path : str
            HuggingFace 格式的模型路径。
        device : str
            推理设备，默认 "cuda"。
        seed : int
            随机种子。
        enable_tools : bool
            是否启用工具调用。
        max_tool_calls : int
            最大工具调用轮数。
        tool_save_dir : str or None
            工具图像保存目录，默认为系统临时目录。
        keep_tool_images : bool
            是否保留工具图像。
        """
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.enable_tools = enable_tools
        self.max_tool_calls = max_tool_calls
        self.tool_save_dir = tool_save_dir
        self.keep_tool_images = keep_tool_images

        set_seed(seed)

        # ---- 加载模型 ----
        self._load_model()

        # ---- 推理配置 ----
        self.gen_config = GenerationConfig(
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            max_new_tokens=2560,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # ---- 工具图像目录 ----
        if self.enable_tools:
            if self.tool_save_dir is None:
                import tempfile
                self.tool_save_dir = tempfile.mkdtemp(prefix="iqa_tools_")
            os.makedirs(self.tool_save_dir, exist_ok=True)

    def _resolve_load_path(self):
        """解析实际的可加载路径（处理 FSDP / actor 子目录等情况）"""
        load_path = self.model_path

        # 若根目录无 config.json，尝试 actor/huggingface 子目录
        if not os.path.exists(os.path.join(load_path, "config.json")):
            alt_path = os.path.join(load_path, "actor", "huggingface")
            if os.path.exists(os.path.join(alt_path, "config.json")):
                print(f"[INFO] Using model from subdir: {alt_path}")
                return alt_path

        # 检查权重文件是否存在
        _hf_weight_names = ("pytorch_model.bin", "model.safetensors")
        _has_weights = any(
            os.path.isfile(os.path.join(load_path, n)) for n in _hf_weight_names
        ) or any(
            f.startswith("model-") and f.endswith(".safetensors")
            for f in (os.listdir(load_path) if os.path.isdir(load_path) else [])
        )

        if not _has_weights:
            _actor_dir = os.path.join(self.model_path, "actor")
            _has_fsdp = os.path.isdir(_actor_dir) and any(
                re.match(r"model_world_size_\d+_rank_\d+\.pt", f)
                for f in (os.listdir(_actor_dir) if os.path.isdir(_actor_dir) else [])
            )
            if _has_fsdp:
                _merge_cmd = (
                    f"python train/grpo/EasyR1_IQA-T1/scripts/model_merger.py "
                    f"--local_dir {os.path.abspath(_actor_dir)}"
                )
                raise RuntimeError(
                    f"Checkpoint 下未找到 HuggingFace 权重文件（在 {load_path}）。"
                    f" 当前为 FSDP 分片格式，请先合并再推理。在项目根目录执行:\n  {_merge_cmd}\n"
                    "合并完成后会在此目录生成 model.safetensors，再重新运行。"
                )
            raise RuntimeError(
                f"在 {load_path} 下未找到模型权重文件（pytorch_model.bin / model.safetensors）。"
                "请确认 model_path 指向已导出权重的 HuggingFace 格式 checkpoint。"
            )

        return load_path

    def _load_model(self):
        """加载模型与 processor"""
        print(f"[1/3] Loading model from {self.model_path} ...")

        load_path = self._resolve_load_path()

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            load_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": 0} if self.device == "cuda" else None,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            load_path,
            fix_mistral_regex=True,
        )

        print("Model loaded.")

    def infer(self, image_path):
        """
        对单张图像进行 IQA 推理。

        Parameters
        ----------
        image_path : str
            待评估图像的路径。

        Returns
        -------
        dict
            {
                "image_path": str,        # 输入图像路径
                "score": float or None,   # 预测分数 (1.00~5.00)
                "raw_response": str,      # 模型原始输出
                "tools_used": list[str],  # 使用的工具列表
                "num_tool_calls": int,    # 工具调用次数
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"[2/3] Running inference on: {image_path}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

        if self.enable_tools:
            resp, tools_used, tool_images = generate_with_tools(
                model=self.model,
                processor=self.processor,
                initial_messages=messages,
                image_path=image_path,
                gen_config=self.gen_config,
                device=self.device,
                tool_save_dir=self.tool_save_dir,
                max_tool_calls=self.max_tool_calls,
            )

            # 清理工具图像
            if not self.keep_tool_images and tool_images:
                try:
                    from pathlib import Path
                    sample_tool_dir = os.path.join(
                        self.tool_save_dir, Path(image_path).stem
                    )
                    if os.path.exists(sample_tool_dir):
                        shutil.rmtree(sample_tool_dir)
                except Exception:
                    pass
        else:
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.gen_config,
                    use_cache=True,
                )

            output_trimmed = outputs[0][inputs.input_ids.shape[1]:]
            resp = self.processor.decode(output_trimmed, skip_special_tokens=False)
            tools_used = []

        score = extract_score(resp)

        print(f"[3/3] Inference complete. Score = {score}")

        return {
            "image_path": image_path,
            "score": score,
            "raw_response": resp,
            "tools_used": tools_used,
            "num_tool_calls": len(tools_used),
        }

    def cleanup(self):
        """清理临时工具图像目录"""
        if self.enable_tools and not self.keep_tool_images:
            try:
                if self.tool_save_dir and os.path.exists(self.tool_save_dir):
                    shutil.rmtree(self.tool_save_dir)
            except Exception:
                pass


# ================== CLI 入口 ==================

def parse_args():
    parser = argparse.ArgumentParser(
        description="IQA Single-Image Inference — 对单张图片进行质量评估"
    )
    parser.add_argument(
        "--image_path", "-i",
        type=str,
        required=True,
        help="待评估图像的路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径（HuggingFace 格式，若不指定请在脚本中设置）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备，默认 cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--enable_tools",
        action="store_true",
        default=True,
        help="启用工具调用功能",
    )
    parser.add_argument(
        "--no_tools",
        action="store_true",
        help="禁用工具调用",
    )
    parser.add_argument(
        "--max_tool_calls",
        type=int,
        default=6,
        help="最大工具调用轮数",
    )
    parser.add_argument(
        "--tool_save_dir",
        type=str,
        default="result",
        help="工具图像保存目录，默认为 result/",
    )
    parser.add_argument(
        "--keep_tool_images",
        action="store_true",
        default=True,
        help="保留工具图像（默认保留）",
    )
    parser.add_argument(
        "--output_json",
        "-o",
        type=str,
        default=None,
        help="将结果保存为 JSON 文件，默认保存到 result/{image_name}/result.json",
    )
    return parser.parse_args()


def main():
    from pathlib import Path

    args = parse_args()

    enable_tools = args.enable_tools and not args.no_tools

    # 自动确定输出目录: tool_save_dir/{image_name}/ 存工具图
    image_name = Path(args.image_path).stem
    tool_result_dir = os.path.join(args.tool_save_dir, image_name)
    os.makedirs(tool_result_dir, exist_ok=True)

    # 所有结果累计在 result/result.json
    os.makedirs("result", exist_ok=True)
    output_json = args.output_json or os.path.join("result", "result.json")

    print("=" * 60)
    print("IQA Single-Image Inference")
    print(f"Model       : {args.model_path}")
    print(f"Image       : {args.image_path}")
    print(f"Device      : {args.device}")
    print(f"Tool dir    : {tool_result_dir}")
    print(f"Tools       : {'enabled' if enable_tools else 'disabled'}")
    if enable_tools:
        print(f"Max calls   : {args.max_tool_calls}")
    print("=" * 60)

    inferencer = IQAInference(
        model_path=args.model_path,
        device=args.device,
        seed=args.seed,
        enable_tools=enable_tools,
        max_tool_calls=args.max_tool_calls,
        tool_save_dir=args.tool_save_dir,
        keep_tool_images=args.keep_tool_images,
    )

    try:
        result = inferencer.infer(args.image_path)

        print("\n" + "=" * 60)
        print(f"Predicted Score : {result['score']}")
        print(f"Tools Used      : {result['tools_used'] if result['tools_used'] else 'None'}")
        print(f"Num Tool Calls  : {result['num_tool_calls']}")
        print("=" * 60)
        print("\n--- Raw Response ---")
        print(result["raw_response"])

        # 累计保存到 result/result.json
        os.makedirs("result", exist_ok=True)
        all_results = []
        if os.path.exists(output_json):
            try:
                with open(output_json, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
                    if not isinstance(all_results, list):
                        all_results = [all_results]
            except (json.JSONDecodeError, IOError):
                all_results = []
        all_results.append(result)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResult appended to: {output_json} (total: {len(all_results)})")

    finally:
        inferencer.cleanup()


if __name__ == "__main__":
    main()
