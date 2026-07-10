from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from vllm import RequestOutput, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image as _ds_process_image
from ...utils.dataset import process_video as _ds_process_video
from .tool_envs import ToolBase

# ---------------------------------------------------------------------------
# 监控日志：保存 agent 多轮生成的详细信息
# ---------------------------------------------------------------------------
_AGENT_LOG_PATH = os.environ.get(
    "AGENT_ROLLOUT_LOG_PATH",
    "./agent_rollout_monitor.jsonl",
)
_AGENT_LOG_ENABLED = os.environ.get("AGENT_ROLLOUT_LOG", "0") == "1"
# 每个 step 最多记录多少个样本（避免日志文件太大）
_AGENT_LOG_MAX_SAMPLES = int(os.environ.get("AGENT_ROLLOUT_LOG_MAX_SAMPLES", "4"))


def _log_agent_event(event: Dict[str, Any]) -> None:
    """将 agent rollout 事件追加写入 JSONL 日志文件（静默失败不影响训练）。"""
    if not _AGENT_LOG_ENABLED:
        return
    try:
        log_dir = os.path.dirname(_AGENT_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(_AGENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _concat_vllm_input(
    prompt_token_ids: torch.Tensor | np.ndarray,
    response_token_ids: torch.Tensor,
    tokenizer=None,
) -> List[int]:
    """
    Concatenate prompt + newly generated response token ids, taking care of
    potential OOV ids for some Qwen base tokenizers.
    """
    if tokenizer is not None:
        max_token_id = max(tokenizer.get_vocab().values())
        tokenizer_size = len(tokenizer)
        max_token_id = max(max_token_id, tokenizer_size)
        valid_token_mask = torch.le(response_token_ids, max_token_id)
        response_token_ids = torch.masked_select(response_token_ids, valid_token_mask)

    if isinstance(prompt_token_ids, torch.Tensor):
        out = torch.cat([prompt_token_ids, response_token_ids.to(prompt_token_ids.device)], dim=-1)
        return out.cpu().numpy().flatten().tolist()
    else:
        out = np.concatenate([prompt_token_ids, response_token_ids.cpu().numpy()], axis=-1)
        return out.flatten().tolist()


def _merge_multi_modal_inputs(mm_input: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two multi‑modal input dicts by concatenating along the batch dimension.
    """
    if not mm_input and not other:
        return {}
    if not mm_input:
        return dict(other)
    if not other:
        return dict(mm_input)

    out: Dict[str, Any] = {}
    for key, mm_value in mm_input.items():
        if key not in other:
            out[key] = mm_value
            continue
        other_value = other.pop(key)
        if isinstance(mm_value, np.ndarray) and isinstance(other_value, np.ndarray):
            merged = np.concatenate([mm_value, other_value], axis=0)
        elif isinstance(mm_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            merged = torch.cat([mm_value, other_value], dim=0)
        else:
            raise ValueError(f"Invalid multi‑modal input types: {type(mm_value)=}, {type(other_value)=}")
        out[key] = merged
    out.update(other)
    return out


def _preprocess_multi_modal_inputs(prompt_str: str, processor, **kwargs: Any) -> Tuple[str, torch.Tensor, Dict[str, Any]]:
    """
    将工具返回的多模态观测（仅含占位符 + 图像）编码成：
    - vllm 侧使用的纯 token 序列（prompt_str_vllm）
    - 模型侧使用的 token 序列（obs_token_ids_model）
    - multi_modal_inputs（供 position_ids 计算）
    """
    if processor is None or "multi_modal_data" not in kwargs:
        return prompt_str, torch.zeros(0, dtype=torch.long), {}

    # 用与主 prompt 一致的占位符替换规则
    vllm_input_prompt = prompt_str.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    input_mm_data = kwargs.get("multi_modal_data", {"image": []})

    # 复用 EasyR1 自带的图像 / 视频预处理逻辑
    images = [img for img in input_mm_data.get("image", [])]
    videos = [vid for vid in input_mm_data.get("video", [])]

    processed_images = None
    processed_videos = None
    if images:
        processed_images = [_ds_process_image(img, None, None) for img in images]
    if videos:
        processed_videos = [
            _ds_process_video(vid, None, None, video_fps=2.0)[0] for vid in videos  # use default fps
        ]

    if processed_images is not None:
        model_inputs = processor(text=[vllm_input_prompt], images=processed_images, return_tensors="pt")
    else:
        model_inputs = processor(text=[vllm_input_prompt], videos=processed_videos, return_tensors="pt")

    input_ids = model_inputs.pop("input_ids")[0]
    attention_mask = model_inputs.pop("attention_mask")[0]

    # RoPE 索引在外部由 EasyR1 的 FSDPWorker 统一处理
    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    return vllm_input_prompt, input_ids, dict(model_inputs)


def execute_tool_call(
    sample: Dict[str, Any],
    tokenizer,
    processor,
    pbar: tqdm | None = None,
) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
    """
    调用一个具体工具，并将其输出转换为 token 序列 + 多模态结构。
    """
    action_string = sample.get("action", "")
    tool = sample.get("tool", None)

    # 非 agent 数据：没有工具名或 action 为空，直接结束这一轮
    if action_string == "" or tool is None:
        return {}, 0.0, True, {}

    tool_result, reward, done, info = tool.execute(action_string)

    # 无观测（例如已经生成最终答案）
    if not tool_result:
        obs_info: Dict[str, Any] = {}
    elif isinstance(tool_result, str):
        # 纯文本观测
        obs_token_ids = tokenizer.encode(tool_result, add_special_tokens=False)
        obs_info = {
            "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
            "prompt_token_ids_model": torch.tensor(obs_token_ids),
        }
    elif isinstance(tool_result, dict):
        # {"prompt": "...", "multi_modal_data": {...}}
        prompt_str = tool_result.pop("prompt", "")
        if not prompt_str:
            raise ValueError("tool_result['prompt'] must be non‑empty when returning dict.")

        prompt_str_vllm, obs_token_ids_model, mm_inputs = _preprocess_multi_modal_inputs(
            prompt_str, processor, **tool_result
        )
        obs_token_ids_vllm = tokenizer.encode(prompt_str_vllm, add_special_tokens=False, return_tensors="pt")[0]
        obs_info = {
            "prompt_token_ids_vllm": obs_token_ids_vllm,
            "prompt_token_ids_model": obs_token_ids_model,
            **tool_result,
        }
        if mm_inputs:
            obs_info["multi_modal_inputs"] = mm_inputs
    else:
        raise ValueError(f"Invalid tool_result type: {type(tool_result)=} -- {tool_result}")

    if pbar is not None:
        pbar.update(1)
    return obs_info, reward, done, info


class ParallelEnv:
    """
    多环境并行版工具调用环境，接口对齐 OpenAI Gym:

    - reset: 初始化每个样本对应的工具（例如 IQA Toolbox）
    - step: 读取 vLLM 输出文本，解析并执行工具，返回新的观测 + reward + done
    """

    def __init__(self, env_config, tokenizer, processor, **_: Any) -> None:
        self.config = env_config
        self.tokenizer = tokenizer
        self.processor = processor

        # 每个样本对应一个 ToolBase 实例或 None
        self.tools: List[ToolBase | None] = []

    def step(
        self,
        active_indices: Sequence[int],
        actions: Sequence[RequestOutput],
    ) -> Tuple[List[Dict[str, Any]], List[float], List[bool], Dict[str, Any]]:
        obs_list: List[Dict[str, Any]] = [{} for _ in range(len(actions))]
        reward_list: List[float] = [0.0 for _ in range(len(actions))]
        done_list: List[bool] = []
        valid_indices: List[int] = []
        real_indices: List[int] = []
        valid_actions: List[str] = []

        # 1) 过滤无效 action（长度为 0 或因长度截断）
        for i, (idx, act) in enumerate(zip(active_indices, actions)):
            if not act.outputs or act.outputs[0].finish_reason == "length":
                done_list.append(True)
                continue
            if len(act.outputs[0].token_ids) == 0:
                done_list.append(True)
                continue

            done_list.append(False)
            real_indices.append(i)
            valid_indices.append(idx)
            valid_actions.append(act.outputs[0].text)

        agent_inputs: List[Dict[str, Any]] = []
        for i, idx, action in zip(real_indices, valid_indices, valid_actions):
            agent_inputs.append(
                {
                    "idx": i,
                    "valid_idx": idx,
                    "action": action,
                    "tool": self.tools[idx],
                }
            )

        # 2) 执行工具调用（串行或并行）
        num_workers = min(self.config.concurrent_workers, len(valid_actions))
        pbar = (
            tqdm(total=len(valid_actions), desc=f"Tool calling on {num_workers} workers")
            if self.config.show_tqdm
            else None
        )

        if num_workers <= 1:
            for agi in agent_inputs:
                subidx = agi["idx"]
                obs, reward, done, _ = execute_tool_call(agi, self.tokenizer, self.processor, pbar=pbar)
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done
        else:
            partial_tool_func = partial(execute_tool_call, tokenizer=self.tokenizer, processor=self.processor, pbar=pbar)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                raw_outputs = list(executor.map(partial_tool_func, agent_inputs))
            for agi, raw in zip(agent_inputs, raw_outputs):
                obs, reward, done, _ = raw
                subidx = agi["idx"]
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done

        return obs_list, reward_list, done_list, {}

    def reset(
        self,
        prompts: DataProto,
        vllm_inputs: Sequence[Dict[str, Any]],
        n: int = 1,
        **_: Any,
    ) -> None:
        """
        初始化每个样本的工具实例。
        """
        self.tools = []
        num_prompts = len(prompts)
        assert num_prompts == len(vllm_inputs), f"{num_prompts=}, {len(vllm_inputs)=}"

        for i in range(num_prompts):
            data_item = prompts[i]
            tool_name_raw = data_item.non_tensor_batch.pop(self.config.tool_name_key, "")
            tool_name = str(tool_name_raw) if tool_name_raw is not None else ""
            raw_prompt = data_item.non_tensor_batch.get("raw_prompt", None)

            vllm_input_item = vllm_inputs[i]
            multi_modal_data = vllm_input_item.get("multi_modal_data", None)
            origin_multi_modal_data = data_item.non_tensor_batch.get("origin_multi_modal_data", None)

            # ---------- 监控：诊断 tool 初始化参数 ----------
            if i == 0:
                _log_agent_event({
                    "event": "reset_first_sample_diag",
                    "timestamp": time.time(),
                    "tool_name_key": self.config.tool_name_key,
                    "tool_name_raw": str(tool_name_raw),
                    "tool_name": tool_name,
                    "has_raw_prompt": raw_prompt is not None,
                    "has_multi_modal_data": multi_modal_data is not None,
                    "has_origin_multi_modal_data": origin_multi_modal_data is not None,
                    "origin_mm_keys": list(origin_multi_modal_data.keys()) if isinstance(origin_multi_modal_data, dict) else None,
                    "non_tensor_batch_keys": list(data_item.non_tensor_batch.keys()),
                })
            # ---------------------------------------------------

            for _ in range(n):
                if tool_name:
                    try:
                        tool_obj = ToolBase.create(tool_name)
                        tool_obj.reset(
                            raw_prompt=raw_prompt,
                            multi_modal_data=deepcopy(multi_modal_data),
                            origin_multi_modal_data=deepcopy(origin_multi_modal_data),
                            cache_dir=getattr(self.config, "cache_dir", "/tmp/iqa_tool_output"),
                            cleanup_cache_on_close=getattr(self.config, "cleanup_cache_on_close", True),
                        )
                        self.tools.append(tool_obj)
                    except Exception as e:
                        # 工具初始化失败时记录日志并降级为 None
                        _log_agent_event({
                            "event": "tool_init_error",
                            "timestamp": time.time(),
                            "sample_idx": i,
                            "tool_name": tool_name,
                            "error": str(e),
                        })
                        print(f"[AGENT MONITOR] Tool init failed for sample {i}: {e}")
                        self.tools.append(None)
                else:
                    # 非 agent 数据，使用占位 None
                    self.tools.append(None)

    def close(self) -> None:
        for tool in self.tools:
            if tool is None:
                continue
            try:
                tool.close()
            except Exception as e:
                print(f"[AGENT MONITOR] Tool close failed: {e}")
        self.tools = []


def agent_rollout_loop(
    config,
    vllm_engine,
    vllm_inputs: List[Dict[str, Any]],
    prompts: DataProto,
    multi_modal_inputs: np.ndarray | None,
    sampling_params: SamplingParams,
    tokenizer,
    processor,
) -> DataProto:
    """
    多轮 agent rollout 主循环。

    基本流程（与 DeepEyes 中实现保持一致但依赖尽量简化）：
    - 使用 vLLM 逐步生成 action 文本
    - 调用 ParallelEnv 执行工具，得到新的多模态观测
    - 将观测再次拼接回 prompt，继续下一轮，直到：
      * 生成最终答案，或
      * 达到 max_turns，或
      * 序列长度达到上限
    - 返回：
      * response: 最后一轮回答对应的 token
      * action_mask: 哪些 token 来自模型 action（与工具观测区分）
      * attention_mask / position_ids: 完整多轮轨迹的掩码与 RoPE 索引
      * env_reward: 与 response 对齐的过程奖励
      * tool_cnt: 每个样本共调用了多少次工具
    """
    agent_sampling_params = sampling_params.clone()
    agent_sampling_params.detokenize = True
    agent_sampling_params.skip_special_tokens = False
    agent_sampling_params.spaces_between_special_tokens = False
    agent_sampling_params.n = 1
    agent_sampling_params.include_stop_str_in_output = True
    max_generated_tokens = min(config.agent.single_response_max_tokens, config.response_length)
    agent_sampling_params.max_tokens = max_generated_tokens

    # 自定义 stop 条件：在现有 custom_stop 基础上加入工具名 token
    custom_stop = list(getattr(config.agent, "custom_stop", []))
    tool_stop_tokens: List[str] = []
    try:
        from .envs.mm_process_engine.iqa_toolbox import ALL_TOOL_NAMES

        tool_stop_tokens = [f"<{name}>" for name in ALL_TOOL_NAMES]
    except Exception:  # pragma: no cover - 失败时静默
        tool_stop_tokens = []

    if tool_stop_tokens:
        custom_stop.extend(tool_stop_tokens)

    if custom_stop:
        prev_stop = sampling_params.stop or []
        agent_sampling_params.stop = prev_stop + custom_stop

    # NOTE: multi_modal_inputs 参数实际传入的是原始 multi_modal_data（含 PIL Image
    # 等非 Tensor 对象），不能直接用于 dp_actor 的 torch.cat。初始 prompt 图像的
    # 处理由 fsdp_workers._process_multi_modal_inputs 统一完成；这里 mm_input_list
    # 仅收集 agent 工具调用产生的已处理 Tensor 输入。
    batch_size = len(vllm_inputs)
    vllm_input_list: List[Dict[str, Any]] = []
    running_states: List[torch.Tensor] = []
    running_action_masks: List[torch.Tensor] = []
    running_attn_masks: List[torch.Tensor] = []
    reward_tensor_list: List[torch.Tensor] = []
    active_mask: List[bool] = []
    mm_input_list: List[Dict[str, Any]] = []
    tool_call_cnt_list: List[int] = []

    env = ParallelEnv(config.agent, tokenizer, processor)
    env.reset(prompts, vllm_inputs, n=sampling_params.n)

    # ---------- 监控：打印 env.reset 结果 ----------
    _n_none_tools = sum(1 for t in env.tools if t is None)
    _n_valid_tools = len(env.tools) - _n_none_tools
    _log_agent_event({
        "event": "env_reset",
        "timestamp": time.time(),
        "batch_size": batch_size,
        "n": sampling_params.n,
        "total_tools": len(env.tools),
        "valid_tools": _n_valid_tools,
        "none_tools": _n_none_tools,
        "max_turns": config.agent.max_turns,
        "single_response_max_tokens": config.agent.single_response_max_tokens,
        "stop_tokens": agent_sampling_params.stop,
    })
    if _n_none_tools > 0:
        print(
            f"[AGENT MONITOR] WARNING: {_n_none_tools}/{len(env.tools)} tools are None! "
            f"Check that env_name/raw_prompt/origin_multi_modal_data are in gen_batch."
        )
    # -------------------------------------------------

    # 监控：记录每个样本每轮的生成内容
    _monitor_traces: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size * sampling_params.n)]

    # interleave n samples if n > 1
    for i in range(batch_size):
        for _ in range(sampling_params.n):
            vllm_input_list.append(deepcopy(vllm_inputs[i]))
            prompt_ids = prompts.batch["input_ids"][i, :].clone()
            running_states.append(prompt_ids)

            prompt_mask = prompts.batch["attention_mask"][i, :].clone()
            running_action_masks.append(prompt_mask)
            running_attn_masks.append(prompt_mask)

            reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
            reward_tensor_list.append(reward_tensor)
            active_mask.append(True)
            mm_input_list.append({})  # 仅收集工具调用产生的 Tensor 输入
            tool_call_cnt_list.append(0)

    pg = vllm_ps.get_tp_group()
    max_total_length = config.prompt_length + config.response_length

    for step in range(config.agent.max_turns):
        if sum(active_mask) == 0:
            break

        active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
        active_vllm_inputs = [vi for vi, is_active in zip(vllm_input_list, active_mask) if is_active]
        actions: List[RequestOutput] = vllm_engine.generate(
            prompts=active_vllm_inputs,
            sampling_params=agent_sampling_params,
            use_tqdm=False,
        )

        if pg.is_first_rank:
            obs_results = env.step(active_indices, actions)
        else:
            obs_results = None
        observations, rewards, dones, _ = pg.broadcast_object(obs_results)

        for idx, obs, act, rew, done in zip(active_indices, observations, actions, rewards, dones):
            # ---------- 监控：记录每轮生成内容 ----------
            _action_text = act.outputs[0].text if act.outputs else ""
            _finish_reason = act.outputs[0].finish_reason if act.outputs else "none"
            _n_tokens = len(act.outputs[0].token_ids) if act.outputs else 0
            if idx < _AGENT_LOG_MAX_SAMPLES:
                _monitor_traces[idx].append({
                    "turn": step,
                    "action_text": _action_text[:500],  # 截断避免日志过大
                    "finish_reason": _finish_reason,
                    "n_tokens": _n_tokens,
                    "reward": rew,
                    "done": done,
                    "has_obs": bool(obs and ("prompt_token_ids_vllm" in obs or "prompt_token_ids_model" in obs)),
                    "tool_name": obs.get("tool_used", None) if isinstance(obs, dict) else None,
                })
            # -----------------------------------------------

            response_token_ids = torch.tensor(
                act.outputs[0].token_ids,
                dtype=torch.int64,
                device=running_states[idx].device,
            )
            running_states[idx] = torch.cat([running_states[idx], response_token_ids])
            vllm_input_list[idx]["prompt_token_ids"] = _concat_vllm_input(
                vllm_input_list[idx]["prompt_token_ids"],
                response_token_ids,
                tokenizer=tokenizer,
            )

            action_reward = torch.zeros_like(
                response_token_ids,
                dtype=torch.float,
                device=reward_tensor_list[idx].device,
            )
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward])
            reward_tensor_list[idx][-1] += rew

            action_mask = torch.ones_like(
                response_token_ids,
                dtype=torch.int64,
                device=running_action_masks[idx].device,
            )
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])
            running_attn_masks[idx] = torch.cat([running_attn_masks[idx], action_mask])

            if (
                running_states[idx].shape[-1] >= max_total_length
                or len(vllm_input_list[idx]["prompt_token_ids"]) >= max_total_length
            ):
                active_mask[idx] = False
                continue

            if done or step == config.agent.max_turns - 1:
                active_mask[idx] = False
                continue

            tool_call_cnt_list[idx] += 1

            # 将工具观测拼接回下一轮输入
            if "prompt_token_ids_vllm" in obs and "prompt_token_ids_model" in obs:
                obs_token_ids_vllm = obs["prompt_token_ids_vllm"]
                obs_token_ids_model = obs["prompt_token_ids_model"].to(running_states[idx].device)

                if (
                    len(vllm_input_list[idx]["prompt_token_ids"]) + len(obs_token_ids_vllm) >= max_total_length
                    or running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length
                ):
                    active_mask[idx] = False
                    continue

                vllm_input_list[idx]["prompt_token_ids"] = _concat_vllm_input(
                    vllm_input_list[idx]["prompt_token_ids"],
                    obs_token_ids_vllm,
                    tokenizer=tokenizer,
                )

                running_states[idx] = torch.cat([running_states[idx], obs_token_ids_model])
                obs_reward = torch.zeros(
                    len(obs_token_ids_model),
                    dtype=torch.float,
                    device=reward_tensor_list[idx].device,
                )
                reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)

                obs_mask = torch.zeros(
                    len(obs_token_ids_model),
                    dtype=torch.int64,
                    device=running_action_masks[idx].device,
                )
                running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
                attn_mask = torch.ones(
                    len(obs_token_ids_model),
                    dtype=torch.int64,
                    device=running_attn_masks[idx].device,
                )
                running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])

                mm_data = obs.get("multi_modal_data", {})
                if "image" in mm_data:
                    vllm_input_list[idx].setdefault("multi_modal_data", {}).setdefault("image", [])
                    vllm_input_list[idx]["multi_modal_data"]["image"] += mm_data["image"]

                mm_input = obs.get("multi_modal_inputs", {})
                if mm_input:
                    mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], mm_input)

            if (
                running_states[idx].shape[-1] >= max_total_length
                or len(vllm_input_list[idx]["prompt_token_ids"]) >= max_total_length
            ):
                active_mask[idx] = False

    env.close()

    # ---------- 监控：在 loop 结束后记录完整样本信息 ----------
    _log_sample_count = min(_AGENT_LOG_MAX_SAMPLES, len(running_states))
    for _si in range(_log_sample_count):
        # 解码完整响应（跳过 prompt 部分）
        _prompt_len = prompts.batch["attention_mask"][_si // sampling_params.n].sum().item()
        _response_ids = running_states[_si][int(_prompt_len):]
        try:
            _full_response = tokenizer.decode(
                _response_ids.cpu().tolist() if isinstance(_response_ids, torch.Tensor) else _response_ids,
                skip_special_tokens=False,
            )
        except Exception:
            _full_response = "<decode_error>"

        _log_agent_event({
            "event": "sample_complete",
            "timestamp": time.time(),
            "sample_idx": _si,
            "total_turns": len(_monitor_traces[_si]),
            "tool_calls": tool_call_cnt_list[_si],
            "total_tokens": len(running_states[_si]) if isinstance(running_states[_si], list) else running_states[_si].shape[0],
            "full_response": _full_response[:2000],  # 截断
            "turn_details": _monitor_traces[_si],
        })

    # 打印摘要到 stdout
    _total_samples = len(running_states)
    _avg_tool_calls = sum(tool_call_cnt_list) / max(len(tool_call_cnt_list), 1)
    print(
        f"[AGENT MONITOR] Rollout done: {_total_samples} samples, "
        f"avg_tool_calls={_avg_tool_calls:.1f}, "
        f"log_path={_AGENT_LOG_PATH}"
    )
    # --------------------------------------------------------

    target_device = prompts.batch["input_ids"].device
    running_states = [state[:max_total_length].tolist() for state in running_states]
    state_tensor = VF.pad_2d_list_to_length(running_states, tokenizer.pad_token_id, max_total_length).to(target_device)

    running_action_masks = [mask[:max_total_length].tolist() for mask in running_action_masks]
    action_mask_tensor = VF.pad_2d_list_to_length(running_action_masks, 0, max_total_length).to(target_device)

    running_attn_masks = [mask[:max_total_length].tolist() for mask in running_attn_masks]
    attn_mask_tensor = VF.pad_2d_list_to_length(running_attn_masks, 0, max_total_length).to(target_device)

    # 简单基于 attention_mask 构造 position_ids（与 EasyR1 dataset 中的默认逻辑保持一致）
    position_ids_tensor = torch.clip(attn_mask_tensor.cumsum(dim=-1) - 1, min=0)

    reward_tensor_list = [reward[:max_total_length].tolist() for reward in reward_tensor_list]
    reward_tensor = VF.pad_2d_list_to_length(reward_tensor_list, 0.0, max_total_length).to(target_device)

    tool_call_tensor = torch.tensor(tool_call_cnt_list, dtype=torch.float32, device=target_device).unsqueeze(1)
    return DataProto.from_dict(
        tensors={
            "response": state_tensor[:, -config.response_length :],
            "action_mask": action_mask_tensor,
            "attention_mask": attn_mask_tensor,
            "position_ids": position_ids_tensor,
            "env_reward": reward_tensor[:, -config.response_length :],
            "tool_cnt": tool_call_tensor,
        },
        non_tensors={"multi_modal_inputs": mm_input_list} if processor is not None else None,
    )
