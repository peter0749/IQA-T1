# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class AgentConfig:
    """
    Agent 配置，用于控制是否启用多轮工具调用 rollout 以及相关超参。
    """

    activate_agent: bool = False
    # DataProto.non_tensor_batch 中用于查找工具名的 key，例如 "env_name"
    tool_name_key: str = ""
    # 每轮模型生成的最多 token 数（单轮内）
    single_response_max_tokens: int = 1024
    # 最多工具调用轮数
    max_turns: int = 4
    # 并发执行工具的 worker 数
    concurrent_workers: int = 1
    # 是否在工具执行时显示 tqdm
    show_tqdm: bool = False
    # 自定义停止字符串列表（例如 ["</answer_end>"]）
    custom_stop: list[str] = field(default_factory=list)
    # 工具缓存根目录
    cache_dir: str = "/tmp/iqa_tool_output"
    # rollout 结束时是否清理该 agent/tool 实例创建的缓存
    cleanup_cache_on_close: bool = True


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    val_override_config: dict[str, Any] = field(default_factory=dict)
    # Agent 相关配置，可选
    agent: AgentConfig = field(default_factory=AgentConfig)
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
