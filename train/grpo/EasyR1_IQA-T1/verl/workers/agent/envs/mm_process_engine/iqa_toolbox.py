import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image

from ...tool_envs import ToolBase


# ---------------------------------------------------------------------------
# Import IQA tools from the ThinkVis-IQA project root
# ---------------------------------------------------------------------------
_IQA_ROOT = os.environ.get(
    "THINKVIS_IQA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../scripts")),
)
if _IQA_ROOT not in sys.path:
    sys.path.insert(0, _IQA_ROOT)

_TOOLS_IMPORT_ERROR: str | None = None
try:
    import tools as iqa_tools  # type: ignore
except Exception as e:  # pragma: no cover - best‑effort import
    iqa_tools = None
    _TOOLS_IMPORT_ERROR = str(e)

if iqa_tools is not None:
    ALL_TOOL_NAMES = getattr(iqa_tools, "ALL_TOOL_NAMES", [])
    TOOL_RESIZE_CONFIG = getattr(iqa_tools, "TOOL_RESIZE_CONFIG", {})
    resize_tool_image = getattr(iqa_tools, "resize_tool_image", None)
else:  # graceful fallback if tools are not available
    ALL_TOOL_NAMES = []
    TOOL_RESIZE_CONFIG = {}
    resize_tool_image = None


# 与评测脚本约定：下一轮输入仅由图像占位符 + 工具图像构成
IQA_OBS_IMAGE_PLACEHOLDER_ONLY = "<image>"


class IQAToolbox(ToolBase):
    """
    IQA 工具环境：
    - 接收一张原始图像
    - 根据模型输出的 <ToolName> 选择具体工具函数
    - 生成并返回工具图像，作为下一轮多模态输入
    """

    name = "iqa_toolbox"

    def __init__(self, _name: str, _desc: str, _params: Dict, **_: Any) -> None:
        super().__init__(name=self.name)
        self.multi_modal_data: Dict[str, Any] | None = None
        self.image_path: str | None = None
        self._temp_path: str | None = None
        self._save_dir = os.environ.get("IQA_TOOL_SAVE_DIR", "/tmp/iqa_tool_output")
        self._cleanup_cache_on_close = True
        self._owned_temp_files: set[str] = set()
        self._owned_episode_dirs: set[str] = set()

    def _apply_runtime_config(self, **kwargs: Any) -> None:
        cache_dir = kwargs.get("cache_dir")
        if cache_dir:
            self._save_dir = cache_dir
        cleanup_flag = kwargs.get("cleanup_cache_on_close")
        if cleanup_flag is not None:
            self._cleanup_cache_on_close = bool(cleanup_flag)

    def _register_temp_file(self, path: str | Path) -> None:
        self._owned_temp_files.add(str(path))

    def _register_episode_dir(self, path: str | Path) -> None:
        self._owned_episode_dirs.add(str(path))

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def _get_image_path(self, origin_multi_modal_data: Dict[str, Any]) -> str:
        # 兼容 dataset 侧使用 "images"（复数）和 vLLM 侧使用 "image"（单数）
        images = origin_multi_modal_data.get("image") or origin_multi_modal_data.get("images") or []
        if not images:
            raise ValueError(
                f"origin_multi_modal_data has no 'image' or 'images' key, "
                f"available keys: {list(origin_multi_modal_data.keys())}"
            )

        first = images[0]
        if isinstance(first, str):
            return first

        if hasattr(first, "save"):  # PIL.Image
            if self._temp_path and os.path.exists(self._temp_path):
                return self._temp_path
            tmp = Path(self._save_dir) / f"{uuid.uuid4().hex[:8]}.jpg"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            first.save(tmp)
            self._temp_path = str(tmp)
            self._register_temp_file(tmp)
            return self._temp_path

        raise TypeError(f"origin_multi_modal_data['image'][0] must be str or PIL.Image, got {type(first)}")

    def reset(self, raw_prompt: str | None, multi_modal_data: Dict[str, Any] | None, origin_multi_modal_data, **_: Any):
        """
        初始化一轮 episode。这里我们只需要原始图像路径。
        """
        del raw_prompt, multi_modal_data  # unused but kept for API 对齐
        self._apply_runtime_config(**_)
        if origin_multi_modal_data is None:
            raise ValueError("origin_multi_modal_data is required for IQAToolbox.")

        self.multi_modal_data = origin_multi_modal_data
        _has_images = bool(
            self.multi_modal_data.get("image") or self.multi_modal_data.get("images")
        )
        if not _has_images:
            raise ValueError(
                f"origin_multi_modal_data has no images, "
                f"available keys: {list(self.multi_modal_data.keys())}"
            )

        self.image_path = self._get_image_path(origin_multi_modal_data)

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _has_final_answer(action_string: str) -> bool:
        """
        检查模型输出中是否已经给出最终 IQA 答案（例如 <answer_start>...<answer_end>）。
        这里保持宽松，仅做简单子串匹配。
        """
        return "<answer_start>" in action_string and "<answer_end>" in action_string

    def _extract_tool_name(self, action_string: str) -> str | None:
        """
        从模型输出中解析工具名：
        - 主路径：查找最后一次出现的 <ToolName>（ALL_TOOL_NAMES 提供枚举）
        """
        if not ALL_TOOL_NAMES:
            return None

        last_name = None
        last_pos = -1
        for name in ALL_TOOL_NAMES:
            token = f"<{name}>"
            pos = action_string.rfind(token)
            if pos > last_pos:
                last_pos = pos
                last_name = name
        return last_name

    # ------------------------------------------------------------------ #
    # Core execute
    # ------------------------------------------------------------------ #
    def execute(self, action_string: str, **_: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        执行本轮模型输出：
        - 若已给出最终答案，则直接结束 episode
        - 否则解析工具名并运行 IQA 工具，返回仅含 <image> 占位符 + 工具图像的观测
        """
        # 1) 最终答案检查必须在工具可用性检查之前，确保即使工具加载失败
        #    模型也能正常结束 episode（否则会被困在无限循环中，收到错误信息但 done=False）
        if self._has_final_answer(action_string):
            return "", 0.0, True, {}

        if iqa_tools is None or not ALL_TOOL_NAMES:
            err = _TOOLS_IMPORT_ERROR or "tools module not loaded"
            msg = f"Error: IQA tools not loaded ({err})"
            return msg, 0.0, False, {"error": msg, "status": "failed"}

        # 2) 解析工具名
        tool_name = self._extract_tool_name(action_string)
        if not tool_name or tool_name not in ALL_TOOL_NAMES:
            msg = f"Error: unknown tool name {tool_name!r}"
            return msg, 0.0, False, {"error": msg, "status": "failed"}

        func = getattr(iqa_tools, tool_name, None)
        if func is None:
            msg = f"Error: tool {tool_name} not callable"
            return msg, 0.0, False, {"error": msg, "status": "failed"}

        # 3) 运行工具并加载输出图像
        try:
            assert self.image_path is not None, "image_path is not initialized in IQAToolbox.reset"
            episode_dir = os.path.join(self._save_dir, uuid.uuid4().hex[:8])
            os.makedirs(episode_dir, exist_ok=True)
            self._register_episode_dir(episode_dir)
            result = func(self.image_path, episode_dir)
            if result is None:
                raise RuntimeError(f"Tool {tool_name} returned None")
            save_path = result.get("save_path")
            if not save_path or not os.path.exists(save_path):
                raise RuntimeError(f"Tool {tool_name} did not produce a valid save_path")

            if tool_name in TOOL_RESIZE_CONFIG and resize_tool_image is not None:
                resize_tool_image(Path(save_path), Path(save_path), tool_name)

            tool_pil = Image.open(save_path).convert("RGB")
        except Exception as e:  # pragma: no cover - runtime safety
            msg = f"Error: {e}"
            return msg, 0.0, False, {"error": str(e), "status": "failed"}

        # 4) 返回仅含占位符 + 工具图像的观测（与推理端行为保持一致）
        obs: Dict[str, Any] = {
            "prompt": IQA_OBS_IMAGE_PLACEHOLDER_ONLY,
            "multi_modal_data": {"image": [tool_pil]},
        }
        reward = 0.0  # 过程奖励留空，由最终 reward function 决定
        done = False
        info = {"status": "success", "tool_used": tool_name}
        return obs, reward, done, info

    def close(self) -> None:
        if not self._cleanup_cache_on_close:
            return

        for path in list(self._owned_temp_files):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass
        self._owned_temp_files.clear()

        for path in sorted(self._owned_episode_dirs, key=len, reverse=True):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception:
                pass
        self._owned_episode_dirs.clear()

        try:
            if self._temp_path and os.path.isfile(self._temp_path):
                os.remove(self._temp_path)
        except Exception:
            pass
        self._temp_path = None
