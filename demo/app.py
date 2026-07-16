#!/usr/bin/env python3
"""
IQA-T1 Gradio Demo — Tool-based Visual Evidence Reasoning for Image Quality Assessment.
Self-contained: all dependencies are inside demo/.

Usage:
    python app.py                          # default model from HF Hub
    python app.py --model_path /path/to/model  # local checkpoint
"""

import sys
import os
import re
import glob
import tempfile
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import gradio as gr
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Spaces / ZeroGPU detection
# ---------------------------------------------------------------------------

is_spaces = os.getenv("SPACE_ID") is not None
spaces_available = False
GPU = None

if is_spaces:
    try:
        from spaces import GPU
        spaces_available = True
    except ImportError:
        pass

def gpu_decorator(func):
    """Apply ZeroGPU decorator on Spaces, no-op locally."""
    if spaces_available and GPU is not None:
        return GPU(func)
    return func

temp_dir = None
if not is_spaces:
    temp_dir = os.path.join(_DEMO_DIR, ".gradio_temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = temp_dir

# ---------------------------------------------------------------------------
# All 15 IQA tool names (hardcoded to avoid importing tools.py which needs cv2)
# ---------------------------------------------------------------------------

ALL_TOOL_NAMES = [
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

_TOOL_NAME_PATTERN = re.compile(r"<(" + "|".join(re.escape(n) for n in ALL_TOOL_NAMES) + r")>")

# ---------------------------------------------------------------------------
# Score colour mapping
# ---------------------------------------------------------------------------

def score_color(score: Optional[float]) -> str:
    """Return an emoji label + hex colour for a MOS score (1–5)."""
    if score is None:
        return "#888888", "⚪ N/A"
    if score >= 4.0:
        return "#2ecc71", f"🟢 Excellent ({score:.2f})"
    if score >= 3.0:
        return "#3498db", f"🔵 Good ({score:.2f})"
    if score >= 2.0:
        return "#f39c12", f"🟠 Fair ({score:.2f})"
    return "#e74c3c", f"🔴 Poor ({score:.2f})"


# ---------------------------------------------------------------------------
# Reasoning chain formatting
# ---------------------------------------------------------------------------

def format_reasoning_html(raw_response: str) -> Tuple[str, List[str]]:
    """Highlight tool-invocation tokens in the reasoning text.

    Returns (html_string, ordered_tool_names).
    """
    tool_names = [m.group(1) for m in _TOOL_NAME_PATTERN.finditer(raw_response)]

    def _replace(m: re.Match) -> str:
        name = m.group(1)
        return (
            f'<span style="background:#dbeafe;color:#1e40af;font-weight:700;'
            f'padding:2px 6px;border-radius:4px;display:inline-block;margin:1px 0">'
            f'🛠 &lt;{name}&gt;</span>'
        )

    html = _TOOL_NAME_PATTERN.sub(_replace, raw_response)
    html = html.replace("\n", "<br>")

    return html, tool_names


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_inferencer: Optional[object] = None  # IQAInference singleton


def load_model(model_path: str, device: str = "cuda") -> object:
    """Load (or return cached) IQAInference instance."""
    global _inferencer
    if _inferencer is not None and _inferencer.model_path == model_path:
        return _inferencer

    from infer import IQAInference

    if is_spaces:
        # On Spaces: allow HF Hub download
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    _inferencer = IQAInference(
        model_path=model_path,
        device=device,
        enable_tools=True,
        max_tool_calls=6,
        tool_save_dir=tempfile.mkdtemp(prefix="iqa_demo_"),
        keep_tool_images=True,
    )
    return _inferencer


# ---------------------------------------------------------------------------
# Core inference logic
# ---------------------------------------------------------------------------

@gpu_decorator
def run_inference(image: np.ndarray, model_path: str, temperature: float, top_p: float, do_sample: bool):
    """Run IQA inference on a single image."""
    if image is None:
        return _empty_results("Please upload an image first.")

    tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image.astype("uint8")).convert("RGB")
        pil_img.save(tmp_img.name)

        global _inferencer
        inferencer = load_model(model_path)

        # Apply generation config
        inferencer.gen_config.do_sample = do_sample
        inferencer.gen_config.temperature = temperature
        inferencer.gen_config.top_p = top_p
        if not do_sample:
            inferencer.gen_config.top_k = 0

        # Run inference
        result = inferencer.infer(tmp_img.name)
        score = result["score"]
        tools_used = result["tools_used"]
        response = result["raw_response"]

        # Collect tool images from save dir
        img_stem = Path(tmp_img.name).stem
        tool_dir = os.path.join(inferencer.tool_save_dir, img_stem)
        tool_path_map: dict = {}
        if os.path.isdir(tool_dir):
            for tname in tools_used:
                p = os.path.join(tool_dir, f"{tname}.png")
                if os.path.isfile(p):
                    tool_path_map[tname] = p
            if not tool_path_map:
                for p in sorted(glob.glob(os.path.join(tool_dir, "*.png"))):
                    tname = os.path.splitext(os.path.basename(p))[0]
                    tool_path_map[tname] = p

        # Score display
        color_hex, label = score_color(score)

        # Reasoning chain with highlighted tool tokens
        reasoning_html, tool_names = format_reasoning_html(response)

        # Tool selector options
        tool_choices = list(tool_names) if tool_names else []
        default_tool = tool_choices[0] if tool_choices else None

        # Preview image for the first tool
        first_preview = tool_path_map.get(tool_names[0], None) if tool_names else None
        first_label = tool_names[0] if tool_names else ""

        return (
            gr.update(value=f"<h1 style='text-align:center;color:{color_hex};font-size:72px;margin:0'>{score:.2f} / 5.00</h1><p style='text-align:center;font-size:18px'>{label}</p>"),
            reasoning_html,
            gr.update(choices=tool_choices, value=default_tool),
            first_preview if first_preview else None,
            f"**{first_label}**" if first_label else "",
            tool_path_map,
            f"✅ Done · {len(tools_used)} tools used · Score {score:.2f}" if score else "⚠️ No score extracted",
        )
    finally:
        try:
            os.unlink(tmp_img.name)
        except OSError:
            pass


def on_tool_select(selected: str, tool_path_map: dict):
    """When user selects a tool via Dropdown, update the preview image."""
    if not selected or not tool_path_map:
        return None, ""
    path = tool_path_map.get(selected)
    return path if path else None, f"**{selected}**"


def _empty_results(msg: str):
    return (
        gr.update(value="<h1 style='text-align:center;color:#888;font-size:48px'>-- / 5.00</h1>"),
        "",
        gr.update(choices=[], value=None),
        None, "", {}, msg,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

TITLE = """
<div align="center">
<h1>🛠️ IQA-T1: Tool-based Visual Evidence Reasoning for IQA</h1>
<p>
<a href="https://arxiv.org/abs/your-paper-id">📄 Paper</a> &nbsp;·&nbsp;
<a href="https://huggingface.co/zibuyu-02/IQA-T1">🤗 Model</a> &nbsp;·&nbsp;
<a href="https://huggingface.co/datasets/zibuyu-02/Q-Tool">📊 Dataset</a> &nbsp;·&nbsp;
<a href="https://github.com/zibuyu-02/IQA-T1">💻 Code</a>
</p>
</div>
"""


def build_demo(default_model_path: str = "zibuyu-02/IQA-T1"):
    with gr.Blocks(title="IQA-T1", theme=gr.themes.Soft(), css="""
        footer { visibility: hidden; }
        .tool-reasoning { max-height: 450px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; background: #fafafa; font-family: monospace; font-size: 14px; line-height: 1.7; }
    """) as demo:

        gr.HTML(TITLE)

        tool_path_state = gr.State({})

        with gr.Row():
            # ---- Left: Input ----
            with gr.Column(scale=2):
                input_image = gr.Image(
                    label="📷 Input Image",
                    type="numpy",
                    height=400,
                )
                gr.Examples(
                    examples=[str(p) for p in sorted(Path(_DEMO_DIR, "examples").glob("*.jpg"))],
                    inputs=input_image,
                    label="🖼️ Example Images",
                )
                with gr.Row():
                    btn_run = gr.Button("▶️ Start Assessment", variant="primary", size="lg")
                with gr.Accordion("⚙️ Advanced", open=False):
                    model_path = gr.Textbox(
                        label="Model Path",
                        value=default_model_path,
                        info="HuggingFace model ID or local path.",
                    )
                    do_sample = gr.Checkbox(
                        label="Enable Sampling",
                        value=False,
                        info="Off = greedy. On = sample with temperature.",
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                        info="Higher = more random.",
                    )
                    top_p = gr.Slider(
                        label="Top-p",
                        minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                        info="Nucleus sampling threshold.",
                    )

            # ---- Right: Results ----
            with gr.Column(scale=3):
                score_html = gr.HTML(
                    value="<h1 style='text-align:center;color:#888;font-size:48px'>-- / 5.00</h1>"
                )
                status = gr.Markdown("Upload an image and click **Start Assessment**.")

                with gr.Accordion("📝 Tool Reasoning Chain", open=True):
                    reasoning_html = gr.HTML(
                        value="<p style='color:#888'>Reasoning chain will appear here after assessment.</p>",
                        elem_classes=["tool-reasoning"],
                    )
                    tool_selector = gr.Dropdown(
                        label="🔧 Select Tool to Preview",
                        choices=[],
                        value=None,
                        interactive=True,
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        selected_tool = gr.Image(
                            label="🔍 Tool Preview",
                            type="filepath",
                            height=380,
                        )
                    with gr.Column(scale=1):
                        tool_label = gr.Markdown("")

        # ---- Event Bindings ----
        btn_run.click(
            fn=run_inference,
            inputs=[input_image, model_path, temperature, top_p, do_sample],
            outputs=[
                score_html, reasoning_html, tool_selector,
                selected_tool, tool_label, tool_path_state, status,
            ],
        )

        tool_selector.change(
            fn=on_tool_select,
            inputs=[tool_selector, tool_path_state],
            outputs=[selected_tool, tool_label],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IQA-T1 Gradio Demo")
    parser.add_argument("--model_path", type=str, default="zibuyu-02/IQA-T1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    if not is_spaces:
        print(f"Loading model: {args.model_path}  (device={args.device})")
        load_model(args.model_path, device=args.device)
        print("Model ready.")
    else:
        print("Running on HF Spaces with ZeroGPU — model will load on first inference.")

    demo = build_demo(default_model_path=args.model_path)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
