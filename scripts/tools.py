import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import traceback

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

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


TOOL_RESIZE_CONFIG = {
    # spatial maps
    "GradientMagnitudeMap": 384,
    "GradientOrientationCoherenceMap": 384,
    "HighFrequencyResidualMap": 384,
    "DoGSharpnessMap": 384,
    "NoiseResidualMap": 384,
    "LocalGradientDeviationMap": 384,
    "MSCNNormalizedLuminanceMap": 384,
    "PerceptualColorfulnessMap": 384,
    "ChromaticDeviationMap": 384,

    # histogram / statistics
    # "GradientMagnitudeHistogram": 384,
    # "LuminanceHistogram": 384,
    # "MSCNDistributionHistogram": 384,

    # frequency
    "FourierMagnitudeSpectrum": 384,

    # binary / mask
    "ExtremeLuminanceMap": 384,
    "ColorSaturationClippingMap": 384,
}

# =========================
# Tool image resize
# =========================

def resize_keep_ratio_long_side(
    img,
    target_long_side,
    interp=cv2.INTER_AREA
):
    """
    Resize image keeping aspect ratio.
    Long side is resized to target_long_side.
    """
    h, w = img.shape[:2]

    if max(h, w) == target_long_side:
        return img

    if h > w:
        new_h = target_long_side
        new_w = round(w * target_long_side / h)
    else:
        new_w = target_long_side
        new_h = round(h * target_long_side / w)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interp
    )
    return resized



def resize_tool_image(raw_path, resized_path, tool_name):
    img = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return

    target_long = TOOL_RESIZE_CONFIG.get(tool_name)

    # 插值策略
    if tool_name in [
        "ExtremeLuminanceMap",
        "ColorSaturationClippingMap",
    ]:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA

    if target_long is None:
        resized = img
    else:
        resized = resize_keep_ratio_long_side(
            img,
            target_long,
            interp
        )

    resized_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(resized_path), resized)


# =========================
# Raw completeness check
# =========================

def is_valid_image(path):
    if not path.exists():
        return False
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img is not None and img.size > 0

def raw_tools_complete(raw_dir: Path):
    if not raw_dir.exists():
        return False
    for name in ALL_TOOL_NAMES:
        if not is_valid_image(raw_dir / f"{name}.png"):
            return False
    return True

# =========================
# Resize from raw only
# =========================

def generate_resized_from_raw(raw_dir, resized_dir):
    for tool_name in ALL_TOOL_NAMES:
        raw_img = raw_dir / f"{tool_name}.png"
        resized_img = resized_dir / f"{tool_name}.png"

        if not raw_img.exists():
            continue
        if resized_img.exists():
            continue

        resize_tool_image(raw_img, resized_img, tool_name)


# def batch_process_image_folder(
#     image_folder,
#     save_dir="./save",
#     error_log="error_log.txt",
#     skip_existing=True   # 🔥 是否跳过已处理图片
# ):
#     image_folder = Path(image_folder)

#     if not image_folder.exists() or not image_folder.is_dir():
#         print(f"Error: '{image_folder}' is not a valid directory.")
#         return

#     image_files = [
#         p for p in image_folder.iterdir()
#         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
#     ]

#     print(f"Found {len(image_files)} images in '{image_folder}'")

#     error_log = Path(error_log)
#     error_log.write_text("")

#     for img_path in tqdm(image_files, desc="Processing images", unit="img"):

#         # ===== ✅ 跳过已处理图像 =====
#         if skip_existing:
#             out_dir = Path(save_dir) / img_path.stem
#             if out_dir.exists():
#                 continue

#         try:
#             generate_all_maps(str(img_path), save_dir)

#         except Exception:
#             with error_log.open("a", encoding="utf-8") as f:
#                 f.write(f"{img_path.name}\n")
#                 f.write(traceback.format_exc())
#                 f.write("\n" + "-" * 60 + "\n")

#     print("\nBatch processing completed!")
#     print(f"If any error occurred, see: {error_log}")

# Function to save the result as an image
def save_image(output, filename):
    plt.imsave(filename, output, cmap='gray')

# Function to create directory structure
def create_save_directory(base_dir, image_name):
    """创建保存目录结构，返回包含子目录的路径"""
    # 提取不带扩展名的图片名称
    image_name_no_ext = Path(image_name).stem
    
    # 创建基础保存目录
    save_dir = Path(base_dir) / image_name_no_ext
    save_dir.mkdir(parents=True, exist_ok=True)
    
    return str(save_dir)


def get_tool_save_path(image_path, save_dir, tool_name):
    """
    获取工具图像的保存路径，并检查是否已存在
    
    Args:
        image_path: 原始图像路径
        save_dir: 保存目录
        tool_name: 工具名称
        
    Returns:
        tuple: (target_dir, save_path, already_exists)
    """
    image_name = Path(image_path).name
    target_dir = create_save_directory(save_dir, image_name)
    save_path = f"{target_dir}/{tool_name}.png"
    already_exists = os.path.exists(save_path)
    
    return target_dir, save_path, already_exists

def normalize_grad_fixed(grad_mag, clip_val=255):
    grad_clipped = np.clip(grad_mag, 0, clip_val)
    grad_norm = (grad_clipped / clip_val) * 255.0
    return grad_norm.astype(np.uint8)


# =========================
# 结构（Structure）
# =========================
# 1. GradientMagnitudeMap
def GradientMagnitudeMap(image_path, save_dir="./save", skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "GradientMagnitudeMap"
    )
    if skip_existing and already_exists:
        return {"GradientMagnitudeMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx**2 + gy**2)

    grad_mag_norm = normalize_grad_fixed(grad_mag, clip_val=255)

    # 可视化为热力图
    plt.imsave(save_path, grad_mag_norm, cmap='hot', format='png', origin='upper')
    plt.close()

    return {"GradientMagnitudeMap": grad_mag_norm, "save_path": save_path}

def GradientMagnitudeHistogram(image_path, save_dir="./save", figsize=(8, 5), skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "GradientMagnitudeHistogram"
    )
    if skip_existing and already_exists:
        return {"GradientMagnitudeHistogram": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Sobel 梯度
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # 梯度幅值
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # 拉平成一维
    grad_data = grad_mag.flatten()

    # 绘制直方图
    plt.figure(figsize=figsize)
    plt.hist(
        grad_data,
        bins=100,
        density=True,
        color='gray',
        # log=True,
        alpha=0.7,
        edgecolor='black'
    )

    mean_grad = np.mean(grad_data)
    plt.axvline(mean_grad, color='red', linestyle='--',
                label=f'Mean: {mean_grad:.2f}')

    plt.xlabel('Gradient Magnitude', fontsize=20)
    plt.ylabel('Normalized Frequency', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")

    return {
        "GradientMagnitudeHistogram": grad_data,
        "save_path": save_path
    }

# 2. GradientOrientationConsistencyMap
def GradientOrientationCoherenceMap(image_path, save_dir="./save", ksize=15, skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "GradientOrientationCoherenceMap"
    )
    if skip_existing and already_exists:
        return {"GradientOrientationCoherenceMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # 梯度方向（单位向量）
    mag = np.sqrt(gx**2 + gy**2)
    zero_mask = mag < 1e-6
    ux = np.zeros_like(gx)
    uy = np.zeros_like(gy)
    ux[~zero_mask] = gx[~zero_mask] / mag[~zero_mask]
    uy[~zero_mask] = gy[~zero_mask] / mag[~zero_mask]

    # 局部方向平均
    # mean_ux = cv2.GaussianBlur(ux, (ksize, ksize), 0)
    # mean_uy = cv2.GaussianBlur(uy, (ksize, ksize), 0)

    kernel = np.ones((ksize, ksize)) / (ksize * ksize)
    mean_ux = cv2.filter2D(ux, -1, kernel)
    mean_uy = cv2.filter2D(uy, -1, kernel)

    # 一致性 = 平均向量长度
    consistency = np.sqrt(mean_ux**2 + mean_uy**2)

    consistency_norm = cv2.normalize(
        consistency, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(str(save_path), consistency_norm)

    return {"GradientOrientationCoherenceMap": consistency_norm, "save_path": save_path}


# =========================
# 清晰度（Sharpness）
# =========================
# 1. HighFrequencyEnergyMap
def HighFrequencyResidualMap(image_path, save_dir="./save", sigma=3.0, skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "HighFrequencyResidualMap"
    )
    if skip_existing and already_exists:
        return {"HighFrequencyResidualMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    low_freq = cv2.GaussianBlur(gray, (0, 0), sigma)
    high_freq = gray - low_freq
    hf_energy = np.abs(high_freq)
    # hf_energy = high_freq**2

    hf_norm = cv2.normalize(
        hf_energy, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(str(save_path), hf_norm)

    return {"HighFrequencyResidualMap": hf_norm, "save_path": save_path}

# 2. DoGSharpnessMap
def DoGSharpnessMap(image_path, save_dir="./save", sigma1=1.0, sigma2=2.5, skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "DoGSharpnessMap"
    )
    if skip_existing and already_exists:
        return {"DoGSharpnessMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)

    dog = np.abs(g1 - g2)

    dog_norm = cv2.normalize(
        dog, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(str(save_path), dog_norm)

    return {"DoGSharpnessMap": dog_norm, "save_path": save_path}

# =========================
# 噪声（Noise）
# =========================
# 1. NoiseResidualMap
def NoiseResidualMap(image_path, save_dir="./save", skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "NoiseResidualMap"
    )
    if skip_existing and already_exists:
        return {"NoiseResidualMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. 高频残差（轻度）
    low = cv2.GaussianBlur(gray, (0, 0), 1.0)
    high = gray - low

    # 2. 结构检测（边缘）
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    grad_smooth = cv2.GaussianBlur(grad_mag, (5, 5), 1.0)
    g_norm = grad_smooth / (grad_smooth.max() + 1e-6)
    structure_strength = g_norm ** 0.6   # <1：增强弱结构
    structure_mask = 1.0 - structure_strength

    # 3. 噪声残差（抑制结构）
    noise_residual = np.abs(high) * structure_mask

    # 4. 归一化（非常重要）
    noise_norm = cv2.normalize(
        noise_residual, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(str(save_path), noise_norm)

    return {"NoiseResidualMap": noise_norm,"save_path": save_path}

# =========================
# 伪影（Artifacts）
# =========================
# 3. FFTMagnitudeSpectrum
def FourierMagnitudeSpectrum(image_path, save_dir="./save", skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "FourierMagnitudeSpectrum"
    )
    if skip_existing and already_exists:
        return {"FourierMagnitudeSpectrum": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    # 1. 灰度 + float
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # 3. 幅度谱（log 压缩）
    magnitude = np.log(1 + np.abs(fshift))

    # 4. 归一化到 0–255
    magnitude_norm = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 5. 可视化
    cv2.imwrite(str(save_path), magnitude_norm)

    print(f"Saved: {save_path}")
    return {"FourierMagnitudeSpectrum": magnitude_norm, "save_path": save_path}

# =========================
# 亮度（Luminance）
# =========================

# 1. BrightnessHistogram
def LuminanceHistogram(image_path, save_dir="./save", figsize=(8, 5), skip_existing=True):
    """
    Brightness distribution histogram based on standard luminance formula:
    Y = 0.299R + 0.587G + 0.114B
    """
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "LuminanceHistogram"
    )
    if skip_existing and already_exists:
        return {"LuminanceHistogram": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    
    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 标准亮度计算
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = brightness.astype(np.uint8)
    
    # 拉平成一维
    brightness_data = brightness.flatten()
    
    # 绘图
    plt.figure(figsize=figsize)
    plt.hist(
        brightness_data,
        bins=50,
        range=(0, 255),
        density=True,
        color='gray',
        alpha=0.7,
        edgecolor='black'
    )
    
    mean_brightness = np.mean(brightness_data)
    plt.axvline(mean_brightness, color='red', linestyle='--',
                label=f'Mean: {mean_brightness:.1f}')
    
    plt.xlabel('Luminance Value', fontsize=20)
    plt.ylabel('Normalized Frequency', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    
    return {"LuminanceHistogram": brightness_data, "save_path": save_path}


# 3. ExtremeLuminanceMap
def ExtremeLuminanceMap(image_path, save_dir="./save",
                    low_thresh=30, high_thresh=225, skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "ExtremeLuminanceMap"
    )
    if skip_existing and already_exists:
        return {"ExtremeLuminanceMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    saturation = np.zeros_like(gray, dtype=np.uint8)
    saturation[gray <= low_thresh] = 255
    saturation[gray >= high_thresh] = 255

    cv2.imwrite(str(save_path), saturation)

    return {"ExtremeLuminanceMap": saturation, "save_path": save_path}    

# =========================
# 色彩（Color）
# =========================

# 1. ColorfulnessMap
def PerceptualColorfulnessMap(image_path, save_dir="./save", skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "PerceptualColorfulnessMap"
    )
    if skip_existing and already_exists:
        return {"PerceptualColorfulnessMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    rg = R - G
    yb = 0.5 * (R + G) - B

    colorfulness = np.sqrt(rg**2 + yb**2)

    color_norm = cv2.normalize(
        colorfulness, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(str(save_path), color_norm)

    return {"PerceptualColorfulnessMap": color_norm, "save_path": save_path}

# 2. ColorDeviationMap
def ChromaticDeviationMap(image_path, save_dir="./save", skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "ChromaticDeviationMap"
    )
    if skip_existing and already_exists:
        return {"ChromaticDeviationMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1] - 128
    b = lab[:, :, 2] - 128

    deviation = np.sqrt(a**2 + b**2)

    dev_norm = cv2.normalize(
        deviation, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(str(save_path), dev_norm)

    return {"ChromaticDeviationMap": dev_norm, "save_path": save_path}

# 3. ColorSaturationClippingMap
def ColorSaturationClippingMap(
    image_path, save_dir="./save", sat_thresh=245, skip_existing=True
):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "ColorSaturationClippingMap"
    )
    if skip_existing and already_exists:
        return {"ColorSaturationClippingMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    clip_map = np.zeros_like(saturation, dtype=np.uint8)
    clip_map[saturation >= sat_thresh] = 255

    cv2.imwrite(str(save_path), clip_map)

    return {"ColorSaturationClippingMap": clip_map, "save_path": save_path}


# =========================
# 自然度（Naturalness）
# =========================
# 1. MSCNCoefficientMap
def MSCNNormalizedLuminanceMap(image_path, save_dir="./save", ksize=7, eps=1e-8, skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "MSCNNormalizedLuminanceMap"
    )
    if skip_existing and already_exists:
        return {"MSCNNormalizedLuminanceMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mu = cv2.GaussianBlur(gray, (ksize, ksize), 1.166)
    mu_sq = mu * mu
    sigma = cv2.GaussianBlur(gray * gray, (ksize, ksize), 1.166)
    sigma = np.sqrt(np.abs(sigma - mu_sq))

    mscn = (gray - mu) / (sigma + eps)

    mscn_norm = cv2.normalize(
        mscn, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(str(save_path), mscn_norm)

    return {"MSCNNormalizedLuminanceMap": mscn_norm, "save_path": save_path}


def MSCNDistributionHistogram(image_path, save_dir="./save", figsize=(8, 5), ksize=7, sigma=1.166, skip_existing=True):
    """
    MSCN (Mean Subtracted Contrast Normalized) coefficient histogram.
    Widely used in NR-IQA methods such as BRISQUE / NIQE.
    """
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "MSCNDistributionHistogram"
    )
    if skip_existing and already_exists:
        return {"MSCNDistributionHistogram": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    # ===== 灰度化 =====
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ===== 计算局部均值和方差 =====
    mu = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    mu_sq = mu * mu

    sigma_local = cv2.GaussianBlur(gray * gray, (ksize, ksize), sigma)
    sigma_local = np.sqrt(np.abs(sigma_local - mu_sq))

    # ===== MSCN 系数 =====
    eps = 1e-8
    mscn = (gray - mu) / (sigma_local + eps)

    # ===== 拉平成一维样本 =====
    mscn_data = mscn.flatten()

    # 可选：去除极端异常值（更稳定）
    mscn_data = mscn_data[np.abs(mscn_data) < 10]

    # ===== 绘制直方图 =====
    plt.figure(figsize=figsize)
    plt.hist(
        mscn_data,
        bins=100,
        density=True,
        color='gray',
        alpha=0.7,
        edgecolor='black'
    )

    mean_val = np.mean(mscn_data)
    plt.axvline(
        mean_val,
        color='red',
        linestyle='--',
        label=f'Mean: {mean_val:.2f}'
    )

    plt.xlabel('MSCN Coefficient Value', fontsize=20)
    plt.ylabel('Normalized Frequency', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")

    return {
        "MSCNDistributionHistogram": mscn_data,
        "save_path": save_path
    }


# 2. GradientDistributionDeviationMap
def LocalGradientDeviationMap(image_path, save_dir="./save", skip_existing=True):
    # 检查是否已存在
    target_dir, save_path, already_exists = get_tool_save_path(
        image_path, save_dir, "LocalGradientDeviationMap"
    )
    if skip_existing and already_exists:
        return {"LocalGradientDeviationMap": None, "save_path": save_path}
    
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    local_mean = cv2.GaussianBlur(grad_mag, (7, 7), 1.5)
    deviation = np.abs(grad_mag - local_mean)

    dev_norm = cv2.normalize(
        deviation, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(str(save_path), dev_norm)

    return {"LocalGradientDeviationMap": dev_norm, "save_path": save_path}




# Main function to process an image and generate all MPB maps
def generate_all_maps(image_path, save_dir="./save_tools"):
    """处理图片并生成所有特征图"""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # print(f"Processing image: {image_path}")
    # print("-" * 50)
    
    # 调用所有函数生成特征图
    results = {}
    
    results['GradientMagnitudeMap'] = GradientMagnitudeMap(image_path, save_dir)
    results['GradientOrientationCoherenceMap'] = GradientOrientationCoherenceMap(image_path, save_dir)
    results['GradientMagnitudeHistogram'] = GradientMagnitudeHistogram(image_path, save_dir)

    results['HighFrequencyResidualMap'] = HighFrequencyResidualMap(image_path, save_dir)
    results['DoGSharpnessMap'] = DoGSharpnessMap(image_path, save_dir)
    results['FourierMagnitudeSpectrum'] = FourierMagnitudeSpectrum(image_path, save_dir)

    results['NoiseResidualMap'] = NoiseResidualMap(image_path, save_dir)

    results['LuminanceHistogram'] = LuminanceHistogram(image_path, save_dir)
    results['ExtremeLuminanceMap'] = ExtremeLuminanceMap(image_path, save_dir)

    results['PerceptualColorfulnessMap'] = PerceptualColorfulnessMap(image_path, save_dir)
    results['ChromaticDeviationMap'] = ChromaticDeviationMap(image_path, save_dir)
    results['ColorSaturationClippingMap'] = ColorSaturationClippingMap(image_path, save_dir)

    results['MSCNNormalizedLuminanceMap'] = MSCNNormalizedLuminanceMap(image_path, save_dir)
    results['MSCNDistributionHistogram'] = MSCNDistributionHistogram(image_path, save_dir)
    results['LocalGradientDeviationMap'] = LocalGradientDeviationMap(image_path, save_dir)

    # print("-" * 50)
    # print("All maps have been saved to their respective directories!")
    
    return results

# 批量处理示例函数
def batch_process_images(image_paths, save_dir="./save"):
    """批量处理多张图片"""
    all_results = {}
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        results = generate_all_maps(image_path, save_dir)
        all_results[Path(image_path).name] = results
    
    return all_results

# =========================
# Process one image
# =========================

def process_one_image(image_path, raw_root, resized_root):
    image_name = Path(image_path).stem
    raw_dir = Path(raw_root) / image_name
    resized_dir = Path(resized_root) / image_name

    # Case 1: raw 已完整 → 只补 resize
    if raw_tools_complete(raw_dir):
        generate_resized_from_raw(raw_dir, resized_dir)
        return

    # Case 2: raw 不存在或不完整 → 生成 raw
    generate_all_maps(image_path, raw_root)

    # 再补 resize
    generate_resized_from_raw(raw_dir, resized_dir)

# =========================
# Batch processing
# =========================

def batch_process_image_folder(
    image_folder,
    raw_root="./tools_raw",
    resized_root="./tools_resized",
    error_log="error_log.txt"
):
    image_folder = Path(image_folder)
    error_log = Path(error_log)
    error_log.write_text("")

    image_files = [
        p for p in image_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Found {len(image_files)} images.")

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            process_one_image(
                img_path,
                raw_root,
                resized_root
            )
        except Exception:
            with error_log.open("a", encoding="utf-8") as f:
                f.write(f"{img_path.name}\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-" * 60 + "\n")

    print("Done.")
    print(f"Error log: {error_log}")

# 使用示例：
if __name__ == "__main__":
    # 单张图片处理示例:
    #   image_path = "path/to/your/image.jpg"
    #   generate_all_maps(image_path)
    #
    # 批量处理示例:
    #   image_list = ["img1.jpg", "img2.jpg"]
    #   batch_process_images(image_list)
    #
    #   batch_process_image_folder(
    #       image_folder="YOUR_IMAGE_DIR",
    #       raw_root="YOUR_RAW_ROOT",
    #       resized_root="YOUR_RESIZED_ROOT",
    #   )
    pass