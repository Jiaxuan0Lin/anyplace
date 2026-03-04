import os
import os.path as osp
from platform import processor
import torch
import numpy as np
from PIL import Image
import re

# --- Molmo 依赖库 ---
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# --- SAM 2 依赖库 ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# =================== 路径配置 ===================
# --- Molmo 模型路径 ---
MOLMO_MODEL_DIR = "/home/ustc/anyplace/anyplace/model_weights/Molmo-7B-D-0924"
# --- SAM 2 模型路径 ---
SAM2_MODEL_DIR = "/home/ustc/anyplace/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "sam2_hiera_l.yaml"


def load_perception_models(weights_dir):
    """
    统一从本地路径加载 Molmo 和 SAM 2 模型。
    """
    print("====== 开始加载视觉感知模型 ======")
    
    # ---------------------------------------------------------
    # 1. 加载 Molmo-7B
    # ---------------------------------------------------------
    print("正在加载 Molmo-7B...")
    molmo_processor = AutoProcessor.from_pretrained(
        MOLMO_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    molmo_model = AutoModelForCausalLM.from_pretrained(
        MOLMO_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    # ---------------------------------------------------------

    # 2. 加载 SAM 2 (Large 版本)
    print("正在加载 SAM 2...")
    sam2_checkpoint = SAM2_MODEL_DIR
    sam2_model_cfg = SAM2_MODEL_CFG
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=torch.device("cuda"))
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("====== 模型加载完毕！ ======")
    return molmo_processor, molmo_model, sam2_predictor


def get_placement_2d_point(image_path, text_prompt, processor, model):
    """
    输入图片和提示词，调用 Molmo 模型，返回目标在图片上的 [x, y] 像素坐标。
    """
    print(f"Molmo 正在观察图片并寻找: {text_prompt}")
    
    # 1. 加载并转换图片
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    # 2. 预处理 
    inputs = processor.process(
        images=[image],
        text=text_prompt
    )
    # 添加一个 Batch 维度，并转移到模型所在的设备上
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # 3. 模型生成
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # 4. 解码输出
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Molmo 的原始文字回复: {generated_text}")

    # 5. 正则表达式解析坐标
    match = re.search(r'x="([\d.]+)"\s+y="([\d.]+)"', generated_text)
    if not match:
        raise ValueError("未在 Molmo 输出中找到坐标信息。")
    pixel_x = int((float(match.group(1)) / 100.0) * image_width)
    pixel_y = int((float(match.group(2)) / 100.0) * image_height)

    # 6. 返回像素坐标
    return [pixel_x, pixel_y]


def get_sam2_mask_from_point(image_path, point_2d, predictor):
    """
    输入图片、2D像素点和已加载的 SAM 2 预测器，返回目标区域的 2D 布尔掩码。
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    predictor.set_image(image_np)
    input_point = np.array([point_2d])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    best_mask = masks[0]

    print(f"SAM 2 分割成功！Mask 尺寸为: {best_mask.shape}")
    return best_mask


def get_depth()