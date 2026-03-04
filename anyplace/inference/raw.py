import numpy as np

def get_sam2_mask_from_point(image_path, point_2d, predictor):
    """
    输入图片、2D像素点和已加载的 SAM 2 预测器，返回目标区域的 2D 布尔掩码 (Mask)。
    """
    print(f"SAM 2 接收到提示点：{point_2d}，正在分割目标区域...")
    
    # 1. 读取图片并转换为 Numpy 数组
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    
    # 2. 提取图像特征 (Image Embedding)
    # 这一步 SAM 2 会对整张图进行深度卷积分析，把边缘、光影等特征缓存起来。
    predictor.set_image(image_np)
    
    # 3. 构造 SAM 2 需要的输入格式
    # point_coords: 需要是一个 Nx2 的 Numpy 数组
    input_point = np.array([point_2d])
    
    # point_labels: 1 表示“前景(目标)”，0 表示“背景(不要的地方)”
    input_label = np.array([1])
    
    # 4. 执行预测
    # multimask_output=False 意味着我们不需要它给出“大、中、小”三种模棱两可的选择，
    # 而是直接让它输出置信度最高的那唯一一个 Mask。
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False, 
    )
    
    # masks 的形状通常是 (1, H, W)，表示 1 个通道，高 H，宽 W
    # 我们把这唯一的一个 Mask 提取出来，它是一个里面全是 True 或 False 的二维矩阵
    best_mask = masks[0]
    
    print(f"SAM 2 分割成功！Mask 尺寸为: {best_mask.shape}")
    return best_mask