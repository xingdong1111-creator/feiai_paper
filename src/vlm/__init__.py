"""one-in-one-out VLM 模块。
主线(Path B grounding, 坐标作输出): grounding.GroundingVLM + grounding_data。
对照 ablation: stage1 小3D CNN(model/dataset/encoder3d); 读出/生成式 stage2(stage2/stage2_data)。
"""
from .grounding import GroundingVLM, build_groundingvlm
from .model import build_oneinout, OneInOutNet
from .dataset import WholeCaseDataset, collate_cases

__all__ = ["GroundingVLM", "build_groundingvlm",
           "build_oneinout", "OneInOutNet", "WholeCaseDataset", "collate_cases"]
