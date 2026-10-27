"""one-in-one-out VLM 模块: 3D 编码 + 坐标条件 query 头(阶段1), 后续接冻结 LLM(阶段2)。"""
from .model import build_oneinout, OneInOutNet
from .dataset import WholeCaseDataset, collate_cases

__all__ = ["build_oneinout", "OneInOutNet", "WholeCaseDataset", "collate_cases"]
