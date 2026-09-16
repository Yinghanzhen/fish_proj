from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, List, Optional

@dataclass
class TopFishData:
  """俯视角单条鱼的数据结构"""

  fish_id: int
  bbox: Tuple[float, float, float, float]  # 2D 检测框 [xmin, ymin, xmax, ymax]
  head_kpt_2d: Tuple[float, float]  # 鱼头关键点 2D 像素坐标 (x, y)
  tail_kpt_2d: Tuple[float, float]  # 鱼尾关键点 2D 像素坐标 (x, y)

  # 待后期计算的 3D 数据与几何特征（默认为 None / 0.0）
  weight: float = None
  head_kpt_3d: Optional[Tuple[float, float, float]] = None  # 鱼头 3D 空间坐标 (X, Y, Z)
  tail_kpt_3d: Optional[Tuple[float, float, float]] = None  # 鱼尾 3D 空间坐标 (X, Y, Z)
  spine_2d: Optional[List[Optional[List[float]]]] = None  # [x, y] 或 [None, None] 占位
  spine_3d: Optional[List[List[float]]] = None  # 骨架线 3D 坐标阵列，Shape 为 (N, 3)
  spine_length: float = 0.0  # 骨架线沿线总弧长 (体长)

  max_body_width: float = 0.0  # 法线切片提取的最大体宽 (物理尺寸)
  mean_body_width: float = 0.0  # 平均体宽 (物理尺寸)
  proj_area_2d: float = 0.0  # 俯视投影物理面积
  back_curvature: float = (
      0.0  # 2D弧长展平 + 1D去趋势解耦后的背部曲率 (多项式二次项系数 a)
  )
  back_height: float = 0.0  # 剥离俯仰倾斜后的净背部最大厚度 (Delta Z_rel)

  # 导出计算字段：检测框中心点 (自动生成，便于后续代价值匹配)
  x_mid: float = field(init=False)
  y_mid: float = field(init=False)

  mask: Optional[np.ndarray] = None

  def __post_init__(self):
    self.x_mid = (self.bbox[0] + self.bbox[2]) / 2.0
    self.y_mid = (self.bbox[1] + self.bbox[3]) / 2.0
    if self.spine_3d is not None and not isinstance(self.spine_3d, np.ndarray):
      self.spine_3d = np.array(self.spine_3d, dtype=np.float64)

    # 自动转换 mask 数据类型（防止传入的是 List[List[int]]）
    if self.mask is not None and not isinstance(self.mask, np.ndarray):
      self.mask = np.array(self.mask, dtype=bool)


@dataclass
class CameraParams:
    """相机内外参及物理介质参数配置"""
    fx: float  # 焦距 fx
    fy: float  # 焦距 fy
    cx: float  # 光心 cx
    cy: float  # 光心 cy
    air_dist_mm: float  # 俯视: 相机到水面距离 (mm) | 侧视: 相机到玻璃外表面距离 (mm)
    n_air: float = 1.000  # 空气折射率
    n_water: float = 1.333  # 水折射率