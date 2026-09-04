import numpy as np
import joblib
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from typing import List


class FishSpinePCAPriorTrainer:
    """
    鱼体 3D 脊柱骨架线 PCA 形状先验训练器
    """

    def __init__(self, n_target_points: int = 100, n_components: int = 5):
        """
        :param n_target_points: 重采样后的固定骨架点数 (默认 100 点)
        :param n_components: 保留的 PCA 主成分数量 (通常 3-6 个主成分即可解释 95%+ 的形状变异)
        """
        self.n_target_points = n_target_points
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.mean_shape_ = None  # 均值形状 (1, 3N)

    def resample_spine(self, spine_3d: np.ndarray) -> np.ndarray:
        """
        根据弧长对单条任意点数的 3D 骨架线进行等间距重采样
        :param spine_3d: 原始骨架点数组, shape=(M, 3)
        :return: 重采样后的骨架点数组, shape=(n_target_points, 3)
        """
        spine_3d = np.array(spine_3d, dtype=np.float64)
        if len(spine_3d) < 2:
            raise ValueError("单条骨架线至少需要 2 个 3D 坐标点")

        # 1. 计算累积弧长
        dists = np.linalg.norm(np.diff(spine_3d, axis=0), axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(dists)])
        total_length = cum_dist[-1]

        if total_length == 0:
            return np.tile(spine_3d[0], (self.n_target_points, 1))

        # 2. 建立弧长到坐标的 3D 参数插值函数
        interp_func = interp1d(cum_dist, spine_3d, axis=0, kind='linear')

        # 3. 均匀采样 fixed_points 个点
        target_cum_dists = np.linspace(0.0, total_length, self.n_target_points)
        resampled_spine = interp_func(target_cum_dists)

        return resampled_spine

    @staticmethod
    def align_to_origin(spine_3d: np.ndarray) -> np.ndarray:
        """
        中心化对齐：将骨架线的中心（或头部起点）平移至原点 (0, 0, 0)
        这里采用将头部（第一个点）作为原点对齐的方式，符合鱼体生长方向
        """
        return spine_3d - spine_3d[0]

    def fit(self, raw_spines_list: List[np.ndarray]) -> "FishSpinePCAPriorTrainer":
        """
        预处理所有骨架线并训练 PCA 模型
        :param raw_spines_list: n 条鱼的原始 3D 骨架线列表, 每个元素为 shape=(Mi, 3) 的点集
        """
        print(f" 开始预处理 {len(raw_spines_list)} 条原始 3D 骨架线...")

        processed_shapes = []

        for idx, raw_spine in enumerate(raw_spines_list):
            try:
                # 1. 弧长重采样到固定点数 (N, 3)
                resampled = self.resample_spine(raw_spine)

                # 2. 空间中心化对齐
                aligned = self.align_to_origin(resampled)

                # 3. 展平为 1D 特征向量 (3 * N,)
                flat_vector = aligned.reshape(-1)
                processed_shapes.append(flat_vector)

            except Exception as e:
                print(f" 警告: 第 {idx} 条骨架线处理失败, 已跳过 ({e})")

        if not processed_shapes:
            raise RuntimeError("没有有效的骨架数据供训练！")

        X_train = np.array(processed_shapes)  # shape=(n_samples, 3 * n_target_points)
        print(f" 训练矩阵构建完成，样本维度: {X_train.shape}")

        # 4. 训练 PCA 模型
        self.pca.fit(X_train)
        self.mean_shape_ = self.pca.mean_

        # 打印解释方差比
        explained_variance = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f" PCA 训练完成！前 {self.n_components} 个主成分累计解释方差: {explained_variance:.2f}%")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_):
            print(f"   - PC{i + 1}: {ratio * 100:.2f}%")

        return self

    def save_model(self, file_path: str = "fish_pca_prior.pkl"):
        """
        打包保存 PCA 模型、参数以及配置，供后续提取/补全调用
        """
        if self.mean_shape_ is None:
            raise RuntimeError("模型尚未训练，请先调用 .fit() 方法！")

        save_dict = {
            "pca_model": self.pca,
            "mean_shape": self.mean_shape_,  # 均值形状向量 (3N,)
            "components": self.pca.components_,  # 特征向量/基底矩阵
            "singular_values": self.pca.singular_values_,  # 奇异值/方差大小
            "n_target_points": self.n_target_points,  # 采样点数配置
            "n_components": self.n_components  # 主成分数量
        }

        joblib.dump(save_dict, file_path)
        print(f" PCA 形状先验模型已成功保存至: {file_path}")


if __name__ == "__main__":
    # 1. 模拟生成 n 条长度不一、包含随机摆尾弯曲的 3D 骨架线原始坐标数据
    np.random.seed(42)
    n_fishes = 50
    mock_raw_spines = []

    for _ in range(n_fishes):
        # 每条鱼的原始采样点数随机在 60 ~ 150 之间
        num_pts = np.random.randint(60, 150)

        # 沿 X 轴延伸体长，Y 轴正弦弯曲（模拟摆尾），Z 轴微小起伏
        t = np.linspace(0, np.random.uniform(20.0, 35.0), num_pts)
        x = t + np.random.normal(0, 0.1, num_pts)
        y = np.sin(t * 0.15) * np.random.uniform(1.0, 3.0)
        z = np.cos(t * 0.10) * 0.5

        # 随机叠加平移与旋转偏移，模拟未对齐的原始现实坐标
        raw_spine = np.column_stack([x, y, z]) + np.random.uniform(-50, 50, 3)
        mock_raw_spines.append(raw_spine)

    # 2. 实例化训练器并训练
    trainer = FishSpinePCAPriorTrainer(
        n_target_points=100,  # 统一对齐到 100 个 3D 点
        n_components=5  # 保留 5 个主成分
    )

    trainer.fit(mock_raw_spines)

    # 3. 保存模型文件
    model_save_path = "fish_spine_pca_model.pkl"
    trainer.save_model(model_save_path)
