import glob
import os
from typing import Dict, Tuple
import cv2
import numpy as np


def calibrate_single_camera(
    image_dir: str,
    pattern_size: Tuple[int, int] = (10, 7),
    square_size_mm: float = 20.0,
    air_dist_mm: float = 400.0,
    glass_thick_mm: float = 0.0,
    is_side_view: bool = False,
    visualize: bool = False,
    cam_name: str = "Camera",
) -> Tuple[Dict, np.ndarray]:
    """单相机内参标定核心函数 (已优化水下角点提取与畸变保存)"""
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    objpoints = []
    imgpoints = []

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(image_dir, ext)))

    if not images:
        raise FileNotFoundError(
            f"在 [{cam_name}] 文件夹 '{image_dir}' 中没有找到有效的标定图片！"
        )

    print(f"\n--- 开始标定 [{cam_name}] (图片数量: {len(images)}) ---")

    img_shape = None
    valid_count = 0
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # 创建 CLAHE 提升水下图像对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for img_path in sorted(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 水下预处理：自适应直方图均衡化，增强角点对比度
        gray_enhanced = clahe.apply(gray)

        if img_shape is None:
            img_shape = gray.shape[::-1]

        # 增加 CALIB_CB_ACCURACY 提高搜索精度
        ret, corners = cv2.findChessboardCorners(
            gray_enhanced,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            # 亚像素级精细化角点
            corners_subpix = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            valid_count += 1

            if visualize:
                vis_img = img.copy()
                cv2.drawChessboardCorners(
                    vis_img, pattern_size, corners_subpix, ret
                )
                cv2.imshow(f"Corners - {cam_name}", vis_img)
                cv2.waitKey(100)

    if visualize:
        cv2.destroyAllWindows()

    if valid_count < 3:
        raise ValueError(
            f"[{cam_name}] 识别成功的有效图片少于 3 张，无法标定！"
        )

    # 张正友标定法求解
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    fx, fy, cx, cy = (
        float(mtx[0, 0]),
        float(mtx[1, 1]),
        float(mtx[0, 2]),
        float(mtx[1, 2]),
    )

    # 计算重投影均方根误差 (RMSE)
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += error**2
        total_points += len(objpoints[i])

    mean_error = np.sqrt(total_error / total_points)

    print(f"[{cam_name}] 标定完成! 重投影误差 (RMSE): {mean_error:.4f} 像素")

    # 保存完整的内参矩阵以及畸变系数 dist
    cam_dict = {
        "fx": round(fx, 4),
        "fy": round(fy, 4),
        "cx": round(cx, 4),
        "cy": round(cy, 4),
        "dist_coeff": dist.ravel().tolist(),  # 保存 [k1, k2, p1, p2, k3] 畸变系数
        "air_dist_mm": air_dist_mm,
        "glass_thick_mm": glass_thick_mm if is_side_view else 0.0,
        "n_air": 1.000,
        "n_water": 1.333,
        "n_glass": 1.490 if is_side_view else 1.000,  # 若为钢化玻璃建议填 1.520
    }

    return cam_dict, dist


def calibrate_dual_cameras(
    top_image_dir: str,
    side_image_dir: str,
    pattern_size: Tuple[int, int] = (10, 7),
    square_size_mm: float = 20.0,
    top_air_dist_mm: float = 400.0,
    side_air_dist_mm: float = 200.0,
    side_glass_thick_mm: float = 10.0,
    visualize: bool = False,
) -> Tuple[Dict, Dict]:
    """【主入口】批量处理俯视与侧视双相机标定"""
    print("====== 双视角相机 (俯视 + 侧视) 联合标定程序启动 ======")

    # 1. 标定俯视相机
    top_cam_dict, _ = calibrate_single_camera(
        image_dir=top_image_dir,
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
        air_dist_mm=top_air_dist_mm,
        glass_thick_mm=0.0,
        is_side_view=False,
        visualize=visualize,
        cam_name="俯视相机 (Top-View)",
    )

    # 2. 标定侧视相机
    side_cam_dict, _ = calibrate_single_camera(
        image_dir=side_image_dir,
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
        air_dist_mm=side_air_dist_mm,
        glass_thick_mm=side_glass_thick_mm,
        is_side_view=True,
        visualize=visualize,
        cam_name="侧视相机 (Side-View)",
    )

    return top_cam_dict, side_cam_dict


if __name__ == "__main__":
    # 1. 模拟配置路径与标定板参数 (请根据实际项目路径修改)
    TOP_IMAGES_PATH = "./calibration_data/top_view"
    SIDE_IMAGES_PATH = "./calibration_data/side_view"

    # 标定板参数: (内部角点列数, 内部角点行数), 方格边长(mm)
    PATTERN_SIZE = (10, 7)
    SQUARE_SIZE_MM = 20.0

    # 物理水下/介质环境配置 (单位: mm)
    TOP_AIR_DIST_MM = 400.0       # 俯视相机镜头到水面的物理距离
    SIDE_AIR_DIST_MM = 200.0      # 侧视相机镜头到水箱玻璃外表面的物理距离
    SIDE_GLASS_THICK_MM = 10.0    # 侧视水箱玻璃厚度

    VISUALIZE = False              # 是否弹窗预览角点检测结果

    print("      水下双视角相机标定程序开始运行          ")

    try:
        # 2. 调用联合标定入口函数
        top_cam_dict, side_cam_dict = calibrate_dual_cameras(
            top_image_dir=TOP_IMAGES_PATH,
            side_image_dir=SIDE_IMAGES_PATH,
            pattern_size=PATTERN_SIZE,
            square_size_mm=SQUARE_SIZE_MM,
            top_air_dist_mm=TOP_AIR_DIST_MM,
            side_air_dist_mm=SIDE_AIR_DIST_MM,
            side_glass_thick_mm=SIDE_GLASS_THICK_MM,
            visualize=VISUALIZE,
        )

        # 3. 漂亮地打印字典参数，方便复制粘贴到后续的 3D 重建配置中
        import json

        print("\n【俯视相机 (Top-View) 相机参数】:")
        print(json.dumps(top_cam_dict, indent=4, ensure_ascii=False))

        print("\n【侧视相机 (Side-View) 相机参数】:")
        print(json.dumps(side_cam_dict, indent=4, ensure_ascii=False))


    except FileNotFoundError as e:
        print(f"\n[错误] 找不到文件夹或图片: {e}")
        print("请检查 `TOP_IMAGES_PATH` 和 `SIDE_IMAGES_PATH` 路径是否正确。")
    except Exception as e:
        print(f"\n[运行异常]: {e}")