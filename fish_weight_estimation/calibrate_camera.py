import glob
import os
from typing import Dict, Optional, Tuple
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
    """单相机内参标定核心函数 (由双视角标定函数统一调用)"""
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

    for img_path in sorted(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            corners_subpix = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            valid_count += 1

            if visualize:
                cv2.drawChessboardCorners(
                    img, pattern_size, corners_subpix, ret
                )
                cv2.imshow(f"Corners - {cam_name}", img)
                cv2.waitKey(150)

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

    # 提取基本内参
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

    # 构造对应 CameraParams 结构的参数字典
    cam_dict = {
        "fx": round(fx, 4),
        "fy": round(fy, 4),
        "cx": round(cx, 4),
        "cy": round(cy, 4),
        "air_dist_mm": air_dist_mm,
        "glass_thick_mm": glass_thick_mm if is_side_view else 0.0,
        "n_air": 1.000,
        "n_water": 1.333,
        "n_glass": 1.490 if is_side_view else 1.000,
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
    """【主入口】批量处理俯视与侧视双相机标定

    :param top_image_dir: 俯视相机棋盘格图片文件夹路径
    :param side_image_dir: 侧视相机棋盘格图片文件夹路径
    :param pattern_size: 棋盘格内部角点数量 (cols, rows)
    :param square_size_mm: 棋盘格物理边长 (mm)
    :param top_air_dist_mm: 俯视相机到水面的距离 (mm)
    :param side_air_dist_mm: 侧视相机到玻璃外表面的距离 (mm)
    :param side_glass_thick_mm: 侧视玻璃厚度 (mm)
    :param visualize: 是否弹窗查看识别结果
    :return: (top_cam_dict, side_cam_dict) 两个相机的参数字典
    """
    print("      双视角相机 (俯视 + 侧视) 联合标定程序启动      ")

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

    # 3. 输出汇总信息
    print(" 双视角相机标定最终结果 ")
    print("俯视相机参数 (top_cam_dict):")
    print(" ", top_cam_dict)
    print("\n侧视相机参数 (side_cam_dict):")
    print(" ", side_cam_dict)

    return top_cam_dict, side_cam_dict