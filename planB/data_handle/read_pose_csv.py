import pandas as pd
import os
import glob


def get_best_pose_map50(csv_path):
    """
    读取姿态 mAP50 的最高值和对应轮次

    Args:
        csv_path: results.csv 文件路径
    """
    # 读取文件
    df = pd.read_csv(csv_path)

    print(f"📁 文件: {os.path.basename(os.path.dirname(csv_path))}")

    # 查找姿态 mAP50 列
    map50_col = None
    for col in df.columns:
        if 'mAP50(P)' in col:
            map50_col = col
            break

    if map50_col is None:
        print("❌ 未找到姿态 mAP50 列")
        print(f"可用的列: {[c for c in df.columns if '(P)' in c]}")
        return None

    # 找最大值和对应轮次
    best_idx = df[map50_col].idxmax()
    best_value = df[map50_col].max()
    best_epoch = df.loc[best_idx, 'epoch'] if 'epoch' in df.columns else best_idx

    # 同时找 mAP50-95
    map95_col = None
    for col in df.columns:
        if 'mAP50-95(P)' in col:
            map95_col = col
            break

    print(f"\n🏆 姿态 mAP50 最高值: {best_value:.4f}")
    print(f"📍 对应轮次: Epoch {int(best_epoch)}")

    if map95_col:
        best95_value = df[map95_col].iloc[best_idx]
        print(f"📊 对应 mAP50-95: {best95_value:.4f}")

    # 同时显示检测的 mAP50 作为参考
    map50_b_col = None
    for col in df.columns:
        if 'mAP50(B)' in col:
            map50_b_col = col
            break

    if map50_b_col:
        best_b_value = df[map50_b_col].iloc[best_idx]
        print(f"📦 对应检测 mAP50: {best_b_value:.4f}")

    print("\n" + "-" * 60)

    # 额外显示前3名
    print("\n🥇 姿态 mAP50 前3名:")
    top3 = df.nlargest(3, map50_col)
    for i, (idx, row) in enumerate(top3.iterrows(), 1):
        epoch = int(row['epoch']) if 'epoch' in df.columns else idx
        map50 = row[map50_col]
        map95 = row[map95_col] if map95_col else None
        if map95 is not None:
            print(f"  #{i}: Epoch {epoch:3d} → mAP50={map50:.4f}, mAP50-95={map95:.4f}")
        else:
            print(f"  #{i}: Epoch {epoch:3d} → mAP50={map50:.4f}")

    return {'best_epoch': int(best_epoch), 'best_map50': best_value}


def find_latest_and_get_best(search_dir=None):
    """
    自动找到最新的训练结果并读取最佳值
    """
    if search_dir is None:
        search_dir = os.path.join(os.path.dirname(__file__), 'runs', 'pose')

    # 查找所有 results.csv
    csv_files = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file == 'results.csv':
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print(f"❌ 未找到 results.csv，请检查路径: {search_dir}")
        return None

    # 按修改时间排序，取最新的
    latest = max(csv_files, key=os.path.getmtime)
    print(f"✅ 找到最新训练: {os.path.basename(os.path.dirname(latest))}")
    print(f"📂 路径: {latest}\n")

    return get_best_pose_map50(latest)


if __name__ == '__main__':
    # ====== 使用方法 ======

    # 方法1：直接指定文件路径
    csv_path = r"D:\Project\ultralytics-main-pose\fish_detect_competition\runs\pose\train3\results.csv"

    if os.path.exists(csv_path):
        get_best_pose_map50(csv_path)
    else:
        print(f"⚠️ 文件不存在，尝试自动查找最新训练...\n")
        find_latest_and_get_best()

    # 方法2：自动查找最新训练（取消注释使用）
    # find_latest_and_get_best()