import pandas as pd

# 读取训练结果
df = pd.read_csv(r"D:\Project\fish_planB\runs\segment\train3\results.csv")
df.columns = df.columns.str.strip()   # 去掉列名空格

print(f"\n【Box Loss - 边界框损失】")
print(f"  最小值: {df['train/box_loss'].min():.4f} (Epoch {df['train/box_loss'].idxmin()})")
print(f"  最终值: {df['train/box_loss'].iloc[-1]:.4f}")

print(f"\n【Class Loss - 分类损失】")
print(f"  最小值: {df['train/cls_loss'].min():.4f} (Epoch {df['train/cls_loss'].idxmin()})")
print(f"  最终值: {df['train/cls_loss'].iloc[-1]:.4f}")

if 'train/dfl_loss' in df.columns:
    print(f"\n【DFL Loss - 分布焦点损失】")
    print(f"  最小值: {df['train/dfl_loss'].min():.4f} (Epoch {df['train/dfl_loss'].idxmin()})")
    print(f"  最终值: {df['train/dfl_loss'].iloc[-1]:.4f}")

print(f"\n【mAP50 - 平均精度(IoU=0.5)】")
print(f"  最大值: {df['metrics/mAP50(B)'].max():.4f} (Epoch {df['metrics/mAP50(B)'].idxmax()})")
print(f"  最终值: {df['metrics/mAP50(B)'].iloc[-1]:.4f}")

if 'metrics/mAP50-95(B)' in df.columns:
    print(f"\n【mAP50-95 - 平均精度(IoU=0.5-0.95)】")
    print(f"  最大值: {df['metrics/mAP50-95(B)'].max():.4f} (Epoch {df['metrics/mAP50-95(B)'].idxmax()})")
    print(f"  最终值: {df['metrics/mAP50-95(B)'].iloc[-1]:.4f}")

if 'metrics/precision(B)' in df.columns:
    print(f"\n【Precision - 精确率】")
    print(f"  最大值: {df['metrics/precision(B)'].max():.4f} (Epoch {df['metrics/precision(B)'].idxmax()})")
    print(f"  最终值: {df['metrics/precision(B)'].iloc[-1]:.4f}")

if 'metrics/recall(B)' in df.columns:
    print(f"\n【Recall - 召回率】")
    print(f"  最大值: {df['metrics/recall(B)'].max():.4f} (Epoch {df['metrics/recall(B)'].idxmax()})")
    print(f"  最终值: {df['metrics/recall(B)'].iloc[-1]:.4f}")

if 'val/box_loss' in df.columns:
    print(f"\n【验证集 Box Loss】")
    print(f"  最小值: {df['val/box_loss'].min():.4f} (Epoch {df['val/box_loss'].idxmin()})")
    print(f"  最终值: {df['val/box_loss'].iloc[-1]:.4f}")

if 'val/cls_loss' in df.columns:
    print(f"\n【验证集 Class Loss】")
    print(f"  最小值: {df['val/cls_loss'].min():.4f} (Epoch {df['val/cls_loss'].idxmin()})")
    print(f"  最终值: {df['val/cls_loss'].iloc[-1]:.4f}")



print("\n" + "=" * 60)
print("Mask 指标")

# 1. 训练集 seg_loss
if 'train/seg_loss' in df.columns:
    print(f"\n【Seg Loss - 分割损失】")
    print(f"  最小值: {df['train/seg_loss'].min():.4f} (Epoch {df['train/seg_loss'].idxmin()})")
    print(f"  最终值: {df['train/seg_loss'].iloc[-1]:.4f}")
else:
    print("\n[WARN] 没有 train/seg_loss 列")

# 2. 验证集 seg_loss
if 'val/seg_loss' in df.columns:
    print(f"\n【验证集 Seg Loss】")
    print(f"  最小值: {df['val/seg_loss'].min():.4f} (Epoch {df['val/seg_loss'].idxmin()})")
    print(f"  最终值: {df['val/seg_loss'].iloc[-1]:.4f}")
else:
    print("\n[WARN] 没有 val/seg_loss 列")

# 3. Mask Precision
if 'metrics/precision(M)' in df.columns:
    print(f"\n【Mask Precision - 掩码精确率】")
    print(f"  最大值: {df['metrics/precision(M)'].max():.4f} (Epoch {df['metrics/precision(M)'].idxmax()})")
    print(f"  最终值: {df['metrics/precision(M)'].iloc[-1]:.4f}")
else:
    print("\n[WARN] 没有 metrics/precision(M) 列")

# 4. Mask Recall
if 'metrics/recall(M)' in df.columns:
    print(f"\n【Mask Recall - 掩码召回率】")
    print(f"  最大值: {df['metrics/recall(M)'].max():.4f} (Epoch {df['metrics/recall(M)'].idxmax()})")
    print(f"  最终值: {df['metrics/recall(M)'].iloc[-1]:.4f}")
else:
    print("\n[WARN] 没有 metrics/recall(M) 列")

# 5. Mask mAP50
if 'metrics/mAP50(M)' in df.columns:
    print(f"\n【Mask mAP50 - 掩码 mAP(IoU=0.5)】")
    print(f"  最大值: {df['metrics/mAP50(M)'].max():.4f} (Epoch {df['metrics/mAP50(M)'].idxmax()})")
    print(f"  最终值: {df['metrics/mAP50(M)'].iloc[-1]:.4f}")
else:
    print("\n[WARN] 没有 metrics/mAP50(M) 列")

# 6. Mask mAP50-95
if 'metrics/mAP50-95(M)' in df.columns:
    print(f"\n【Mask mAP50-95 - 掩码 mAP(IoU=0.5-0.95)】")
    print(f"  最大值: {df['metrics/mAP50-95(M)'].max():.4f} (Epoch {df['metrics/mAP50-95(M)'].idxmax()})")
    print(f"  最终值: {df['metrics/mAP50-95(M)'].iloc[-1]:.4f}")
else:
    print("\n[WARN] 没有 metrics/mAP50-95(M) 列")

# 7. 同时打印对应的 epoch
print("\n" + "=" * 60)
print("最佳 Epoch 汇总")

best_epochs = {}

if 'metrics/mAP50(B)' in df.columns:
    best_epochs['Box mAP50'] = int(df['metrics/mAP50(B)'].idxmax()) + 1
if 'metrics/mAP50(M)' in df.columns:
    best_epochs['Mask mAP50'] = int(df['metrics/mAP50(M)'].idxmax()) + 1
if 'metrics/mAP50-95(B)' in df.columns:
    best_epochs['Box mAP50-95'] = int(df['metrics/mAP50-95(B)'].idxmax()) + 1
if 'metrics/mAP50-95(M)' in df.columns:
    best_epochs['Mask mAP50-95'] = int(df['metrics/mAP50-95(M)'].idxmax()) + 1

for k, v in best_epochs.items():
    print(f"  {k}: Epoch {v}")
