import cv2
import os

# ===== 直接在这里修改 =====
video_path = (r"C:\Users\ying'han'zhen\Desktop\video\珍珠龙胆石斑.mp4")   # 视频路径
time_list = [214,280]         # 多个时间点(秒)
output_dir = "./frames3"                 # 保存目录
quality = 90
# =========================

os.makedirs(output_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
name = os.path.splitext(os.path.basename(video_path))[0]

for t in time_list:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if ret:
        save_path = os.path.join(output_dir, f"{name}_{t}s.jpg")
        cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"✅ {t}s")
cap.release()