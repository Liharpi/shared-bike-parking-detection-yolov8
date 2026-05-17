import cv2
import os


def extract_frames_from_folder(video_folder, output_folder, frame_interval=30):
    """批量从文件夹中的所有视频抽取帧"""

    # 1. 自动创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. 获取文件夹下所有视频文件 (支持 mp4, avi, mov 等)
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    # 找出现有文件夹里所有的文件，并筛选出视频格式
    try:
        video_files = [f for f in os.listdir(video_folder) if os.path.splitext(f)[1].lower() in valid_extensions]
    except FileNotFoundError:
        print(f"❌ 找不到文件夹：{video_folder}，请检查路径！")
        return

    if not video_files:
        print(f"❌ 在 {video_folder} 中没有找到任何视频文件！")
        return

    print(f"🔍 成功找到 {len(video_files)} 个视频文件，准备开启批量流水线...")
    total_saved_count = 0

    # 3. 开始循环处理每一个视频
    for video_name in video_files:
        video_path = os.path.join(video_folder, video_name)
        # 获取不带后缀的视频名，例如 "视频1"
        video_base_name = os.path.splitext(video_name)[0]

        print(f"\n▶️ 正在处理: {video_name} ...")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"⚠️ 警告: 无法打开 {video_name}，可能已损坏或加密，自动跳过。")
            continue

        current_frame = 0
        video_saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % frame_interval == 0:
                # 【核心修改1】在图片名加上视频名作为前缀，绝对不会覆盖！
                file_name = f"{video_base_name}_frame_{current_frame:04d}.jpg"
                save_path = os.path.join(output_folder, file_name)

                # 【核心修改2】使用原生文件流写入，完美支持带中文的路径！
                cv2.imencode('.jpg', frame)[1].tofile(save_path)

                video_saved_count += 1
                total_saved_count += 1

            current_frame += 1

        cap.release()
        print(f"✅ {video_name} 榨干完毕！单管提取了 {video_saved_count} 张。")

    print(f"\n🎉 批量任务圆满结束！共提取 {total_saved_count} 张极品图片。")
    print(f"📁 它们现在安全地躺在: {output_folder}")


# ==========================================
# 👇 只需要修改下面这两个文件夹路径 👇
# ==========================================
if __name__ == "__main__":
    # 【改这里 1】把你那3个视频，都放进这个叫 videos 的文件夹里
    INPUT_FOLDER = os.environ.get("VIDEO_INPUT_DIR", "videos_in")

    # 【改这里 2】抽出来的所有图片，都会存在这里（不用改，就用你刚才的路径）
    OUTPUT_DIR = os.environ.get("FRAME_OUTPUT_DIR", "images_out")

    # 【改这里 3】因为你的视频可能比较短，建议把间隔调小，比如设为 10 或 15
    INTERVAL = 10

    extract_frames_from_folder(INPUT_FOLDER, OUTPUT_DIR, frame_interval=INTERVAL)
