import cv2
import os
from ultralytics import YOLO


def batch_smart_extract(input_dir, output_dir, extract_interval=30):
    """
    V2.0 批量智能抽帧脚本：自动遍历文件夹内所有视频
    """
    # 1. 检查输入文件夹是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 找不到视频文件夹：{input_dir}，请检查路径！")
        return

    # 2. 找出文件夹里所有的视频文件 (支持 mp4, avi, mov 等格式)
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if len(video_files) == 0:
        print(f"❌ 在 {input_dir} 中没有找到任何视频文件！")
        return

    # 3. 加载 AI 模型并创建输出文件夹
    print("正在加载 AI 过滤模型...")
    model = YOLO('yolov8n.pt')
    os.makedirs(output_dir, exist_ok=True)

    global_saved_count = 0  # 记录总共存了多少张图

    # 4. 开始极其舒适的“自动化流水线”批量处理
    print(f"\n🚀 启动流水线！共发现 {len(video_files)} 个视频文件，准备开搞...")

    for video_name in video_files:
        video_path = os.path.join(input_dir, video_name)
        print(f"\n🎬 正在处理视频: [ {video_name} ] ...")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 当前视频播完了，准备切下一个

            # 每隔 extract_interval 帧检查一次
            if frame_count % extract_interval == 0:
                results = model(frame, classes=[1], verbose=False)

                # 如果发现单车
                if len(results[0].boxes) > 0:
                    global_saved_count += 1
                    # 极其细节：给图片起个好名字（加上原视频名，防止不同视频的图片名字冲突）
                    clean_video_name = os.path.splitext(video_name)[0]
                    save_name = f"bike_{clean_video_name}_{global_saved_count}.jpg"
                    save_path = os.path.join(output_dir, save_name)

                    cv2.imwrite(save_path, frame)
                    print(f"  ✅ 提取成功 -> {save_name} (累计: {global_saved_count} 张)")

            frame_count += 1

        cap.release()

    print(f"\n🎉 批量抽帧彻底完工！")
    print(f"📈 战报：处理了 {len(video_files)} 个视频，成功提炼出 {global_saved_count} 张绝佳图片！")
    print(f"📁 宝藏已存入: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    # ================= 极简配置区 =================
    # 1. 在你的代码同级目录下，新建一个叫 "videos_in" 的文件夹，把所有视频全扔进去！
    INPUT_FOLDER = "videos_in"

    # 2. 跑出来的图片会自动全部装进这个 "images_out" 文件夹里
    OUTPUT_FOLDER = "images_out"
    # ==========================================

    batch_smart_extract(INPUT_FOLDER, OUTPUT_FOLDER)