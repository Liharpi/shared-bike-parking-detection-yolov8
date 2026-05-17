import cv2
from ultralytics import YOLO


def main():
    # 1. 加载官方的 YOLOv8 Nano 预训练模型 (初次运行会自动下载一个十几MB的权重文件)
    print("正在加载 YOLO 模型...")
    model = YOLO("yolov8n.pt")

    # 2. 调用电脑默认摄像头 (参数 0 代表自带摄像头，如果有外接摄像头可以改为 1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接！")
        return

    print("摄像头已开启，按 'q' 键退出。")

    # 3. 实时读取画面并进行目标检测
    while True:
        success, frame = cap.read()
        if not success:
            break

        # 使用 YOLO 模型对当前帧进行推理
        # stream=True 适合视频流，加快处理速度
        results = model.predict(frame, stream=True, verbose=False)

        # 遍历检测结果，并将画好检测框的画面提取出来
        for r in results:
            annotated_frame = r.plot()

        # 显示带有检测框的实时画面
        cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

        # 等待按键输入，如果按下 'q' 则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()