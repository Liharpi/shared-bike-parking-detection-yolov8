import os

from ultralytics import YOLO

# 必须加上这一行作为“主程序入口锁”
if __name__ == '__main__':
    # 下面这三行必须要缩进（前面有4个空格）
    model_path = os.environ.get("MODEL_PATH", "weights/last.pt")

    # 加载你的进度包
    model = YOLO(model_path)

    # 启动复飞！
    model.train(resume=True)
