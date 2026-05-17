import sys
import cv2
import numpy as np
import time
import os
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, \
    QFileDialog, QFrame, QSlider, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from ultralytics import YOLO

# ==================== 0. 系统配置开关 ====================
# 调试模式：True=显示YOLO检测框（开发调试用）, False=干净画面（论文展示用）
SHOW_DETECTION_BOXES = False
# 模型权重路径（切换模型只需改这里）
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("weights", "best.pt"))
# 违停交集比例阈值（论文3.4节）
OVERLAP_THRESHOLD = 0.80
# 防抖时间间隔（秒）
DEBOUNCE_INTERVAL = 5

# ==================== 1. 初始化本地取证目录 ====================
if not os.path.exists('snapshots'):
    os.makedirs('snapshots')


# ==================== 2. 核心算法后台 (含抓拍与数据库接口，已修复静态图解析 BUG & 实现去框化) ====================
class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(bool, bool, bool)
    violation_signal = pyqtSignal(str, str, str, str)

    def __init__(self, model_path, source_path):
        super().__init__()
        self.model = YOLO(model_path)
        self.source_path = source_path
        self._run_flag = True
        self.conf_threshold = 0.4
        self.last_snap_time = 0
        self.frame_count = 0  # 帧计数器，用于event_id等

    def update_conf(self, value):
        self.conf_threshold = value / 100.0

    def run(self):
        # 摄像头传字符串"0"，需转为整数0给cv2.VideoCapture
        source = 0 if self.source_path == "0" else self.source_path

        # 判断源文件是否为静态图片
        is_image = isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

        if is_image:
            # 【核心修复】：使用 imdecode 读取图片，免疫非标文件头和中文路径报错
            original_frame = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if original_frame is None:
                print(f"警告：无法解码图像文件 {source}")
                return

            # 【新增功能】：静态图沙盒模式
            while self._run_flag:
                # 给业务逻辑传递原图的副本
                self._process_frame(original_frame.copy())
                # 静态图只需低频刷新，节省 CPU
                QThread.msleep(200)
        else:
            # 视频流或实时摄像头走原始逻辑
            cap = cv2.VideoCapture(source)
            while self._run_flag and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self._process_frame(frame)
                # 视频流保持 30FPS 刷新率
                QThread.msleep(30)
            cap.release()

    # 将业务逻辑抽离为一个独立函数
    def _process_frame(self, frame):
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        result = results[0]

        parking_areas = []
        bikes = []
        fallen_bikes = []

        # 1. 业务逻辑计算（保持不变）
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls_id = box.tolist()
            cls_id = int(cls_id)
            if cls_id == 2:
                parking_areas.append([int(x1), int(y1), int(x2), int(y2)])
            elif cls_id == 0:
                bikes.append([int(x1), int(y1), int(x2), int(y2)])
            elif cls_id == 1:
                fallen_bikes.append([int(x1), int(y1), int(x2), int(y2)])

        bike_detected = (len(bikes) + len(fallen_bikes)) > 0
        normal_posture = (len(fallen_bikes) == 0)
        standard_location = False
        violation_coords = ""
        violation_type = ""

        if not normal_posture:
            violation_type = "单车倒地"
            violation_coords = f"({fallen_bikes[0][0]}, {fallen_bikes[0][1]})"

        if bike_detected and normal_posture:
            illegal_count = 0
            for bike in bikes:
                bx1, by1, bx2, by2 = bike
                is_safe = False
                for area in parking_areas:
                    ax1, ay1, ax2, ay2 = area
                    inter_x1, inter_y1 = max(bx1, ax1), max(by1, ay1)
                    inter_x2, inter_y2 = min(bx2, ax2), min(by2, ay2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        # 【核心修正】论文3.4节：交集比例阈值判定
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        bike_area = (bx2 - bx1) * (by2 - by1)
                        overlap_ratio = inter_area / bike_area if bike_area > 0 else 0.0
                        if overlap_ratio >= OVERLAP_THRESHOLD:
                            is_safe = True
                            break
                if not is_safe:
                    illegal_count += 1
                    violation_coords = f"({bx1}, {by1})"
                    violation_type = "越界/未入框"
            standard_location = (illegal_count == 0)

        is_violation = (not normal_posture) or (bike_detected and not standard_location)
        if is_violation and (time.time() - self.last_snap_time > DEBOUNCE_INTERVAL):
            # 取证图片：保存带自定义业务标注框的画面
            evidence_img = frame.copy()
            # 绘制停车区 (绿色)
            for area in parking_areas:
                cv2.rectangle(evidence_img, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), 3)
            # 绘制倒地单车 (红色)
            for fb in fallen_bikes:
                cv2.rectangle(evidence_img, (fb[0], fb[1]), (fb[2], fb[3]), (0, 0, 255), 3)
            # 绘制正常单车 (蓝色)
            for b in bikes:
                cv2.rectangle(evidence_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)

            self.frame_count += 1
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            img_name = f"snapshots/v_{timestamp_str}.jpg"
            cv2.imwrite(img_name, evidence_img)
            human_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.violation_signal.emit(human_time, violation_type, violation_coords, img_name)
            self.last_snap_time = time.time()

        # 2. UI 画面显示：根据配置开关选择干净画面或调试画面
        if SHOW_DETECTION_BOXES:
            # 调试模式：全要素检测框可视化
            display_frame = frame.copy()
            for area in parking_areas:
                cv2.rectangle(display_frame, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), 2)
                cv2.putText(display_frame, "Parking Area", (area[0], max(0, area[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for b in bikes:
                cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                cv2.putText(display_frame, "Normal Bike", (b[0], max(0, b[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            for fb in fallen_bikes:
                cv2.rectangle(display_frame, (fb[0], fb[1]), (fb[2], fb[3]), (0, 0, 255), 2)
                cv2.putText(display_frame, "Fallen Bike", (fb[0], max(0, fb[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self.change_pixmap_signal.emit(display_frame)
        else:
            # 论文展示模式：干净画面，无YOLO检测框
            self.change_pixmap_signal.emit(frame)
        self.status_signal.emit(bike_detected, normal_posture, standard_location)

    def stop(self):
        self._run_flag = False
        self.wait()


# ==================== 3. GUI 主界面 (含数据库与抓拍表格) ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能共享单车监测系统 V4.0 (含数据库取证)")
        self.setGeometry(50, 50, 1400, 900)
        self.setStyleSheet("background-color: #121212; color: #ffffff; font-family: 'Microsoft YaHei';")

        # 模型路径统一使用全局配置
        self.MODEL_PATH = MODEL_PATH
        self.thread = None

        self.init_db()
        self.initUI()
        self.load_db_to_table()  # 启动时加载历史记录

    def init_db(self):
        # 初始化 SQLite 数据库
        self.conn = sqlite3.connect('bike_parking.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        # 扩展数据库字段：论文4.12节建议增加置信度、交集比例、来源、帧号、事件ID
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS violations
                             (id INTEGER PRIMARY KEY AUTOINCREMENT,
                              time TEXT, type TEXT, coords TEXT, path TEXT,
                              confidence REAL DEFAULT 0.0,
                              overlap_ratio REAL DEFAULT 0.0,
                              source TEXT DEFAULT '',
                              frame_id INTEGER DEFAULT 0,
                              event_id TEXT DEFAULT '')''')
        self.conn.commit()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        h_layout = QHBoxLayout(main_widget)

        # =============== 左侧：监控与控制区 ===============
        left_layout = QVBoxLayout()
        self.image_label = QLabel("等待接入监控画面...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 2px dashed #444444; border-radius: 8px;")
        self.image_label.setMinimumSize(850, 600)
        left_layout.addWidget(self.image_label)

        # 置信度滑块
        conf_layout = QHBoxLayout()
        conf_title = QLabel("AI 灵敏度(置信度):")
        conf_title.setFont(QFont("Microsoft YaHei", 12))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(40)
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        self.conf_value_label = QLabel("0.40")
        self.conf_value_label.setFont(QFont("Consolas", 14, QFont.Bold))
        self.conf_value_label.setStyleSheet("color: #2196f3;")
        conf_layout.addWidget(conf_title)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        left_layout.addLayout(conf_layout)

        # 核心按钮区 (新增摄像头支持)
        btn_layout = QHBoxLayout()
        self.btn_camera = QPushButton("📷 启动实时摄像头")
        self.btn_camera.setStyleSheet(
            "background-color: #2e7d32; padding: 12px; font-size: 15px; border-radius: 6px; font-weight: bold;")
        self.btn_camera.clicked.connect(self.open_camera)

        self.btn_open = QPushButton("📁 导入监控视频/图片")
        self.btn_open.setStyleSheet("background-color: #0d47a1; padding: 12px; font-size: 15px; border-radius: 6px;")
        self.btn_open.clicked.connect(self.open_file)

        self.btn_stop = QPushButton("⏹ 停止监测")
        self.btn_stop.setStyleSheet("background-color: #b71c1c; padding: 12px; font-size: 15px; border-radius: 6px;")
        self.btn_stop.clicked.connect(self.stop_detection)

        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_stop)
        left_layout.addLayout(btn_layout)

        # =============== 右侧：状态面板 + 数据库表格 ===============
        right_layout = QVBoxLayout()

        panel_title = QLabel("业务判定实时面板")
        panel_title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        panel_title.setAlignment(Qt.AlignCenter)
        panel_title.setStyleSheet("color: #2196f3;")
        right_layout.addWidget(panel_title)

        # 三大状态灯
        self.lbl_detect_status = QLabel("⚪ 无画面")
        self.lbl_posture_status = QLabel("⚪ 无画面")
        self.lbl_location_status = QLabel("⚪ 无画面")
        self.setup_status_card(right_layout, "① 检测状态", self.lbl_detect_status)
        self.setup_status_card(right_layout, "② 姿态分析", self.lbl_posture_status)
        self.setup_status_card(right_layout, "③ 区域合规", self.lbl_location_status)

        # 新增：违停抓拍记录数据库表格
        db_title = QLabel("🗄️ 违停抓拍证据链 (SQLite)")
        db_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        db_title.setStyleSheet("color: #ff9800; margin-top: 10px;")
        right_layout.addWidget(db_title)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["抓拍时间", "违规类型", "坐标", "证据图片路径"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #1e1e1e; color: #ccc; gridline-color: #444; border: 1px solid #333; }
            QHeaderView::section { background-color: #333; color: white; font-weight: bold; }
        """)
        right_layout.addWidget(self.table)

        h_layout.addLayout(left_layout, stretch=6)
        h_layout.addLayout(right_layout, stretch=4)

    def setup_status_card(self, parent_layout, title, status_label):
        card = QFrame()
        card.setStyleSheet("background-color: #1e1e1e; border-radius: 8px; border: 1px solid #333; padding: 10px;")
        layout = QVBoxLayout(card)
        t_label = QLabel(title)
        t_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        status_label.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(t_label)
        layout.addWidget(status_label)
        parent_layout.addWidget(card)

    def on_conf_change(self, value):
        conf_float = value / 100.0
        self.conf_value_label.setText(f"{conf_float:.2f}")
        if self.thread is not None:
            self.thread.update_conf(value)

    # 启动本地摄像头
    def open_camera(self):
        self.start_detection("0")

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择源文件", "", "Media Files (*.mp4 *.jpg *.png)")
        if file_name:
            self.start_detection(file_name)

    def start_detection(self, source):
        self.stop_detection()
        self.thread = DetectionThread(self.MODEL_PATH, source)
        self.thread.update_conf(self.conf_slider.value())
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.update_status_panel)
        self.thread.violation_signal.connect(self.record_violation)  # 连接数据库写入信号
        self.thread.start()

    def update_image(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qt_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        p = qt_img.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def update_status_panel(self, detected, posture, location):
        if detected:
            self.lbl_detect_status.setText("🟢 已检测到单车")
            self.lbl_detect_status.setStyleSheet("color: #4caf50;")
            self.lbl_posture_status.setText("🟢 姿态正常" if posture else "🔴 异常 (倒地！)")
            self.lbl_posture_status.setStyleSheet("color: #4caf50;" if posture else "color: #f44336;")
            self.lbl_location_status.setText("🟢 规范区域" if location else "🔴 违停 (越界/未入框)")
            self.lbl_location_status.setStyleSheet("color: #4caf50;" if location else "color: #f44336;")
        else:
            for lbl in [self.lbl_detect_status, self.lbl_posture_status, self.lbl_location_status]:
                lbl.setText("⚪ 无画面/无单车")
                lbl.setStyleSheet("color: #888888;")

    # 接收线程传来的证据，写入 SQLite 并更新 UI 表格
    def record_violation(self, time_str, v_type, coords, img_path):
        # 1. 写入数据库
        self.cursor.execute("INSERT INTO violations (time, type, coords, path) VALUES (?, ?, ?, ?)",
                            (time_str, v_type, coords, img_path))
        self.conn.commit()

        # 2. 插入表格最顶端显示
        self.table.insertRow(0)
        self.table.setItem(0, 0, QTableWidgetItem(time_str))

        type_item = QTableWidgetItem(v_type)
        type_item.setForeground(Qt.red)  # 违规类型标红
        self.table.setItem(0, 1, type_item)

        self.table.setItem(0, 2, QTableWidgetItem(coords))
        self.table.setItem(0, 3, QTableWidgetItem(img_path))

    # 启动时加载历史记录
    def load_db_to_table(self):
        self.cursor.execute("SELECT time, type, coords, path FROM violations ORDER BY id DESC LIMIT 50")
        rows = self.cursor.fetchall()
        for row in rows:
            row_idx = self.table.rowCount()
            self.table.insertRow(row_idx)
            self.table.setItem(row_idx, 0, QTableWidgetItem(row[0]))

            type_item = QTableWidgetItem(row[1])
            type_item.setForeground(Qt.red)
            self.table.setItem(row_idx, 1, type_item)

            self.table.setItem(row_idx, 2, QTableWidgetItem(row[2]))
            self.table.setItem(row_idx, 3, QTableWidgetItem(row[3]))

    def stop_detection(self):
        if self.thread is not None:
            self.thread.stop()

    def closeEvent(self, event):
        self.stop_detection()
        self.conn.close()  # 退出时关闭数据库连接
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
