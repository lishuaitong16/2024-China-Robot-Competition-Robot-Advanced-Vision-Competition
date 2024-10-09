import sys
import cv2 as cv
from ultralytics import YOLO
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
import numpy as np


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        try:
            # 尝试使用相机索引 0
            orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)

            # 检查是否成功打开相机
            if not orbbec_cap.isOpened():
                raise Exception('无法打开相机')

            # 加载yolo模型
            model = YOLO('yolov8n.pt')

            while self._run_flag:
                if orbbec_cap.grab():
                    # 获取彩色图像数据
                    ret_bgr, bgr_image = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)

                    # 只在成功获取图像时才进行处理
                    if ret_bgr:
                        res = model(bgr_image)
                        results = res[0].plot()
                        rgb_image = cv.cvtColor(results, cv.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        self.change_pixmap_signal.emit(qt_img)

                # 限制循环速率
                self.msleep(30)

        except Exception as e:
            print(f'发生异常: {e}')

        finally:
            # 释放相机资源
            if orbbec_cap.isOpened():
                orbbec_cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignCenter)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        left_layout.addWidget(self.video_label)
        main_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()

        start_button = QPushButton('开始')
        start_button.setObjectName('startButton')
        start_button.clicked.connect(self.start_video)

        stop_button = QPushButton('结束')
        stop_button.setObjectName('stopButton')
        stop_button.clicked.connect(self.stop_video)

        right_layout.addWidget(start_button)
        right_layout.addWidget(stop_button)

        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.setGeometry(100, 100, 1000, 600)
        self.setWindowTitle('OpenCV with PyQt5')
        self.show()

    def start_video(self):
        self.video_thread.start()

    def stop_video(self):
        self.video_thread.stop()

    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget {
            background-color: #f0f0f0;
        }
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 10px;
        }
        QPushButton#startButton:hover, QPushButton#stopButton:hover {
            background-color: #45a049;
        }
    """)

    main_window = MainWindow()
    sys.exit(app.exec_())
