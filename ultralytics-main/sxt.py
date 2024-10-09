import sys
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)

        while self._run_flag:
            ret, cv_img = cap.read()

            if ret:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img)

        cap.release()

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

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        start_button = QPushButton('Start', self)
        start_button.clicked.connect(self.start_video)

        stop_button = QPushButton('Stop', self)
        stop_button.clicked.connect(self.stop_video)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(start_button)
        layout.addWidget(stop_button)

        self.setLayout(layout)

        self.setGeometry(100, 100, 800, 400)
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
    main_window = MainWindow()
    sys.exit(app.exec_())
