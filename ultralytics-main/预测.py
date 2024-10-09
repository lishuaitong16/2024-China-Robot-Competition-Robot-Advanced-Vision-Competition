import cv2 as cv
from ultralytics import YOLO
import sys
import numpy as np

try:
    # 尝试使用相机索引 0
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)

    # 检查是否成功打开相机
    if not orbbec_cap.isOpened():
        raise Exception('无法打开相机')

    # 加载yolo模型
    model = YOLO('yolov8n.pt')

    while True:
        if orbbec_cap.grab():
            # 获取彩色图像数据
            ret_bgr, bgr_image = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)

            # 只在成功获取图像时才进行处理
            if ret_bgr:
                res = model(bgr_image)
                results = res[0].plot()
                cv.imshow('YOLOV8检测图像', results)

        # 等待用户按键，检查是否按下了关闭窗口的按键（27是Esc键的ASCII码）
        key = cv.waitKey(1)
        if key == 27:
            break

except Exception as e:
    print(f'发生异常: {e}')

finally:
    # 释放相机资源
    if orbbec_cap.isOpened():
        orbbec_cap.release()
    cv.destroyAllWindows()
