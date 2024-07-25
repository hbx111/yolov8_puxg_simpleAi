# --*- coding: utf-8 -*--
import argparse
import threading
import dxcam
import cv2
import torch
import win32api
import win32con
from ultralytics import YOLO
import ctypes
import struct
import string
import time
import pyautogui
import logging
from pynput.mouse import Controller, Button
from time import sleep
import pydirectinput
import kmNet as km

km.init('192.168.2.188', '9090', '6E79E04E')
km.monitor(9090)
right_button_down = False
speedd = 50


# 新增函数，用于监听鼠标右键状态
def check_right_button():
    global right_button_down
    while True:
        if km.isdown_right() == 1:
            right_button_down = True
        elif km.isdown_right() == 0:
            right_button_down = False
        time.sleep(0.05)  # 减少CPU负担


# 开启新线程监听鼠标右键状态
threading.Thread(target=check_right_button).start()

# 定义参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", "-m", type=str, default="v8.pt", help="模型路径")
parser.add_argument("--image-size", type=int, default=640, help="模型输入尺寸")
parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
parser.add_argument("--iou-thres", type=float, default=0.1, help="NMS的IOU阈值")
parser.add_argument("--device", type=str, default="", help="运行设备，空则自动选择，选项：cpu, 0, 1, ...")
parser.add_argument("--half", action='store_true', help="使用半精度")
args = parser.parse_args()


# 处理YOLOv8操作的类
class Yolov8:
    def __init__(self, device, half, size):
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = half
        self.size = size

    def LoadModel(self, weights):
        # 使用YOLOv8的API加载模型
        self.model = YOLO(weights)
        if self.device != "cpu" and torch.cuda.is_available():
            self.model.to(self.device)
        if self.half:
            self.model.half()

    def Inference(self, img, conf, iou):
        # 使用YOLOv8的predict方法
        results = self.model(img, conf=conf, iou=iou, max_det=20)
        return results


def DarwBox(img, results):
    global right_button_down
    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()

            # 计算目标中心
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2

            # 绘制边界框
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), thickness=2)

            # 添加标签
            label = f'{r.names[int(cls)]} {conf:.2f}'
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("目标中心：", int(center_x - 320), int(center_y - 307))

            # 只有在鼠标右键持续按下的情况下，才移动鼠标到目标中心
            if right_button_down:
                km.move_auto(int(center_x - 320), int(center_y - 304), 400)

    cv2.imshow("shor_win", img)
    cv2.waitKey(1)


class Capture:
    """
    dx 截图功能包装器
    """

    def __init__(self, size):
        self.dx = dxcam.create()
        self.size = size
        self.w = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        self.h = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        self.x0 = (self.w / 2) - (self.size / 2)
        self.y0 = (self.h / 2) - (self.size / 2)
        self.x1 = (self.w / 2) + (self.size / 2)
        self.y1 = (self.h / 2) + (self.size / 2)
        self.region = (int(self.x0), int(self.y0), int(self.x1), int(self.y1))

    def grab(self):
        img = self.dx.grab(self.region)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            return img
        return False


def run():
    try:
        # 初始化捕获
        cap = Capture(args.image_size)
        # 初始化YOLOv8实例
        yolo = Yolov8(device=args.device, half=args.half, size=args.image_size)
        # 加载模型
        yolo.LoadModel(args.model_path)

        while True:
            # 捕获图像
            img = cap.grab()
            if img is False:
                continue
            # 执行推理
            results = yolo.Inference(img, conf=args.conf_thres, iou=args.iou_thres)
            # 绘制边界框
            DarwBox(img, results)
    except KeyboardInterrupt:
        print("程序被用户中断，正在退出...")
        exit(0)


if __name__ == '__main__':
    run()
