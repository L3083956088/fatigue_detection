import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
import pygame
from datetime import datetime
import matplotlib.pyplot as plt

# 初始化pygame
pygame.mixer.init()

# 加载预训练的眼睛状态识别模型
model = tf.saved_model.load("path/to/eye_state_model")

# 创建主窗口
root = tk.Tk()
root.title("Eye and Head State Detection")

# 创建标签用于显示眼睛状态
eye_label = ttk.Label(root, text="Eye State: ", font=("Helvetica", 14))
eye_label.pack()

# 创建标签用于显示头部状态
head_label = ttk.Label(root, text="Head State: ", font=("Helvetica", 14))
head_label.pack()

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置警报阈值
eye_alarm_threshold = 0.5
head_alarm_threshold = 10  # 调整这个值以适应你的需求

# 创建历史数据列表
eye_history = []
head_history = []

# 创建图表
fig, ax = plt.subplots()
x_data = []
eye_y_data = []
head_y_data = []
eye_line, = ax.plot(x_data, eye_y_data, label='Eye State')
head_line, = ax.plot(x_data, head_y_data, label='Head Angle')
ax.set_title('Eye and Head State Over Time')
ax.set_xlabel('Time')
ax.set_ylabel('Eye State (Closed=1, Open=0)')
ax2 = ax.twinx()
ax2.set_ylabel('Head Angle (degrees)')


# 创建保存截图的目录
screenshot_dir = "screenshots"
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# 启动和停止标志
is_running = False

# 加载人脸检测器和姿态估计器（这里使用一个简单的深度学习模型）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
pose_model = cv2.dnn.readNet("path/to/pose_model.prototxt", "path/to/pose_model.caffemodel")

def estimate_head_pose(face_roi):
    h, w, c = face_roi.shape
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (224, 224), (104, 117, 123))
    pose_model.setInput(blob)
    pose_output = pose_model.forward()

    # 解析姿态估计输出
    yaw = pose_output[0][0]
    pitch = pose_output[0][1]
    roll = pose_output[0][2]

    return yaw, pitch, roll
def update_eye_and_head_state():
    global is_running

    if is_running:
        ret, frame = cap.read()
        if not ret:
            return

        # 调整图像大小
        resized_frame = cv2.resize(frame, (224, 224))
        input_image = np.expand_dims(resized_frame, axis=0)

        # 使用模型预测眼睛状态
        predictions = model(input_image)

        # 解析预测结果
        eye_state = "Open" if predictions[0] > eye_alarm_threshold else "Closed"

        # 在标签上更新眼睛状态
        eye_label.config(text=f"Eye State: {eye_state}")

        # 更新历史数据
        now = datetime.now()
        eye_history.append((now, predictions[0]))

        # 绘制图表
        x_data.append(now)
        eye_y_data.append(predictions[0])
        eye_line.set_data(x_data, eye_y_data)
        ax.relim()
        ax.autoscale_view()

        # 检测到闭眼表示警觉，
        if eye_state == "Closed":
            print("警觉")
            alarm_sound.play()  # 播放警报声音

        # 检测人脸并估计头部姿态
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            yaw, pitch, roll = estimate_head_pose(face_roi)
            head_label.config(text=f"Head Angle: Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f} degrees")

            # 更新历史头部数据
            head_history.append((now, (yaw, pitch, roll)))

            # 绘制头部角度图表
            head_y_data.append(pitch)  # 以头部俯仰角为例
            head_line.set_data(x_data, head_y_data)
            ax2.relim()
            ax2.autoscale_view()

            # 检测头部姿态异常表示警觉
            if abs(pitch) > head_alarm_threshold:
                print("头部姿态异常，警觉")
                alarm_sound.play()  # 播放警报声音
        else:
            head_label.config(text="Head Angle: Not Detected")

        # 更新界面
        root.after(10, update_eye_and_head_state)

def toggle_running():
    global is_running
    is_running = not is_running
    if is_running:
        start_button.config(text="停止")
    else:
        start_button.config(text="开始")

def save_screenshot():
    now = datetime.now()
    filename = f"{screenshot_dir}/screenshot_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
    cv2.imwrite(filename, frame)
    print(f"截图已保存为 {filename}")

# 创建开始/停止按钮
start_button = ttk.Button(root, text="开始", command=toggle_running)
start_button.pack()

# 创建保存截图按钮
screenshot_button = ttk.Button(root, text="保存截图", command=save_screenshot)
screenshot_button.pack()

# 在主窗口中添加一个退出按钮
exit_button = ttk.Button(root, text="退出", command=root.quit)
exit_button.pack()

# 启动UI更新函数
update_eye_and_head_state()

# 进入主事件循环
root.mainloop()

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
