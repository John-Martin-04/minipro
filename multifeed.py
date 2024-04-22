import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define vehicle class IDs
vehicle_class_ids = [2, 3, 5, 7]  # car, truck, bus, motorbike

# Open video capture objects (replace with your video paths)
video_paths = ["Traffic IP Camera video.mp4", "Traffic IP Camera video.mp4", "Traffic IP Camera video.mp4", "Traffic IP Camera video.mp4"]
video_captures = [cv2.VideoCapture(path) for path in video_paths]

# Function to capture video, perform detection, and update label
def update_video_feed(label, video_capture):
    while True:
        ret, frame = video_capture.read()
        if ret:
            try:
                # Process frame with vehicle detection logic
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                # Process detections
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if class_id in vehicle_class_ids and confidence > 0.5:
                            # Process the detection (draw bounding box, etc.)
                            height, width, _ = frame.shape
                            box = detection[0:4] * np.array([width, height, width, height])
                            (x, y, w, h) = box.astype("int")
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Convert OpenCV frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert OpenCV image to Tkinter format
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)

                # Update label with the new image
                label.config(image=photo)
                label.image = photo

                # Delay for 33 milliseconds (30 fps)
                time.sleep(0.033)
            except Exception as e:
                print("Error:", e)
        else:
            print("End of video or error occurred")
            break

# Create the main window
root = tk.Tk()
root.title("Vehicle Detection - 4 Feeds")

# Define video feed layouts
width, height = 320, 240  # Example frame dimensions
grid_cols, grid_rows = 2, 2  # 2x2 grid

# Create empty labels to hold video frames
video_labels = []
for i in range(grid_rows):
    for j in range(grid_cols):
        label = tk.Label(root, width=width, height=height)
        label.grid(row=i, column=j)
        video_labels.append(label)

# Start video updates for each label in a separate thread
threads = []
for label, video_capture in zip(video_labels, video_captures):
    thread = threading.Thread(target=update_video_feed, args=(label, video_capture))
    thread.daemon = True  # Make the thread a daemon so it terminates when the main thread terminates
    thread.start()
    threads.append(thread)

# Start the main event loop for tkinter
root.mainloop()

# Release video captures when closing (optional)
for cap in video_captures:
    cap.release()
