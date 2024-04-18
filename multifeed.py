import cv2
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk

# Function to process video feed
def process_video_feed(cap, label):
    print("Processing video feed...")
    ret, frame = cap.read()
    if ret:
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Initialize lists to store bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id in vehicle_class_ids:
                    # Object detected as a vehicle
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Add bounding box, confidence, and class ID to lists
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression to remove redundant bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Draw bounding boxes and count vehicles
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Vehicles: {len(indices)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to an ImageTk object
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)

        # Update the label with the new frame
        label.config(image=frame)
        label.image = frame

        print("Finished processing video feed.")
        # Call this function again after a delay
        label.after(20, process_video_feed, cap, label)
    else:
        print("Failed to read frame from video feed.")

# Load YOLO
net = cv2.dnn.readNet("/home/john/code/yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define vehicle class IDs
vehicle_class_ids = [2, 3, 5, 7]  # car, truck, bus, motorbike

# Open video captures
caps = [cv2.VideoCapture("Traffic IP Camera video.mp4") for _ in range(4)]

# Create Tkinter window
root = tk.Tk()
root.title("Multi-Feed Vehicle Detection")

# Create labels to display video feeds
labels = [tk.Label(root) for _ in range(4)]
for i, label in enumerate(labels):
    label.grid(row=i // 2, column=i % 2)

# Process each video feed
for cap, label in zip(caps, labels):
    process_video_feed(cap, label)

# Run the Tkinter event loop
root.mainloop()

# Release video captures
for cap in caps:
    cap.release()
