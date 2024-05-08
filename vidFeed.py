import cv2
import numpy as np
import time

frames=0
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

# Open video capture
cap = cv2.VideoCapture("video1.mp4")  # Capture from video file

# last_frame_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if ret :
        frames+=1
        if frames==25 :
            frames=0
        if frames !=0:
            time.sleep(1/35)
    if not ret:
        break
    
    # Process frame if at least one second has passed since the last frame processing
    # if time.time() - last_frame_time >= 1:
    # Detect objects
    if frames==0:
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

        # Display frame
        cv2.imshow("Frame", frame)
    # last_frame_time = time.time()

    # Check for user input to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
