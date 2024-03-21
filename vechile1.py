import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]

# Load image
image =  cv2.imread(r'C://Users/Sanjay/Downloads/car.png')
height, width, channels = image.shape

# Detect objects
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Initialize lists to store bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# Define vehicle class IDs
vehicle_class_ids = [2, 3, 5, 7]  # car, truck, bus, motorbike

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

# Initialize a set to store unique vehicle detections
unique_vehicles = set()

if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Count the number of vehicles
for i in indices:
    class_id = class_ids[i]
    center_x, center_y, w, h = boxes[i]
    unique_vehicles.add((class_id, center_x, center_y, w, h))

vehicles_count = len(unique_vehicles)
print("Number of vehicles detected:", vehicles_count)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()