import cv2
import numpy as np

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

# Open video captures
video_files = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
caps = [cv2.VideoCapture(file) for file in video_files]

# Set up grid layout for displaying video feeds
rows = 2
cols = 2
grid_size = (rows, cols)
grid_width = 800
grid_height = 600
output_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

# Process each video feed
while True:
    for i, cap in enumerate(caps):
        # Read every 25th frame
        for _ in range(25):
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame.")
                break

        if not ret:
            break

        # Resize frame to fit grid
        frame = cv2.resize(frame, (grid_width // cols, grid_height // rows))

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
      cv2.putText(frame, f'Vehicle', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for i, cap in enumerate(caps): 

   # Place frame in grid
      y_start = (i // cols) * (grid_height // rows)
      y_end = y_start + (grid_height // rows)  # Corrected calculation
      if y_end > grid_height:
            y_end = grid_height
      x_start = (i % cols) * (grid_width // cols)
      x_end = x_start + (grid_width // cols)
      if x_end > grid_width:
            x_end = grid_width

      print("x_start:", x_start)
      print("x_end:", x_end)
      print("y_start:", y_start)
      print("y_end:", y_end)
      print("Frame shape:", frame.shape)
      print("Output frame slice shape:", output_frame[y_start:y_end, x_start:x_end].shape)

      output_frame[y_start:y_end, x_start:x_end] = frame

    # Display grid
    cv2.imshow("Grid", output_frame)

    # Check for user input to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and close OpenCV windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
