import cv2

# Open the video file
cap = cv2.VideoCapture("/home/john/code/minipro/Traffic IP Camera video.mp4")  # Replace with your video file path

# Get the frame rate
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Print the frame rate
print("Frame Rate:", frame_rate)

# Release the video capture object
cap.release()