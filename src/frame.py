import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load the YOLO model
model = YOLO('model.pt')
video_path = "./video/example4.mp4"
cap = cv2.VideoCapture(video_path)

# Lists to store coordinates and directions
x_coords = []
y_coords = []
u = []
v = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # object detection
        if len(results) > 0 and len(results[0].boxes.xyxy) > 0:
            # First detected object is robot
            obj = results[0]

            # Bounding box coordinates
            box = obj.boxes.xyxy[0]
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2

            if x_coords and y_coords:
                # Calculate the direction
                u.append(x_center - x_coords[-1])
                v.append(y_center - y_coords[-1])

            # Append center coordinates to the lists
            x_coords.append(x_center)
            y_coords.append(y_center)

            # Visualize the results on the frame
            annotated_frame = obj.plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Plotting
plt.figure()
plt.plot(x_coords, y_coords, 'ro-', linewidth=0.3, alpha=0.5, label='Trajectory')

# Adding arrows to indicate direction
for i in range(1, len(x_coords)):
    plt.quiver(x_coords[i-1], y_coords[i-1], u[i-1], v[i-1], angles='xy', scale_units='xy', scale=1, color='blue')
    
# Custom legend for the direction arrow
arrow_legend = Line2D([0], [0], color='blue', marker='>', markersize=10, label='Time direction', linestyle='None')

# Add the custom legend
plt.legend(handles=[Line2D([0], [0], color='r', linewidth=0.3, alpha=0.5, label='Trajectory'), arrow_legend])

plt.title('Robot Tracking with Direction Arrows')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.show()
