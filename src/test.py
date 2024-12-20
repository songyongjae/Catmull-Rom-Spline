import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

name = 'act4'

# Load the YOLO model
model = YOLO('cbest.pt')
video_path = f"{name}.mp4"
cap = cv2.VideoCapture(video_path)

# Lists to store coordinates and bounding box sizes
x_coords = []
y_coords = []
widths = []
heights = []
frame_numbers = []

# DataFrame to store the results
df = pd.DataFrame(columns=['File_num', 'X', 'Y', 'H', 'W'])

frame_count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Check if any objects are detected
        if len(results) > 0 and len(results[0].boxes.xyxy) > 0:
            # First detected object is robot
            obj = results[0]

            # Bounding box coordinates
            box = obj.boxes.xyxy[0].cpu().numpy()
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]

            # Append center coordinates and box sizes to the lists
            x_coords.append(x_center)
            y_coords.append(y_center)
            widths.append(box_width)
            heights.append(box_height)
            frame_numbers.append(frame_count)

            # Append data to DataFrame
            df = df.append({'File_num': frame_count, 'X': x_center/720, 'Y': y_center/1280, 'H': box_height/1280, 'W': box_width/720}, ignore_index=True)
            
            # Visualize the results on the frame
            annotated_frame = obj.plot()

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)
        else:
            # No detection case
            df = df.append({'File_num': frame_count, 'X': 0, 'Y': 0, 'H': 0, 'W': 0}, ignore_index=True)
            #df = df.append({'File_num': frame_count, 'X': np.nan, 'Y': np.nan, 'H': np.nan, 'W': np.nan}, ignore_index=True)

        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Save the DataFrame to an Excel file
df.to_excel(f'{name}_test.xlsx', index=False)

# Plotting
plt.figure()
plt.plot(x_coords, y_coords, 'ro-', linewidth=0.3, alpha=0.5, label='Trajectory')

# Adding arrows to indicate direction
for i in range(1, len(x_coords)):
    plt.quiver(x_coords[i-1], y_coords[i-1], x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1], angles='xy', scale_units='xy', scale=1, color='blue')

# Custom legend for the direction arrow
arrow_legend = Line2D([0], [0], color='blue', marker='>', markersize=10, label='Time direction', linestyle='None')

# Add the custom legend
plt.legend(handles=[Line2D([0], [0], color='r', linewidth=0.3, alpha=0.5, label='Trajectory'), arrow_legend])

plt.title('Motion Data Before Interpolation')
plt.xlabel('X (pixel)')
plt.ylabel('Y (pixel)')
plt.savefig(f'{name}_testing.png')
plt.show()

