import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

name = 'example4'

# Load the YOLO model
model = YOLO('model.pt')
video_path = f"./video/{name}.mp4"
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
            df = df.append({'File_num': frame_count, 'X': np.nan, 'Y': np.nan, 'H': np.nan, 'W': np.nan}, ignore_index=True)

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

# Interpolation function using CubicSpline with exception handling for trailing NaNs
def interpolate_nans_cubic(series):
    nans = series.isna()
    not_nans = ~nans
    indices = np.arange(len(series))
    
    # Perform cubic spline interpolation
    if not_nans.any():
        cs = CubicSpline(indices[not_nans], series[not_nans])
        interpolated_values = cs(indices)
        
        # Fill in the NaNs using the interpolated values
        series[nans] = interpolated_values[nans]

        # Handle trailing NaNs
        if nans.iloc[-1]:
            last_valid_index = not_nans[::-1].idxmax()
            series[last_valid_index+1:] = series[last_valid_index]
    
    return series

# Apply interpolation to each column
df['X'] = interpolate_nans_cubic(df['X'])
df['Y'] = interpolate_nans_cubic(df['Y'])
df['H'] = interpolate_nans_cubic(df['H'])
df['W'] = interpolate_nans_cubic(df['W'])

# Save the DataFrame to an Excel file
df.to_excel(f'{name}_interpolate.xlsx', index=False)

# Plotting
plt.figure()
plt.plot(df['X']*720, df['Y']*1280, 'ro-', linewidth=0.3, alpha=0.5, label='Trajectory')

# Adding arrows to indicate direction
for i in range(1, len(df)):
    plt.quiver(df['X'][i-1]*720, df['Y'][i-1]*1280, (df['X'][i] - df['X'][i-1])*720, (df['Y'][i] - df['Y'][i-1])*1280, angles='xy', scale_units='xy', scale=1, color='blue')

# Custom legend for the direction arrow
arrow_legend = Line2D([0], [0], color='blue', marker='>', markersize=10, label='Time direction', linestyle='None')

# Add the custom legend
plt.legend(handles=[Line2D([0], [0], color='r', linewidth=0.3, alpha=0.5, label='Trajectory'), arrow_legend])

plt.title('Motion Data After Interpolation')
plt.xlabel('X (pixel)')
plt.ylabel('Y (pixel)')
plt.savefig(f'{name}_interpolate2.png')
plt.show()
