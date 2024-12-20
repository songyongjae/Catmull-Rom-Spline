import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

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
df2 = pd.read_excel('act4_test.xlsx')

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


# Create the interpolated values
def catmull_rom_spline(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2*p0 - 5*p1 + 4*p2 - p3) * t2 + (-p0 + 3*p1 - 3*p2 + p3) * t3)

# Catmull-Rom spline interpolation function
def interpolate_nans_catmull_rom(series):
    nans = series.isna()
    not_nans = ~nans
    indices = np.arange(len(series))

    if not_nans.sum() < 4:
        # Not enough points to apply Catmull-Rom spline interpolation
        return series
    
    interp_values = []
    for i in range(len(series)):
        if not nans[i]:
            interp_values.append(series[i])
        else:
            # Find surrounding points
            idxs = indices[not_nans]
            pos = np.searchsorted(idxs, i)

            # Handle edge cases
            if pos == 0:
                interp_values.append(series[idxs[0]])
            elif pos >= len(idxs):
                interp_values.append(series[idxs[-1]])
            else:
                idx_before = idxs[pos - 1]
                idx_after = idxs[pos]
                idx_before_before = idxs[pos - 2] if pos - 2 >= 0 else idx_before
                idx_after_after = idxs[pos + 1] if pos + 1 < len(idxs) else idx_after

                t = (i - idx_before) / (idx_after - idx_before)
                interp_value = catmull_rom_spline(series[idx_before_before], series[idx_before], series[idx_after], series[idx_after_after], t)
                interp_values.append(interp_value)
    
    ret = np.array(pd.Series(interp_values))

    return ret
    #return pd.Series(interp_values)

# Interpolated DataFrame
df_interpolated = df.copy()

# Apply interpolation to each column
df_interpolated['X'] = interpolate_nans_catmull_rom(df['X'])
df_interpolated['Y'] = interpolate_nans_catmull_rom(df['Y'])
df_interpolated['H'] = interpolate_nans_catmull_rom(df['H'])
df_interpolated['W'] = interpolate_nans_catmull_rom(df['W'])

# Save the DataFrame to an Excel file
df_interpolated.to_excel(f'{name}_interpolate4.xlsx', index=False)

# Plotting
plt.figure()


# Plot original trajectory in red
plt.plot(df['X']*720, df['Y']*1280, 'ro-', linewidth=0.3, alpha=0.5, label='Original Trajectory')

# Plot interpolated points in green where original points were NaNs
nan_indices = df2['X'] == 0
plt.plot(df_interpolated.loc[nan_indices, 'X']*720, df_interpolated.loc[nan_indices, 'Y']*1280, 'go-', linewidth=0.3, alpha=0.5, label='Interpolated Points')


# Adding arrows to indicate direction
for i in range(1, len(df_interpolated)):
    if not pd.isna(df_interpolated['X'][i]) and not pd.isna(df_interpolated['Y'][i]):
        plt.quiver(df_interpolated['X'][i-1]*720, df_interpolated['Y'][i-1]*1280, (df_interpolated['X'][i] - df_interpolated['X'][i-1])*720, (df_interpolated['Y'][i] - df_interpolated['Y'][i-1])*1280, angles='xy', scale_units='xy', scale=1, color='blue')

# Custom legend for the direction arrow
arrow_legend = Line2D([0], [0], color='blue', marker='>', markersize=10, label='Time direction', linestyle='None')

# Add the custom legend
plt.legend(handles=[Line2D([0], [0], color='r', linewidth=0.3, alpha=0.5, label='Original Trajectory'), 
                    Line2D([0], [0], color='g', linewidth=0.3, alpha=0.5, label='Interpolated Points'),
                    arrow_legend])

plt.title('Motion Data After Interpolation')
plt.xlabel('X (pixel)')
plt.ylabel('Y (pixel)')
plt.savefig(f'{name}_interpolate1.png')
plt.show()
