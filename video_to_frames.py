import cv2
import os

# Open the video file
file_name = 'IMG_1336'
os.chdir('')
cap = cv2.VideoCapture('Videos/' + file_name + '.mp4')
skip = 5


# Get fps and number of frames
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(fps, num_frames)

label = file_name + '_skip_' + str(skip)

# Create a directory to store the frames
if not os.path.exists('Videos/' + label):
    os.makedirs('Videos/' + label)

# Extract the frames
for i in range(num_frames):
    ret, frame = cap.read()
    # Extract the frame every 5 frames
    if i % skip == 0:
        cv2.imwrite('Videos/' + file_name + '/frame{:06d}.jpg'.format(i), frame)
    # cv2.imwrite('frame{:06d}.jpg'.format(i), frame)

# Release the video capture object
cap.release()