'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
'''

import numpy as np
import cv2
import glob
import pickle

# Define the chess board rows and columns
rows = 6
cols = 8

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Create the arrays to store the object points and the image points
objectPointsArray = []
imgPointsArray = []

cap = cv2.VideoCapture(0)
frames = []
while True:
    _, frame = cap.read()
    gray = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cv2.imshow("frame", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Video Capture...")
        break
    if cv2.waitKey(10) & 0xFF == ord('p'):
        print("snap!")
        frames.append(frame)
cap.release()
old_frames = pickle.load(open('calibration_frames.p', "rb"))
frames = old_frames + frames
pickle.dump(frames, open('calibration_frames.p', "wb"))

for img in frames:
    # Load the image and convert it to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols))

    print(ret)
    # Make sure the chess board pattern was found in the image
    if ret:
        # Refine the corner position
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Add the object points and the image points to the arrays
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)

        # Draw the corners on the image
        cv2.drawChessboardCorners(img, (rows, cols), corners, ret)

    # Display the image
    cv2.imshow('chess board', img)
    cv2.waitKey(500)

# Calibrate the camera and save the results
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
np.savez('../calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Print the camera calibration error
error = 0

for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

print("Total error: ", error / len(objectPointsArray))




cap = cv2.VideoCapture(0)
frames = []
while True:
    _, frame = cap.read()
    h, w = frame.shape[:2]

    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(frame, mtx, dist, None, newCameraMtx)

    cv2.imshow('chess board', np.hstack((frame, undistortedImg)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Video Capture...")
        break
cap.release()
cv2.destroyAllWindows()
