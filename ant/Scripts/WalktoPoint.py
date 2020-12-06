from LocationCamera.ArucoDetector import AntLocator
from ServoController import UnifiedWalk
import cv2
import numpy as np

locator = AntLocator("../LocationCamera/calib.npz")
_, frame = locator.cap.read()
cv2.imshow('Processed vs unprocessed',
           frame)
cv2.setMouseCallback("Processed vs unprocessed", locator.initialize_mouse_handler, locator)


class WalkToTargetController()

while True:
    _, frame = locator.cap.read()
    annotated_frame = locator.correct_and_annotate_frame(frame)
    cv2.imshow('Processed vs unprocessed', annotated_frame)
    print (locator.ant_orientation_angle())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Video Capture...")
        break
locator.release_capture()
