#!/usr/bin/python
from ServoController.WalkToTargetController import WalkToTargetController
import cv2
import numpy as np

walk_to_target = WalkToTargetController(speed=0.4)

while True:
    annotated_frame = walk_to_target.go_to_target()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Video Capture...")
        break
walk_to_target.locator.release_capture()
