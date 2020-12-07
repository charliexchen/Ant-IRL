#!/usr/bin/python
from LocationCamera.ArucoDetector import AntLocator
from ServoController.SerialServoController import SerialServoController
from ServoController.WalktCycleConfigParser import UnifiedFixedWalkController, WalkCommand
import cv2
import numpy as np

locator = AntLocator("LocationCamera/calib.npz")
_, frame = locator.cap.read()
cv2.imshow('Processed vs Unprocessed',
           frame)
cv2.setMouseCallback("Processed vs Unprocessed", locator.initialize_mouse_handler, locator)


class WalkToTargetController:
    ANGLE_TOLERANCE = 0.2  # Number of radians to target before stopping
    ANGLE_REALIGN_TOLERANCE = 0.4  # Number of radians before triggering realign
    POSITION_TOLERANCE = 20  # Number of pixels to target before stopping
    POSITION_REALIGN_TOLERANCE = 50  # Number of pixels to target before stopping

    def __init__(self, speed=0.3):
        self.unified_walk_controller = UnifiedFixedWalkController(speed=speed)
        self.servo_controller = SerialServoController("/dev/ttyUSB0")

    def align_orientation_command(self, orientation, target_orientation=0):
        discrepancy = orientation - target_orientation
        if discrepancy > np.pi:
            discrepancy -= np.pi * 2
        elif discrepancy < -np.pi:
            discrepancy += np.pi * 2
        if discrepancy > self.ANGLE_TOLERANCE:
            return WalkCommand.RIGHT_TURN
        elif discrepancy < -self.ANGLE_TOLERANCE:
            return WalkCommand.LEFT_TURN

    def align_position_to_target_command(self, position, orientation, target):
        if target is None:
            return WalkCommand.IDLE
        if abs(orientation) > self.ANGLE_TOLERANCE:
            return self.align_orientation_command(orientation)
        position_discrepancy = position - target
        if abs(position_discrepancy[1]) > self.POSITION_TOLERANCE:  # align out y coordinate first
            if position_discrepancy[1] > 0:
                return WalkCommand.BACK
            else:
                return WalkCommand.FORWARD
        elif abs(position_discrepancy[0]) > self.POSITION_TOLERANCE:
            if position_discrepancy[0] > 0:
                return WalkCommand.RIGHT
            else:
                return WalkCommand.LEFT
        else:
            return WalkCommand.IDLE

    def send_serial_command(self, position, orientation, target):
        command = self.align_position_to_target_command(position, orientation, target)
        serial_command = self.unified_walk_controller.get_next_step(command)
        self.servo_controller.send({id: com for id, com in enumerate(serial_command)})


walk_to_target = WalkToTargetController(0.4)
while True:
    _, frame = locator.cap.read()
    annotated_frame = locator.correct_and_annotate_frame(frame)
    cv2.imshow('Processed vs Unprocessed', annotated_frame)
    print(locator.ant_orientation_angle())
    walk_to_target.send_serial_command(locator.ant_centre, locator.ant_orientation_angle(), locator.target)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Video Capture...")
        break
locator.release_capture()
