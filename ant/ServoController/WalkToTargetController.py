import cv2
import numpy as np

from LocationCamera.AntDetectorCamera import AntDetectorCamera
from ServoController.SerialServoController import SerialServoController
from ServoController.WalktCycleConfigParser import (
    UnifiedFixedWalkController,
    WalkCommand,
)


class WalkToTargetController:
    """
    Combines AntDetectorCamera, SerialServoController and UnifiedFixedWalkController so we can have a simple feedback
    loop of the robot walking to a point on the screen using fixed walk cycles. This will be used to reset the
    environment and such
    """

    ANGLE_TOLERANCE = 0.3  # Number of radians to target before stopping
    POSITION_TOLERANCE = 20  # Number of pixels to target before stopping

    def __init__(self, port="/dev/ttyUSB0", speed=0.3, window_name="Ant Location"):
        self.unified_walk_controller = UnifiedFixedWalkController(speed=speed)
        self.servo_controller = SerialServoController(port)
        self.locator = AntDetectorCamera()
        self.window_name = window_name

        self.frame = None
        self.command = WalkCommand.IDLE
        self.serial_command = None
        self.set_mouse_callback()

    def set_mouse_callback(self):
        """
        We have to assign the mouse handler to the window so open CV would let us capture mouse events
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(
            self.window_name, self.locator.initialize_mouse_handler, self.locator
        )

    def align_orientation_command(self, orientation, target_orientation=np.pi / 2):
        if orientation is None:
            return WalkCommand.IDLE
        discrepancy = orientation - target_orientation
        if discrepancy > np.pi:
            discrepancy -= np.pi * 2
        elif discrepancy < -np.pi:
            discrepancy += np.pi * 2
        if discrepancy > self.ANGLE_TOLERANCE / 2:
            return WalkCommand.RIGHT_TURN
        elif discrepancy < -self.ANGLE_TOLERANCE / 2:
            return WalkCommand.LEFT_TURN

    def align_position_to_target_command(self, position, orientation, target):
        """
        Determines the command we should send based on the current state of the environment and the target. This doesn't
        need to be sophisticated, since it will just be used to reset the env.

        Robot prioritises aligning the orientation to the left. We want to minimise the robot turning, in order to
        reduce the chance of the robot getting tangled in the cables (it was really fiddly to solder and I do NOT want
        tear it apart). Then the robot tries to align the vertical, followed by the horizontal position.
        :param position:
        :param orientation:
        :param target:
        :return: WalkCommand which will help get to target
        """
        if target is None:
            return WalkCommand.IDLE
        orientation_command = self.align_orientation_command(orientation)
        if orientation_command is not None:
            return orientation_command
        position_discrepancy = position - target

        if np.linalg.norm(position_discrepancy) > 100:

            position_tolerance = self.POSITION_TOLERANCE * 2
        else:
            position_tolerance = self.POSITION_TOLERANCE

        if (
            abs(position_discrepancy[1]) > position_tolerance
        ):  # align y coordinate first
            if position_discrepancy[1] < 0:
                return WalkCommand.RIGHT
            else:
                return WalkCommand.LEFT

        elif abs(position_discrepancy[0]) > position_tolerance:
            if position_discrepancy[0] > 0:
                return WalkCommand.BACK
            else:
                return WalkCommand.FORWARD
        else:
            return WalkCommand.IDLE

    def send_serial_command(self, position, orientation, target):
        self.command = self.align_position_to_target_command(
            position, orientation, target
        )
        self.serial_command = self.unified_walk_controller.get_next_step(self.command)
        self.servo_controller.send(
            {id: com for id, com in enumerate(self.serial_command)}
        )

    def go_to_target(self, show_image=True):
        _, self.frame = self.locator.cap.read()
        annotated_frame = self.locator.correct_and_annotate_frame(self.frame)
        if show_image:
            cv2.imshow(self.window_name, annotated_frame)
        self.send_serial_command(
            self.locator.ant_centre,
            self.locator.ant_orientation_angle(),
            self.locator.target,
        )

    def end_session(self):
        self.locator.release_capture()
        self.servo_controller.close_ports()

    def get_target_normalised(self):
        return self.locator.target / max(
            self.locator.SCREEN_HEIGHT, self.locator.SCREEN_WIDTH
        )

    def set_target_normalised(self, normalised_target):
        self.locator.target = np.array(normalised_target) * max(
            self.locator.SCREEN_HEIGHT, self.locator.SCREEN_WIDTH
        )

    def get_normalised_position(self, show_image=True):
        _, self.frame = self.locator.cap.read()
        annotated_frame = self.locator.correct_and_annotate_frame(self.frame)
        if show_image:
            cv2.imshow(self.window_name, annotated_frame)
        return self.locator.ant_centre / max(
            self.locator.SCREEN_HEIGHT, self.locator.SCREEN_WIDTH
        )

    def get_orientation_vector(self):
        return self.locator.ant_orientation_vector
