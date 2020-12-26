import cv2
import numpy as np
from cv2 import aruco


class AntDetectorCamera:
    Ant_MARKER_ID = 1
    ARENA_CORNER_MARKER_ID_MAP = {2: 0, 3: 1, 4: 2, 5: 3}
    SCREEN_WIDTH = 575
    SCREEN_HEIGHT = 480
    X_SCALE = 1.4
    Y_SCALE = 1.5

    def __init__(self, calibration_data_path="LocationCamera/calib.npz"):
        self.marker_dict = {}
        self.perspective_matrix = np.identity(3)
        self.cap = cv2.VideoCapture(0)

        self.mtx, self.dist, self.new_camera_matrix, self.roi = self._get_fisheye_correction_params(
            calibration_data_path)

        self.aruco_parameters = aruco.DetectorParameters_create()
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        _, self.current_raw_frame = self.cap.read()
        self.arena_bounds, self.ant_centre, self.ant_front, self.ant_orientation_vector = None, None, None, None
        self.target = None

    @staticmethod
    def initialize_mouse_handler(event, x, y, _flags, ant_locator):
        if event == cv2.EVENT_LBUTTONDOWN:
            ant_locator.target = np.array([x, y])

    def ant_orientation_angle(self):
        if self.ant_orientation_vector is None:
            return None
        return np.arctan2(self.ant_orientation_vector[0], self.ant_orientation_vector[1])

    def _detect_location_and_orentation(self):
        marker_corners = np.array([self.marker_dict[self.Ant_MARKER_ID]])
        corrected_marker_corners = cv2.perspectiveTransform(marker_corners, self.perspective_matrix)[0]
        self.ant_centre = self._get_centre(corrected_marker_corners)
        self.ant_front = (corrected_marker_corners[0] + corrected_marker_corners[
            3]) / 2  # centre of the front left and front right corner of marker
        unnormalised_orientation_vector = self.ant_front - self.ant_centre
        self.ant_orientation_vector = unnormalised_orientation_vector / np.linalg.norm(unnormalised_orientation_vector)

    @staticmethod
    def _get_centre(marker_corners):
        return sum(marker_corners) / 4

    def _get_arena_marker_centres(self):
        """populates an array with corners. Assumes that all corners were detected"""
        arena_marker_points = np.ones((4, 2))
        for id, marker_corners in self.marker_dict.items():
            if id in self.ARENA_CORNER_MARKER_ID_MAP:
                arena_marker_points[self.ARENA_CORNER_MARKER_ID_MAP[id]] = self._get_centre(marker_corners)
        return arena_marker_points

    def _get_arena_bounding_box(self):
        """populates an array with corners. Assumes that all corners were detected"""
        arena_marker_points = self._get_arena_marker_centres()
        arena_centre = self._get_centre(arena_marker_points)
        self.arena_bounds = arena_centre + (arena_marker_points - arena_centre) * np.array([self.X_SCALE, self.Y_SCALE])

    def _is_arena_detected(self):
        for id in self.ARENA_CORNER_MARKER_ID_MAP:
            if id not in self.marker_dict:
                return False
        return True

    def _get_marker_dict_and_annotated_frame(self, img):
        """OpenCV's raw inputs are annoying. This cleans it up to make it more usable. Fetches all the corner ids and
        such from the image, and then returns an annotated image"""
        corners, ids, _rejected_points = aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_parameters)
        marker_dict = {}
        if ids is None:
            return marker_dict, img
        for marker_corners, id in zip(corners, ids):
            marker_dict[id[0]] = marker_corners[0]
        return marker_dict, aruco.drawDetectedMarkers(img, corners, ids)

    def _is_ant_detected(self):
        return self.Ant_MARKER_ID in self.marker_dict

    def _set_perspective_matrix(self, x_border=0, y_border=0):
        dst = np.array([
            [self.SCREEN_WIDTH - 1, self.SCREEN_HEIGHT - 1],
            [0, self.SCREEN_HEIGHT - 1],
            [0, 0],
            [self.SCREEN_WIDTH - 1, 0],

        ], dtype="float32") + np.array([[y_border, x_border]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        self.perspective_matrix = cv2.getPerspectiveTransform(self.arena_bounds.astype(np.float32), dst)

    def correct_image_perspective(self, image, x_border=0, y_border=0):
        return cv2.warpPerspective(image, self.perspective_matrix,
                                   (self.SCREEN_WIDTH + 2 * x_border, self.SCREEN_HEIGHT + 2 * y_border))

    def _get_fisheye_correction_params(self, calibration_data_path):
        with np.load(calibration_data_path) as data:
            mtx = data['mtx']
            dist = data['dist']
        _, frame = self.cap.read()
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1,
                                                               (w, h))
        return mtx, dist, new_camera_matrix, roi

    def _correct_for_fisheye(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.new_camera_matrix)

    def correct_and_annotate_frame(self, image):
        """corrects for Fisheye, finds aruco icons and then corrects the perspective of the image. After that,
        we annotate the image with positions of the markers, robot position and orientation """
        undistorted_image = self._correct_for_fisheye(image)
        self.marker_dict, annotated_image = self._get_marker_dict_and_annotated_frame(undistorted_image)

        if self._is_arena_detected():
            self._get_arena_bounding_box()
            self._set_perspective_matrix()
        if self.arena_bounds is not None:
            cv2.polylines(annotated_image, [self.arena_bounds.astype(int).reshape((-1, 1, 2))], True, (255, 0, 0), 2)
        corrected_image = self.correct_image_perspective(annotated_image)
        if self._is_ant_detected():
            self._detect_location_and_orentation()
            pointer = np.asarray([self.ant_centre, self.ant_front])
            cv2.polylines(corrected_image, [pointer.astype(int).reshape((-1, 1, 2))], True, (0, 255, 0), 5)
        if self.target is not None:
            cv2.circle(corrected_image, tuple(self.target.astype(int)), 20, (0, 0, 255), 3)
        return corrected_image

    def release_capture(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    locator = AntDetectorCamera()
    _, frame = locator.cap.read()
    cv2.imshow('Processed vs unprocessed',
               frame)
    cv2.setMouseCallback("Processed vs unprocessed", locator.initialize_mouse_handler, locator)
    while True:
        _, frame = locator.cap.read()
        annotated_frame = locator.correct_and_annotate_frame(frame)
        cv2.imshow('Processed vs unprocessed',
                   np.hstack((annotated_frame, frame)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting Video Capture...")
            break
    locator.release_capture()
