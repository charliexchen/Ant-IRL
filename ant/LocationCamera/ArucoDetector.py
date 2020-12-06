import cv2
import numpy as np
from cv2 import aruco

cap = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)


class AntLocator():
    Ant_MARKER_ID = 1
    ARENA_CORNER_MARKER_ID_MAP = {2: 0, 3: 1, 4: 2, 5: 3}
    X_SCALE = 1.4
    Y_SCALE = 1.5

    def __init__(self, calibration_data='calib.npz'):
        with np.load(calibration_data) as data:
            self.mtx = data['mtx']
            self.dist = data['dist']

    def detect_location(self, frame):
        pass

    def _get_centre(self, marker_corners):
        return sum(marker_corners) / 4

    def _get_arena_marker_centres(self, all_marker_corners, ids):
        '''populates an array with corners. Assumes that all corners were detected'''
        arena_marker_points = np.ones((4, 2))
        for marker_corners, id in zip(all_marker_corners, ids):
            if id[0] in self.ARENA_CORNER_MARKER_ID_MAP:
                arena_marker_points[self.ARENA_CORNER_MARKER_ID_MAP[id[0]]] = self._get_centre(marker_corners[0])
        return arena_marker_points

    def _get_arena_bounding_box(self, all_marker_corners, ids):
        '''populates an array with corners. Assumes that all corners were detected'''
        arena_marker_points = self._get_arena_marker_centres(all_marker_corners, ids)
        arena_centre = self._get_centre(arena_marker_points)
        return arena_centre + (arena_marker_points - arena_centre) * np.array([self.X_SCALE, self.Y_SCALE])

    def get_arena(self, frame):
        pass
    def _is_arena_detected(self, ids):
        if ids is None:
            return False
        for id in self.ARENA_CORNER_MARKER_ID_MAP:
            if id not in [id_[0] for id_ in ids]:
                return False
        return True


def four_point_transform(image, pts):
    width = 480
    height = 575
    dst = np.array([
        [height - 1, width - 1],
        [0, width-1],
        [0, 0],
        [height - 1, 0],

       ], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (height, width))
    # return the warped image
    return warped


with np.load('calib.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

locator = AntLocator()

while True:
    _, frame = cap.read()
    h, w = frame.shape[:2]
    print(h, w)
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(frame, mtx, dist, None, newCameraMtx)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(undistortedImg, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(undistortedImg.copy(), corners, ids)
    if locator._is_arena_detected(ids):
        bound = locator._get_arena_bounding_box(corners, ids)
        cv2.polylines(frame_markers, [bound.astype(int).reshape((-1, 1, 2))], True, (255, 0, 0), 2)
        cv2.imshow('Processed vs unprocessed',
               np.hstack((frame, four_point_transform(frame_markers, bound.astype(np.float32)))))
    else:
        cv2.imshow('Processed vs unprocessed',
                   np.hstack((frame,frame_markers)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Video Capture...")
        break

cap.release()
cv2.destroyAllWindows()
