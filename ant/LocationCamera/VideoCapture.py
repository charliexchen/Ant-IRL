import cv2
import numpy as np
import pyboof as pb
from collections import deque

cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_EXPOSURE,-10)
pb.init_memmap()

detector = pb.FactoryFiducial(np.uint8).qrcode()


def draw_box(image, box, colour, thickness=3):
    points = np.asarray([vertex.get_tuple() for vertex in box.vertexes])
    cv2.polylines(image, [points.reshape((-1, 1, 2)).astype(np.int32)], True, colour, thickness)


def find_centre(box):
    points = np.asarray([vertex.get_tuple() for vertex in box.vertexes])
    return sum(points) / 4

tracker = []

while True:
    _, frame = cap.read()

    gray = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    detector.detect(pb.ndarray_to_boof(gray))

    for q in detector.detections:
        draw_box(frame, q.bounds, (255, 0, 255))
        draw_box(frame, q.pp_corner, (255, 0, 0))
        draw_box(frame, q.pp_down, (0, 0, 255))
        draw_box(frame, q.pp_right, (0, 255, 0))

        centre = tuple(int(i) for i in find_centre(q.bounds))
        direction = (find_centre(q.pp_right)+find_centre(q.pp_corner))/2
        tracker.append(centre)
        if len(tracker)>1000:
            tracker.pop(0)

        nd_tracker = np.asarray(tracker)

        cv2.line(frame, centre , tuple(int(i) for i in direction), (255, 0, 0), 5)
        cv2.polylines(frame, [nd_tracker.reshape((-1, 1, 2))], False, (0, 255, 0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Video Capture...")
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
