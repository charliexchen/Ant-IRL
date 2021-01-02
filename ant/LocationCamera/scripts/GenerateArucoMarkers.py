import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

cap = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

fig = plt.figure()
nx = 2
ny = 3
for i in range(1, nx * ny + 1):
    ax = fig.add_subplot(ny, nx, i)
    img = aruco.drawMarker(aruco_dict, i, 8000)
    plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
    ax.axis("off")

plt.savefig("markers.pdf")
plt.show()
