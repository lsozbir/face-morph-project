# This module is used for Delaunay triangulation calculations.

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay

# image : image array
# points : Mediapipe normalized face landmark detections
def calculateDelaunayTriangles(image: cv2.typing.MatLike, points):
    return Delaunay(points).simplices

def calculateDelanuayTriangleCoordinates(dt, landmarks):
    result = np.zeros((len(dt), 3, 2), dtype = int)
    for index in range(len(dt)):
        for p in range(3):
            result[index][p][0] = int(landmarks[dt[index][p]][0])
            result[index][p][1] = int(landmarks[dt[index][p]][1])
    return result

# VISUAL FUNCTION
# idtp: image_delaunay_triangle_pixels
def drawDelanuayTriangles(image, idtp):
    result = np.copy(image)
    for i in range(len(idtp)):
        cv2.line(result, idtp[i][0], idtp[i][1], (255,0,0), 1)
        cv2.line(result, idtp[i][1], idtp[i][2], (255,0,0), 1)
        cv2.line(result, idtp[i][2], idtp[i][0], (255,0,0), 1)
    return result
