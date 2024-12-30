# This module provides the most basic functionality without any visualizations to be used in other applications.

import cv2
import VideoLandmarkDetector
import ImageLandmarkDetector
import DelaunayTriangulation
import Warp

class FaceMorpher:

    # Initialize the face morpher
    def __init__(self, cap, image_path, alpha):
        self.cap = cap
        self.alpha = alpha
        self.vld = VideoLandmarkDetector.VideoLandmarkDetector()
        self.ild = ImageLandmarkDetector.ImageLandmarkDetector()

        self.img_file = cv2.imread(image_path)
        self.img = self.img_file
        self.img_mp_landmarks, self.img_landmarks = self.ild.detectImageLandmarks(self.img)

    # Read a frame from provided video device and apply protection
    def getNextFrame(self):
        
        # Read the camera frame
        _, frame = self.cap.read()
        
        # Check if camera landmarks are detected successfully
        frame_mp_landmarks, frame_landmarks = self.vld.detectVideoLandmarks(frame, int(self.cap.get(cv2.CAP_PROP_POS_MSEC)))
        if(len(frame_mp_landmarks.face_landmarks) == 0):
            # Return the original frame if no face is detected
            return frame

        # Delaunay triangulation calculations
        # dti: delaunay_triangle_indices
        # fdtp: frame_delaunay_triangle_pixels
        # idtp: image_delaunay_triangle_pixels
        dti = DelaunayTriangulation.calculateDelaunayTriangles(frame, frame_landmarks[:468])
        fdtp = DelaunayTriangulation.calculateDelanuayTriangleCoordinates(dti, frame_landmarks[:468])
        idtp = DelaunayTriangulation.calculateDelanuayTriangleCoordinates(dti, self.img_landmarks[:468])

        # Warp triangles
        result, _ = Warp.warpTriangles(frame, self.img, fdtp, idtp, dti, self.alpha)

        return result