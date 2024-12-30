# This module is a playground to show multiple outputs between processes. Running main.py renders all intermediate outputs.

import cv2
import VisualUtils as vu
import VideoLandmarkDetector
import ImageLandmarkDetector
import DelaunayTriangulation
import Warp
import FaceMorpher

# OpenCV Settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

vld = VideoLandmarkDetector.VideoLandmarkDetector()
ild = ImageLandmarkDetector.ImageLandmarkDetector()

def run():
    
    # Read the image and detect image landmarks
    img = cv2.imread('images/obama.jpg')
    img_mp_landmarks, img_landmarks = ild.detectImageLandmarks(img)

    # Draw image landmarks on screen
    img_landmarks_visual = vu.draw_landmarks_on_image(img, img_mp_landmarks)
    img_landmarks_visual = vu.ResizeWithAspectRatio(img_landmarks_visual, width=960)
    cv2.imshow("Image Landmarks Visualization", img_landmarks_visual)

    # Start camera loop
    while True:
        
        # Read/show the camera frame and detect camera landmarks
        _, frame = cap.read()
        frame_mp_landmarks, frame_landmarks = vld.detectVideoLandmarks(frame, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        cv2.imshow("Frame", frame)

        # Check if camera landmarks are detected successfully
        if(len(frame_mp_landmarks.face_landmarks) == 0):
            # If nothing is detected, show the original camera frame in all windows
            cv2.imshow("Frame Landmarks Visualization", frame)
            cv2.imshow("Frame Delaunay Visualization", frame)
            cv2.imshow("Substitute Face", frame)
            cv2.imshow("Result", frame)
            k2 = cv2.waitKey(1)
            if k2 == ord('q'):
                break
            continue

        # Delaunay triangulation calculations
        # dti: delaunay_triangle_indices
        # fdtp: frame_delaunay_triangle_pixels
        # idtp: image_delaunay_triangle_pixels
        dti = DelaunayTriangulation.calculateDelaunayTriangles(frame, frame_landmarks[:468])
        fdtp = DelaunayTriangulation.calculateDelanuayTriangleCoordinates(dti, frame_landmarks[:468])
        idtp = DelaunayTriangulation.calculateDelanuayTriangleCoordinates(dti, img_landmarks[:468])

        # Visualize frame landmarks and frame delaunay triangulation 
        frame_landmarks_visual = vu.draw_landmarks_on_image(frame, frame_mp_landmarks)
        frame_delanuay_visual = DelaunayTriangulation.drawDelanuayTriangles(frame, fdtp)
        cv2.imshow("Frame Landmarks Visualization", frame_landmarks_visual)
        cv2.imshow("Frame Delaunay Visualization", frame_delanuay_visual)

        # Do triangle warping and show the results
        result, substitute_face = Warp.warpTriangles(frame, img, fdtp, idtp, dti, 1)
        cv2.imshow("Substitute Face", substitute_face)
        cv2.imshow("Result", result)

        k2 = cv2.waitKey(1)
        if k2 == ord('q'):
            break

# Simple example of face protection module. Other applications can easily use our implementation this way.
def moduleTest():
    fm = FaceMorpher.FaceMorpher(cap, 'images/obama.jpg', 1)
    while True:
        result = fm.getNextFrame()
        cv2.imshow("Result", result)
        k2 = cv2.waitKey(1)
        if k2 == ord('q'):
            break

run()
#moduleTest()
