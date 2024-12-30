# This module is used for runtime benchmarking.

import cv2
import VideoLandmarkDetector
import ImageLandmarkDetector
import DelaunayTriangulation
import Warp
import time
import FaceMorpher

# LOW, MEDIUM, HIGH RESOLUTIONS
resolutions = [[1920, 1080],
               [1280, 720],
               [960, 540]]


# OpenCV Settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[2][0]) #w
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[2][1]) #h
vld = VideoLandmarkDetector.VideoLandmarkDetector()
vld2 = VideoLandmarkDetector.VideoLandmarkDetector()
ild = ImageLandmarkDetector.ImageLandmarkDetector()


def measureSubFunctions():

    #BENCHMARK VARIABLES
    loops = 0
    capture_total = 0
    landmark_total = 0
    delaunay_total = 0
    warp_total = 0
    
    #IMAGE
    img = cv2.imread('images/obama.jpg')
    _, img_landmarks = ild.detectImageLandmarks(img)

    #VIDEO
    while True:
        loops += 1
        if(loops == 1000):
            break

        # CAMERA CAPTURE BENCHMARK
        capture_start = time.perf_counter()
        _, frame = cap.read()
        capture_time = time.perf_counter() - capture_start
        capture_total += capture_time

        # FRAME LANDMARK BENCHMARK
        landmark_start = time.perf_counter()
        frame_mp_landmarks, frame_landmarks = vld.detectVideoLandmarks(frame, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        landmark_time = time.perf_counter() - landmark_start
        landmark_total += landmark_time

        if(len(frame_mp_landmarks.face_landmarks) == 0):
            cv2.imshow("Result", frame)
            k2 = cv2.waitKey(1)
            if k2 == ord('q'):
                break
            continue

        # DELAUNAY CALCULATION BENCHMARK
        delaunay_start = time.perf_counter()
        dti = DelaunayTriangulation.calculateDelaunayTriangles(frame, frame_landmarks[:468])
        fdtp = DelaunayTriangulation.calculateDelanuayTriangleCoordinates(dti, frame_landmarks[:468])
        idtp = DelaunayTriangulation.calculateDelanuayTriangleCoordinates(dti, img_landmarks[:468])
        delaunay_time = time.perf_counter() - delaunay_start
        delaunay_total += delaunay_time
        
        # WARPING BENCHMARK
        warp_start = time.perf_counter()
        result, _ = Warp.warpTriangles(frame, img, fdtp, idtp, dti, 1)
        warp_time = time.perf_counter() - warp_start
        warp_total += warp_time

        cv2.imshow("Result", result)
        k2 = cv2.waitKey(1)
        if k2 == ord('q'):
            break

    # PRINT RESULTS
    print("TOTAL LOOPS: ", loops)
    print("capture_total sum= ", capture_total, "avg=", capture_total / loops)
    print("landmark_total sum= ", landmark_total, "avg=", landmark_total / loops)
    print("delaunay_total sum= ", delaunay_total, "avg=", delaunay_total / loops)
    print("warp_total sum= ", warp_total, "avg=", warp_total / loops)


def measureTotal():
    loops = 0
    total_total = 0
    fm = FaceMorpher.FaceMorpher(cap, 'images/obama.jpg', 1)
    while True:
        loops += 1
        if(loops == 1000):
            break

        # TOTAL TIME BENCHMARK
        total_start = time.perf_counter()
        result = fm.getNextFrame()
        total_time = time.perf_counter() - total_start
        total_total += total_time

        cv2.imshow("Result", result)
        k2 = cv2.waitKey(1)
        if k2 == ord('q'):
            break
    

    print("TOTAL LOOPS: ", loops)
    print("total_total sum= ", total_total, "AVG TIME=", total_total / loops, "AVG FPS=", 1 / (total_total / loops))

#measureSubFunctions()
measureTotal()
