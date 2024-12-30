# This module detects video landmarks from a video stream.

import mediapipe as mp
import numpy as np

class VideoLandmarkDetector:

    def __init__(self):
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # Mediapipe Landmarker Options
        self.model_path = 'face_landmarker.task'
        options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.VIDEO)

        # Mediapipe Create Face Landmarker
        self.landmarker = self.FaceLandmarker.create_from_options(options)

    def detectVideoLandmarks(self, frame, frame_timestamp):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Normalized face landmarks
        # https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/components/containers/NormalizedLandmark
        mp_result = self.landmarker.detect_for_video(mp_image, frame_timestamp)

        if(len(mp_result.face_landmarks) == 0):
            return mp_result, []
        

        height, width = frame.shape[:-1]
        mp_filtered_result = mp_result.face_landmarks[0]
        filtered_result = np.zeros((len(mp_filtered_result), 2))
        
        for i in range (len(mp_filtered_result)):
            filtered_result[i][0] = min(max(0.001, mp_filtered_result[i].x), 0.999) * width
            filtered_result[i][1] = min(max(0.001, mp_filtered_result[i].y), 0.999) * height

        # mp_result is normalized native result of MediaPipe. filtered_result is unnormalized ready to use results.
        return mp_result, filtered_result