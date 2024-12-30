# This module can be used to measure the distance between two faces
# It is used to measure how effective the morphing is to protect privacy

from deepface import DeepFace
from pprint import pprint

# List of available models
models = [
  "VGG-Face", #0
  "Facenet", #1
  "Facenet512", #2
  "OpenFace", #3
  "DeepFace", #4
  "DeepID", #5
  "ArcFace", #6
  "Dlib", #7
  "SFace", #8
  "GhostFaceNet" #9
]

model = models[2]

# Compare two images with each other using the specified model
result = DeepFace.verify(
  img1_path = "images/obama.jpg",
  img2_path = "images/obama.jpg",
  model_name = model
)

# Pretty print the result
pprint(result)