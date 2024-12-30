# This module provides a function to warp triangles from image to frame.

import numpy as np
import cv2

# Indices of Mediapipe inner lips landmarks used for inner mouth correction
innerLips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]

# Method to warp triangles from image to frame
# fdtp: frame_delaunay_triangle_pixels
# idtp: image_delaunay_triangle_pixels
# dti: delaunay_triangle_indices
def warpTriangles(frame, image, fdtp, idtp, dti, alpha):

    # Arrange initial variables
    alpha = max(0.0, min(alpha, 1.0)) # alpha is % of image
    beta = 1 - alpha # beta is % of frame
    substitute_face = np.zeros_like(frame)
    face_mask = np.ones_like(frame, dtype=np.float32)
    result = np.empty_like(frame)
    
    # Start looping all triangles
    for tri in range(len(dti)):

        # Skip warping inside lips to provide natural looking images
        if dti[tri][0] in innerLips and dti[tri][1] in innerLips and dti[tri][2] in innerLips:
            continue

        # Find bounding rectangle for each triangle
        # r1, r2 is (x, y, width , height)
        # https://docs.opencv.org/4.10.0/dd/d49/tutorial_py_contour_features.html
        r1 = cv2.boundingRect(idtp[tri])
        r2 = cv2.boundingRect(fdtp[tri])

        # Offset points
        t1Rect = [(pt[0] - r1[0], pt[1] - r1[1]) for pt in idtp[tri]]
        t2Rect = [(pt[0] - r2[0], pt[1] - r2[1]) for pt in fdtp[tri]]

        # Create the masks that will be used later
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0)
        cv2.fillConvexPoly(face_mask, np.int32(fdtp[tri]), (beta, beta, beta), 16, 0)

        # Extract the triangle region from the image
        imageRect = image[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        # Apply affine transform to the region from image and warp it to frame
        warpMat = cv2.getAffineTransform(np.float32(t1Rect), np.float32(t2Rect))
        frameRect = cv2.warpAffine(imageRect, warpMat, (r2[2], r2[3]), None,
                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        # Apply mask to the warped image region
        frameRect = frameRect * mask

        # Construct the substitute face
        substitute_face[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = (
            (substitute_face[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]) * ((1.0, 1.0, 1.0) - mask) + frameRect
        )
    
    # Apply the substite face to the result with respect to alpha
    result[:, :] = (frame * face_mask) + (alpha, alpha, alpha) * substitute_face[:, :]

    return result, substitute_face
    