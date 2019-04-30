import cv2
import numpy as np


class KeyPointsExtractor(object):
    def __init__(self, num_points=3000, quality=0.01, min_dist=3):
        self.num_points = num_points
        self.quality = quality
        self.min_dist = min_dist
        self.orb = cv2.ORB_create(nfeatures=num_points)

    def extract(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, self.num_points, qualityLevel=self.quality, minDistance=self.min_dist)
        key_points = [cv2.KeyPoint(corner[0][0], corner[0][1], _size=20) for corner in corners]
        # TODO: learn how the input list of keypoints can be changed while computing the descriptors
        key_points, descriptors = self.orb.compute(img, key_points)
        # return corners, descriptors
        return key_points, descriptors

    def extract_orb(self, img):
        kps, des = self.orb.detectAndCompute(img, None)
        kps = [kp.pt for kp in kps]
        return kps, des

