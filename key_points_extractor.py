import cv2
import numpy as np


class KeyPointsExtractor(object):
    def __init__(self, extractor='orb', detector='fast', num_points=3000, quality=0.01, min_dist=3):
        self.num_points = num_points
        self.quality = quality
        self.min_dist = min_dist

        self.extractor_type = extractor
        self.extractor = None
        self.detector_type = detector
        self.detector = None

        self._init_extractor()
        self._init_detector()

    def _init_extractor(self):
        if self.extractor_type == 'orb':
            orb = cv2.ORB_create(nfeatures=self.num_points)
            self.extractor = orb

    def _init_detector(self):
        if self.detector_type == 'fast':
            fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
            self.detector = fast

    def _to_grayscale(self, img):
        if img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray
        return img

    def extract_fast(self, img):
        gray = self._to_grayscale(img)
        kps = self.detector.detect(gray, None)
        kps = np.array([kp.pt for kp in kps], dtype=np.float32)
        return kps

    def extract(self, img):
        gray = self._to_grayscale(img)
        corners = cv2.goodFeaturesToTrack(gray, self.num_points, qualityLevel=self.quality, minDistance=self.min_dist)
        key_points = [cv2.KeyPoint(corner[0][0], corner[0][1], _size=20) for corner in corners]
        key_points, descriptors = self.extractor.compute(img, key_points)
        # return corners, descriptors
        return key_points, descriptors

    def extract_orb(self, img):
        kps, des = self.orb.detectAndCompute(img, None)
        # kps = [kp.pt for kp in kps]
        return kps, des

