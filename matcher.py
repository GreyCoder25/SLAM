import cv2
import numpy as np


class Matcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

    def match(self, descrs1, descrs2):
        matches = self.matcher.knnMatch(descrs1, descrs2, k=2)
        return matches


class Tracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21),
                              # maxLevel = 3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def track(self, img1, img2, img1_fts):
        tracked_fts, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, img1_fts, None, **self.lk_params)
        status = status.reshape(status.shape[0])
        fts1 = img1_fts[status == 1]
        fts2 = tracked_fts[status == 1]
        return fts1, fts2
