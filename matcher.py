import cv2
import numpy as np


class Matcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

    def match(self, descrs1, descrs2):
        matches = self.matcher.knnMatch(descrs1, descrs2, k=2)
        return matches
