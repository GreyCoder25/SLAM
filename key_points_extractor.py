import cv2


class KeyPointsExtractor(object):
    def __init__(self, num_points=1000):
        self.num_points = num_points

    def extract(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, self.num_points, qualityLevel=0.01, minDistance=3)
        return corners
