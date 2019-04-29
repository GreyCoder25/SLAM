import cv2
import numpy as np
import matplotlib.pyplot as plt
from key_points_extractor import KeyPointsExtractor
from matcher import Matcher

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 450


class Viewer(object):
    def __init__(self):
        self.prev_frame_extracts = None
        self.prev_frame = None
        self.kpe = KeyPointsExtractor(num_points=1000, quality=0.01, min_dist=3)
        self.matcher = Matcher()

    @staticmethod
    def show_frame(img):
        cv2.imshow('Car driving video', img)
        print(img.shape)

    def process_frame(self, img):
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        key_points, descriptors = self.kpe.extract(img)
        # key_points, descriptors = self.kpe.extract_orb(img)
        matches = None
        if self.prev_frame_extracts is not None and self.prev_frame is not None:
            matches = self.matcher.match(descriptors, self.prev_frame_extracts['descriptors'])
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            good_matches = [(key_points[m.queryIdx], self.prev_frame_extracts['key_points'][m.trainIdx]) for m in good_matches]
            # img3 = cv2.drawMatches(img, key_points, self.prev_frame, self.prev_frame_extracts['key_points'],
            #                           good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imshow('matches', img3)

        self.prev_frame_extracts = {'key_points': key_points, 'descriptors': descriptors}
        self.prev_frame = img

        if matches is not None:
            for key_point, prev_key_point in good_matches:
                x, y = np.ravel(key_point.pt).astype(np.int32)
                x_prev, y_prev = np.ravel(prev_key_point.pt).astype(np.int32)
                cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
                cv2.circle(img, (x_prev, y_prev), radius=1, color=(0, 255, 0), thickness=-1)
                cv2.line(img, (x, y), (x_prev, y_prev), color=(255, 0, 0), thickness=1)
        return img


if __name__ == "__main__":
    PATH_TO_VIDEO = 'driving_car_videos/Pexels Videos 1578970.mp4'
    PATH_TO_VIDEO2 = 'driving_car_videos/Pexels Videos 4549.mp4'
    viewer = Viewer()
    cap = cv2.VideoCapture(PATH_TO_VIDEO2)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = viewer.process_frame(frame)
            viewer.show_frame(frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
