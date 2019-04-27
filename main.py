import cv2
import numpy as np
from key_points_extractor import KeyPointsExtractor

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300

kpe = KeyPointsExtractor()


def show_frame(img):
    cv2.imshow('Car driving video', img)
    print(img.shape)


def process_frame(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    key_points = kpe.extract(img)
    for key_point in np.int0(key_points):
        x, y = np.ravel(key_point)
        cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
    return img


if __name__ == "__main__":
    cap = cv2.VideoCapture('driving_car_videos/Pexels Videos 1578970.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = process_frame(frame)
            show_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()