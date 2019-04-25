import cv2

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300


def process_frame(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    cv2.imshow('Car driving video', img)
    cv2.waitKey(1)
    print(img.shape)


if __name__ == "__main__":
    cap = cv2.VideoCapture('driving_car_videos/Pexels Videos 1578970.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break