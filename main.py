import os, os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from key_points_extractor import KeyPointsExtractor
from matcher import Matcher, Tracker
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform


class VisualOdometry(object):
    def __init__(self, camera_params, odometry_poses):
        self.prev_frame_extracts = None
        self.prev_frame = None
        self.E = None
        self.F = None
        self.t = np.zeros(3)
        self.R = np.eye(3)
        self.kpe = KeyPointsExtractor(extractor='orb', detector='fast', num_points=5000, quality=0.001, min_dist=3)
        self.matcher = Matcher()
        self.tracker = Tracker()
        self.w = camera_params['frame_width']
        self.h = camera_params['frame_height']
        # self.f = camera_params['focal_length']
        self._init_camera_intrinsic(fx=718.8560, fy=718.8560, cx=607.1928, cy=185.2157)

        with open(odometry_poses) as f:
            self.odometry_poses = f.readlines()

    def _init_camera_intrinsic(self, fx, fy, cx, cy):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]])

    def _E_from_F(self, F):
        return self.K.T.dot(F).dot(self.K)

    def draw_matches(self, img, matches):
        if img.shape[-1] == 1 or len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for key_point, prev_key_point in matches:
            x, y = key_point.astype(np.int32)
            x_prev, y_prev = prev_key_point.astype(np.int32)
            cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
            # cv2.circle(img, (x_prev, y_prev), radius=1, color=(0, 255, 0), thickness=-1)
            cv2.line(img, (x, y), (x_prev, y_prev), color=(255, 0, 0), thickness=1)
        return img

    @staticmethod
    def show_img(img, img_title):
        cv2.imshow(img_title, img)
        # print(img.shape)

    def match(self, kps2, kps1, descs2, descs1):
        matches = self.matcher.match(descs1, descs2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                kp1 = kps1[m.queryIdx].pt
                kp2 = kps2[m.trainIdx].pt
                good_matches.append((kp1, kp2))
        # img3 = cv2.drawMatches(img, key_points, self.prev_frame, self.prev_frame_extracts['key_points'],
        #                           good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow('matches', img3)
        print("good matches: {}".format(len(good_matches)))
        return good_matches

    def track(self, prev_frame, cur_frame, fts_to_track):
        fts1, fts2 = self.tracker.track(prev_frame, cur_frame, fts_to_track)
        good_matches = [(ft1, ft2) for ft1, ft2 in zip(fts1, fts2)]
        return good_matches

    def getAbsoluteScale(self, frame_id):  # specialized for KITTI odometry dataset
        ss = self.odometry_poses[frame_id - 1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.odometry_poses[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])

        return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))

    def process_frame(self, img, img_index):
        # keypoints extraction and descriptors computation
        # key_points, descriptors = self.kpe.extract(img)
        key_points = self.kpe.extract_fast(img)
        print("keypoints: {}".format(len(key_points)))

        good_matches = None
        if self.prev_frame_extracts is not None and self.prev_frame is not None:
            # matching
            # good_matches = self.match(key_points, self.prev_frame_extracts['key_points'],
            #            descriptors, self.prev_frame_extracts['descriptors'])
            good_matches = self.track(self.prev_frame, img, self.prev_frame_extracts['key_points'])

            # filtering
            good_matches = np.array(good_matches)
            model, inliers = ransac((good_matches[:, 0], good_matches[:, 1]),
                                    FundamentalMatrixTransform, min_samples=8,
                                    residual_threshold=1, max_trials=100)
            good_matches = good_matches[inliers]
            print("Number of keypoints after filtering: {}".format(len(good_matches)))

            self.F = model.params
            self.E = self._E_from_F(self.F)
            # self.E, mask = cv2.findEssentialMat(good_matches[:, :, 0], good_matches[:, :, 1],
            #                                     focal=self.fx, pp=(self.cx, self.cy), method=cv2.RANSAC,
            #                                     prob=0.999, threshold=1.0)

            # pose estimation
            # TODO wtf is going on if write [:, :, 0] instead of [:, 0]??????
            _, R, t, mask = cv2.recoverPose(self.E, good_matches[:, 0], good_matches[:, 1],
                                            focal=self.fx, pp=(self.cx, self.cy))

            absolute_scale = self.getAbsoluteScale(img_index)
            if absolute_scale > 0.1:
                self.t = self.t + absolute_scale * self.R.dot(t.reshape(3))
                self.R = R.dot(self.R)
                # self.t = self.t + self.R.dot(t.reshape(3))
                # self.R = R.dot(self.R)

        # self.prev_frame_extracts = {'key_points': key_points, 'descriptors': descriptors}
        self.prev_frame_extracts = {'key_points': key_points}
        self.prev_frame = img

        if img_index == 0:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if good_matches is not None:
            img = self.draw_matches(img, good_matches)
        return img, self.t


def drawing_iteration(map, coords, frame_index):
    x, y, z = coords
    draw_x, draw_y = int(x) + 300, int(z) + 450
    # true_x, true_y = int(vo.trueX) + 290, int(vo.trueZ) + 90

    cv2.circle(map, (draw_x, draw_y), 1, (255, 0, 0), 2)
    # cv2.circle(map, (true_x, true_y), 1, (0, 0, 255), 2)
    cv2.rectangle(map, pt1=(10, 20), pt2=(600, 60), color=(0, 0, 0), thickness=-1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)

    cv2.putText(map, text, org=(20, 40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                color=(255, 255, 255), thickness=1, lineType=8)


if __name__ == "__main__":
    PATH_TO_VIDEO = 'driving_car_videos/Pexels Videos 1578970.mp4'
    PATH_TO_VIDEO2 = 'driving_car_videos/Pexels Videos 4549.mp4'
    PATH_TO_VIDEO_GEOHOT = 'driving_car_videos/test_countryroad.mp4'
    KITTI_EXAMPLE = 'driving_car_videos/test_kitti984.mp4'

    KITTI_SEQUENCE_PATH = '/home/serhiy/data/KITTI_odometry/data_odometry_gray/dataset/sequences/00/image_0/'
    KITTI_ODOMETRY_POSES = '/home/serhiy/data/KITTI_odometry/data_odometry_poses/dataset/poses/00.txt'
    OUTPUT_VIDEOS_DIR = './output_videos/'

    cap = cv2.VideoCapture(KITTI_EXAMPLE)
    # camera parameters
    W = 1241
    H = 376
    MAP_SIZE = 600

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    driving_video_writer = cv2.VideoWriter(OUTPUT_VIDEOS_DIR + 'Car driving.mp4', fourcc, 15, (W, H))
    mapping_video_writer = cv2.VideoWriter(OUTPUT_VIDEOS_DIR + 'Trajectory drawing.mp4', fourcc, 15, (MAP_SIZE, MAP_SIZE))

    map = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    camera_params = {'frame_width': W, 'frame_height': H}
    vo = VisualOdometry(camera_params, KITTI_ODOMETRY_POSES)

    frame_index = 0

    NUM_FRAMES = len(os.listdir(KITTI_SEQUENCE_PATH))
    # for frame_index in range(NUM_FRAMES):
    for frame_index in range(200):
        frame = cv2.imread(KITTI_SEQUENCE_PATH + str(frame_index).zfill(6)+'.png', 0)

        frame, coords = vo.process_frame(frame, frame_index)
        vo.show_img(frame, 'Car driving video')
        drawing_iteration(map, coords, frame_index)
        vo.show_img(map, 'Car trajectory')

        driving_video_writer.write(frame)
        mapping_video_writer.write(map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         frame, coords = vo.process_frame(frame)
    #         vo.show_img(frame, 'Car driving video')
    #         drawing_iteration(map, coords, frame_index)
    #         vo.show_img(map, 'Car trajectory')
    #         if cv2.waitKey(0) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    #     frame_index += 1

    driving_video_writer.release()
    mapping_video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
