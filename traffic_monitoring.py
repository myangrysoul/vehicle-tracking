import math
import os

import tensorflow as tf
import lane_detection
import simple_lane_detection

from absl import app, flags
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from vehicle_detector import VehicleDetector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/good_sample.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './out/video.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'DIVX', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.3, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

feature_extractor_model_file = 'feature_extractor/mars-small128.pb'


def configure_tracker(max_cosine_dist, nn_budget, metric='cosine', max_iou_distance=0.6, max_age=20):
    nearest_neighbor_metric = nn_matching \
        .NearestNeighborDistanceMetric(metric, max_cosine_dist, nn_budget)
    tracker = Tracker(nearest_neighbor_metric, max_iou_distance, max_age)
    return tracker


def extract_features(encoder, frame, boxes, scores, class_names, nms_max):
    features = encoder(frame, boxes)
    detections = [Detection(bbox, score, class_name, feature)
                  for bbox, score, class_name, feature
                  in zip(boxes, scores, class_names, features)
                  ]
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxes, classes, nms_max, scores)
    detections = [detections[i] for i in indices]
    return detections


def get_direction_from_vector(direction, threshold):
    y_dir = direction[1]
    if abs(y_dir) > threshold:
        return simple_lane_detection.TrackDirection.DOWN if y_dir > 0 else simple_lane_detection.TrackDirection.UP
    x_dir = direction[0]
    return simple_lane_detection.TrackDirection.RIGHT if x_dir > 0 else simple_lane_detection.TrackDirection.LEFT


def iterate_over_tracks(tracker: Tracker, frame, line_det: simple_lane_detection.SimpleLaneCounter,
                        direction_vector_threshold: float):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        last_positions = np.array(track.positions, dtype=np.int32)
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        direction = None
        if len(last_positions) > 3:
            cv2.polylines(img=frame, pts=np.int32([last_positions]), isClosed=False, color=color, thickness=3,
                          lineType=cv2.LINE_AA)
            movement_vector = (
                last_positions[:1, 0] - last_positions[-1, 0], last_positions[:1, 1] - last_positions[-1, 1])
            norm = math.sqrt(movement_vector[0] ** 2 + movement_vector[1] ** 2)
            vector_direction = [movement_vector[0] / norm, movement_vector[1] / norm]
            direction = get_direction_from_vector(vector_direction, direction_vector_threshold)
            track.direction = direction
        if not track.counted:
            line_det.process_track(track)

        draw_track(frame, color, bbox, class_name, track.track_id, direction)


def draw_track(frame, color, bbox, class_name, track_id, direction):
    str_direction = ''
    if direction is not None:
        str_direction = direction.value
    cv2.rectangle(img=frame, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])),
                  color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(frame, class_name + "-" + str(track_id) + ' ' + str_direction, (int(bbox[0]), int(bbox[1] - 10)),
                fontFace=cv2.FONT_ITALIC, fontScale=0.5,
                color=(255, 255, 255), thickness=2)


def create_line(x, y):
    return np.array((x, y), dtype=np.int32)


def create_lane_detector(frame_width, frame_height):
    delimiter_y = (frame_height * 3 / 4 + frame_height / 2) / 2
    delimeter = create_line((0, delimiter_y), (frame_width, delimiter_y))
    lane_det = simple_lane_detection.SimpleLaneCounter(delimeter)
    return lane_det


def main(_argv):
    max_cosine_distance = 0.3
    nn_budget = 40
    nms_bbox_max_overlap = 0.7
    tracker = configure_tracker(max_cosine_distance, nn_budget)
    encoder = gdet.create_box_encoder(model_filename=feature_extractor_model_file, batch_size=1)
    input_size = FLAGS.size
    video_path = FLAGS.video
    yolov4_tf_model = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    input_layer = yolov4_tf_model.signatures['serving_default']
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    out = None
    if FLAGS.output:
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (frame_width, frame_height))
    detector = VehicleDetector(input_layer, iou_max=FLAGS.iou, max_output_per_class=60,
                               class_file_name=cfg.YOLO.CLASSES, score_thres=FLAGS.score,
                               original_w=frame_width, original_h=frame_height)
    frame_number = 0
    lane_det = create_lane_detector(frame_width, frame_height)
    while True:
        return_value, frame = video_capture.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_number += 1
        bounding_boxes, scores, names, num_of_objects = detector.inference(frame)
        detections = extract_features(encoder, frame,
                                      bounding_boxes, scores,
                                      names, nms_bbox_max_overlap)
        tracker.predict()
        tracker.update(detections)

        iterate_over_tracks(tracker, frame, lane_det, 0.4)
        lane_det.draw_statistic(frame, frame_width, fps, frame_number)
        lane_det.draw_line(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(result)
        cv2.imshow("Output Video", result)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
