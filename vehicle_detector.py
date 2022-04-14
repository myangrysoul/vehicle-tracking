import cv2
import numpy as np
import tensorflow as tf

import core.utils as utils


class VehicleDetector:

    def __init__(self, input_layer, iou_max, max_output_per_class,
                 class_file_name, score_thres, original_w, original_h,
                 yolo_input_size=416, allowed_classes=None
                 ):
        if allowed_classes is None:
            self.allowed_classes = ['car', 'bus', 'motorbike', 'truck']
        else:
            self.allowed_classes = allowed_classes
        self.input_layer = input_layer
        self.iou_max = iou_max
        self.input_size = yolo_input_size
        self.max_output_per_class = max_output_per_class
        self.score_thres = score_thres
        self.class_names = utils.read_class_names(class_file_name)
        self.original_h = original_h
        self.original_w = original_w

    def prepare_frame(self, frame):
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        return image_data

    def filter_output(self, boxes, scores, classes, total):
        group = (boxes, scores, classes)
        output_arr = []
        for i, entry in enumerate(group):
            output = group[i].numpy()[0]
            output = output[0:int(total)]
            output_arr.append(output)
        return output_arr[0], output_arr[1], output_arr[2]

    def is_close_to_edge(self, minpoint):
        return self.original_w - minpoint[0] < (self.original_w / 5) or self.original_h - minpoint[1] < (
                self.original_h / 5)

    def post_process(self, num_of_objects, classes, xymin):
        deleted_idx = []
        names = []
        for i in range(num_of_objects):
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            minpoint = xymin[i]
            is_close_to_edge = self.is_close_to_edge(minpoint)
            if class_name not in self.allowed_classes or is_close_to_edge:
                deleted_idx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        return deleted_idx, names

    def inference(self, frame):
        input_data = tf.constant(self.prepare_frame(frame))
        pred_bbox = self.input_layer(input_data)
        for key, val in pred_bbox.items():
            boxes = val[:, :, 0:4]
            scores_pred = val[:, :, 4:]
        boxes, scores, classes, detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(scores_pred, (tf.shape(scores_pred)[0], -1, tf.shape(scores_pred)[-1])),
            max_output_size_per_class=self.max_output_per_class,
            max_total_size=self.max_output_per_class,
            iou_threshold=self.iou_max,
            score_threshold=self.score_thres
        )
        num_of_objects = detections.numpy()[0]
        bounding_boxes, scores, classes = self.filter_output(boxes, scores, classes, num_of_objects)
        bounding_boxes, xymin = utils.format_boxes(bounding_boxes, self.original_h, self.original_w)
        deleted_idx, names = self.post_process(num_of_objects, classes, xymin)
        bounding_boxes = np.delete(bounding_boxes, deleted_idx, axis=0)
        scores = np.delete(scores, deleted_idx, axis=0)
        return bounding_boxes, scores, names, num_of_objects
