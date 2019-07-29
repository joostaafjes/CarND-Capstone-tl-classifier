import os
import random
import time

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

class ObjectClassifier:

    def __init__(self):
        # constants
        self.OBJECT_DETECTION_TRAFFIC_LIGHT_CLASS = 10
        self.DETECTION_THRESHOLD = 0.30

        # init object classifier (step 1)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            try:
                with tf.gfile.GFile('models/frozen_inference_graph.pb', 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                    self.session = tf.Session(graph=self.detection_graph)
            except Exception as e:
                print(e)
                exit()

    def get_traffic_light_images(self, image):
        output_dict = self.run_inference_for_single_image(image)
        return self.extract_image_from_boxes(image, output_dict)

    def run_inference_for_single_image(self, image):
        # Get handles to input and output tensors
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.detection_graph.get_tensor_by_name(tensor_name)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Run inference
        start = time.time()
        output_dict = self.session.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image, 0)})
        elapsed = time.time() - start
        # print('inference took:', elapsed, ' seconds')

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        return output_dict

    def extract_image_from_boxes(self, image, output_dict):
        classes = output_dict['detection_classes']
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']

        height, width, _ = image.shape

        output_images = []
        for i in range(len(boxes)):
            confidence = float(scores[i])
            # print('confidence object detection: {}'.format(confidence))
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            # print('bb before: x: {} {} - y: {} {}'.format(xmin, xmax, ymin, ymax))
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            # print('bb after : x: {} {} - y: {} {}'.format(xmin, xmax, ymin, ymax))
            box_width = xmax - xmin
            box_height = ymax - ymin
            box_ratio = float(box_height) / box_width
            if confidence >= self.DETECTION_THRESHOLD and \
               classes[i] == self.OBJECT_DETECTION_TRAFFIC_LIGHT_CLASS and \
               box_width > 21 and \
               box_height > 20 and \
               box_ratio > 1.5:
                print('width: {} - height: {} - ratio: {}'.format(box_width, box_height, box_ratio))
                #
                # extract cropped image from bounding box
                #
                crop_image = image[ymin:ymax, xmin:xmax]
                crop_output_image = Image.fromarray(crop_image)
                output_images.append(crop_output_image)

        return output_images

