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
        self.DETECTION_THRESHOLD = 0.3

        # init object classifier (step 1)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            try:
                with tf.gfile.GFile('models/frozen_inference_graph.pb', 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            except Exception as e:
                print(e)
                exit()

    def get_traffic_light_images(self, image):
        output_dict = self.run_inference_for_single_image(image)
        return self.extract_image_from_boxes(image, output_dict)

    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                start = time.time()
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})
                elapsed = time.time() - start
                print('inference took:', elapsed, ' seconds')

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

        tf_boxes = []
        output_images = []
        os.makedirs('output_images/', exist_ok=True)
        for i in range(len(boxes)):
            confidence = float(scores[i])
            if confidence >= self.DETECTION_THRESHOLD and classes[i] == self.OBJECT_DETECTION_TRAFFIC_LIGHT_CLASS:
                tf_boxes.append(boxes[i])
                confidence = float(scores[i])
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                ymin = int(ymin * height)
                ymax = int(ymax * height)
                xmin = int(xmin * width)
                xmax = int(xmax * width)
                #
                # extract cropped image from bounding box
                #
                crop_image = image[ymin:ymax, xmin:xmax]
                crop_output_image = Image.fromarray(crop_image)
                crop_output_image.save('output_images/' + str(random.randint(1000, 2000)) + '-' + str(i) + '.png')
                output_images.append(crop_output_image)
            else:
                if classes[i] == self.OBJECT_DETECTION_TRAFFIC_LIGHT_CLASS:
                    print(' confidence too low:', confidence)

        print('traffic lights found:', len(output_images))

        return output_images

