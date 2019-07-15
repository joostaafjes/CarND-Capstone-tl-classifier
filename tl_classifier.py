import os

import numpy as np

from object_classifier import ObjectClassifier
from color_classifier import ColorClassifier

class TLClassifier(object):
    def __init__(self):
        # init object classifier (step 1)
        self.object_classifier = ObjectClassifier()

        # init traffic light color classifier (step 2)
        self.color_classifier = ColorClassifier()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # step 1
        traffic_light_images = self.object_classifier.get_traffic_light_images(image)

        traffic_light_color = self.color_classifier.predict_images(traffic_light_images)

        return traffic_light_color

#
# Below is only used for testing
#
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


