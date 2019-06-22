import os
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image, ImageOps
import numpy as np

from styx_msgs.msg import TrafficLight
from object_classifier import ObjectClassifier


class TLClassifier(object):
    def __init__(self):
        # init object classifier (step 1)
        self.object_classifier = ObjectClassifier()

        # init traffic light color classifier (step 2)

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # step 1
        traffic_light_images = self.object_classifier.get_traffic_light_images(image)

        return TrafficLight.UNKNOWN

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

tl_classifier = TLClassifier()

for root, dirs, files in os.walk("images", topdown=False):
    dirname = os.path.dirname('output_images/')
    os.makedirs(dirname, exist_ok=True)
    for filename in files:

        image = load_img(root + '/' + filename)  # this is a PIL image
        image_np = load_image_into_numpy_array(image)
        print(tl_classifier.get_classification(image_np))

