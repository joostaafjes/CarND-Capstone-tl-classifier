from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from enum import Enum

class TrafficLight(Enum):
    UNKNOWN=4
    GREEN=2
    YELLOW=1
    RED=0

class ColorClassifier:
    def __init__(self):
        self.IMAGE_SIZE = 32
        self.class_model = load_model('models'+ '/model.h5')
        self.class_graph = tf.get_default_graph()

    def predict_image(self, image):
        x = self.crop_image(image)
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        with self.class_graph.as_default():
            preds = self.class_model.predict_classes(x)
            prob = self.class_model.predict_proba(x)

        return preds[0], prob[0]

    def predict_images(self, images):
        predictions = []
        for image in images:
            pred, prob = self.predict_image(image)
            predictions.append(pred)
        if len(predictions) > 0:
            return max(predictions, key=predictions.count)
        else:
            return TrafficLight.UNKNOWN

    def crop_image(self, img):
        img.thumbnail((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.ANTIALIAS)
        width, height = img.size
        delta_w = self.IMAGE_SIZE - width
        delta_h = self.IMAGE_SIZE - height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding, fill=0)  # fill with black dots

        return img

