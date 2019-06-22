from typing import overload

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image, ImageOps
from styx_msgs.msg import TrafficLight

class ColorClassifier:
    def __init__(self):
        self.IMAGE_SIZE = 32
        self.class_model = load_model('models'+ '/model.h5')

    def predict_image(self, image):
        x = self.crop_image(image)
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        preds = self.class_model.predict_classes(x)
        prob = self.class_model.predict_proba(x)

        return preds[0], prob[0]

    def predict_images(self, images: list):
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
        delta_w = self.IMAGE_SIZE - img.width
        delta_h = self.IMAGE_SIZE - img.height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding, fill=0)  # fill with black dots

        return img

