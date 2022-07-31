import cv2
import numpy as np

def load_image(path, bgr2rgb = True):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED) if isinstance(path, str) else path
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if bgr2rgb else image
    return image