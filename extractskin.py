import cv2
from constants import *
import numpy as np
from featureutils import extract_features, extract_filename
from sklearn.externals import joblib
import matplotlib.image as mpimg
import os

def predict_pixels(feature, model):
    print('Predicting pixels')
    X = np.array(feature)
    return model.predict(X)


def filter_with_prediction(image, prediction):
    height, width, _ = image.shape
    mask_h, mask_w, _ = image.shape

    mask_w -= 2*neighbour_radius
    mask_h -= 2*neighbour_radius

    pred_matrix = prediction.reshape((mask_h, mask_w))
    selector = pred_matrix == 'skin'
    selector = selector.astype(np.uint8)
    image_center = image[neighbour_radius:(height- neighbour_radius), neighbour_radius:(width - neighbour_radius), :]
    filtered = cv2.bitwise_and(image_center, image_center, mask=selector)

    return filtered, selector


def extract(image, model):
    print('Extracting image')

    features = extract_features(image)
    prediction = predict_pixels(features, model)

    filtered, mask = filter_with_prediction(image, prediction)
    return filtered, mask


def load_image(image_path):
    img = cv2.imread(image_path)
    return img


def load_model(model_path):
    return joblib.load(model_path)


def save_image(skin, mask, img_path, output):
    skin_filename = os.path.join(output, extract_filename(img_path) + '_skin.png')
    mpimg.imsave(skin_filename, skin)

    mask_filename = os.path.join(output, extract_filename(img_path) + '_mask.png')
    mpimg.imsave(mask_filename, mask)


def extract_skin(image_path, model_path, output_path):
    img = load_image(image_path)
    model = load_model(model_path)

    skin, mask = extract(img, model)
    save_image(skin, mask, image_path, output_path)


def extract_skins(image_paths, model_path, output_path):
    for path in image_paths:
        extract_skin(path, model_path, output_path)

